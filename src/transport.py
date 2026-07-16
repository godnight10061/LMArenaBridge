"""Userscript-proxy and Chrome transport support for LMArenaBridge."""

import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

import httpx
from fastapi import HTTPException, Request

from . import constants as _constants
from .browser_utils import _cancel_background_task, _consume_background_task_exception
from .browser_window import browser_window_mode, chrome_window_args

HTTPStatus = _constants.HTTPStatus


def _m():
    """Late import of main module so tests can patch main.X and it is reflected here."""
    from . import main
    return main


class BrowserFetchStreamResponse:
    def __init__(
        self,
        status_code: int,
        headers: Optional[dict],
        text: str = "",
        method: str = "POST",
        url: str = "",
        lines_queue: Optional[asyncio.Queue] = None,
        done_event: Optional[asyncio.Event] = None,
    ):
        self.status_code = int(status_code or 0)
        self.headers = headers or {}
        self._text = text or ""
        self._method = str(method or "POST")
        self._url = str(url or "")
        self._lines_queue = lines_queue
        self._done_event = done_event

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    async def aclose(self) -> None:
        return None

    @property
    def text(self) -> str:
        if self._lines_queue is not None and not self._text:
            # This is a bit dangerous in a property because it's sync, 
            # but BrowserFetchStreamResponse is used in contexts where .text is expected.
            # However, in this codebase, we mostly use await aread() or aiter_lines().
            # Let's make it safe by NOT buffering here, but informing that it might be empty
            # OR better: the codebase should use await aread().
            return self._text
        return self._text

    async def aiter_lines(self):
        if self._lines_queue is not None:
            # Streaming mode
            while True:
                if self._done_event and self._done_event.is_set() and self._lines_queue.empty():
                    break
                try:
                    # Brief timeout to check done_event occasionally
                    line = await asyncio.wait_for(self._lines_queue.get(), timeout=1.0)
                    if line is None: # Sentinel for EOF
                        break
                    yield line
                except asyncio.TimeoutError:
                    continue
        else:
            # Buffered mode
            for line in self._text.splitlines():
                yield line

    async def aread(self) -> bytes:
        if self._lines_queue is not None:
            # If we try to read the full body of a streaming response, we buffer it all first.
            collected = []
            async for line in self.aiter_lines():
                collected.append(line)
            self._text = "\n".join(collected)
            self._lines_queue = None
            self._done_event = None
        return self._text.encode("utf-8")

    def raise_for_status(self) -> None:
        if self.status_code == 0 or self.status_code >= 400:
            request = httpx.Request(self._method, self._url or "https://arena.ai/")
            response = httpx.Response(self.status_code or 502, request=request, content=self._text.encode("utf-8"))
            raise httpx.HTTPStatusError(f"HTTP {self.status_code}", request=request, response=response)




def _touch_userscript_poll(now: Optional[float] = None) -> None:
    """
    Update userscript-proxy "last seen" timestamps.

    Keep the current and legacy timestamps in sync so proxy availability is
    detected consistently.
    """
    ts = float(now if now is not None else _m().time.time())
    _m().USERSCRIPT_PROXY_LAST_POLL_AT = ts
    # Legacy timestamp used by older code paths/tests.
    _m().last_userscript_poll = ts


def _get_userscript_proxy_queue() -> asyncio.Queue:
    if _m()._USERSCRIPT_PROXY_QUEUE is None:
        _m()._USERSCRIPT_PROXY_QUEUE = asyncio.Queue()
    return _m()._USERSCRIPT_PROXY_QUEUE


def _userscript_proxy_is_active(config: Optional[dict] = None) -> bool:
    cfg = config or _m().get_config()
    poll_timeout = 25
    try:
        poll_timeout = int(cfg.get("userscript_proxy_poll_timeout_seconds", 25))
    except Exception:
        poll_timeout = 25
    active_window = max(10, min(poll_timeout + 10, 90))
    # Back-compat: some callers/tests still update the legacy `_m().last_userscript_poll` timestamp.
    try:
        last = max(float(_m().USERSCRIPT_PROXY_LAST_POLL_AT or 0.0), float(_m().last_userscript_poll or 0.0))
    except Exception:
        last = float(_m().USERSCRIPT_PROXY_LAST_POLL_AT or 0.0)
    try:
        delta = float(_m().time.time()) - float(last)
    except Exception:
        delta = 999999.0
    # Guard against clock skew / patched clocks in tests: a "last poll" timestamp in the future is not active.
    if delta < 0:
        return False
    return delta <= float(active_window)


def _userscript_proxy_check_secret(request: Request) -> None:
    cfg = _m().get_config()
    secret = str(cfg.get("userscript_proxy_secret") or "").strip()
    if secret and request.headers.get("X-LMBridge-Secret") != secret:
        raise HTTPException(status_code=401, detail="Invalid userscript proxy secret")


def _cleanup_userscript_proxy_jobs(config: Optional[dict] = None) -> None:
    cfg = config or _m().get_config()
    ttl_seconds = 90
    try:
        ttl_seconds = int(cfg.get("userscript_proxy_job_ttl_seconds", 90))
    except Exception:
        ttl_seconds = 90
    ttl_seconds = max(10, min(ttl_seconds, 600))

    now = _m().time.time()
    expired: list[str] = []
    for job_id, job in list(_m()._USERSCRIPT_PROXY_JOBS.items()):
        created_at = float(job.get("created_at") or 0.0)
        done = bool(job.get("done"))
        picked_up = False
        try:
            picked_up_event = job.get("picked_up_event")
            if isinstance(picked_up_event, asyncio.Event):
                picked_up = bool(picked_up_event.is_set())
        except Exception:
            picked_up = False
        if done and (now - created_at) > ttl_seconds:
            expired.append(job_id)
        # If a job was never picked up, expire it even if not marked done (stuck/abandoned queue entries).
        elif (not done) and (not picked_up) and (now - created_at) > ttl_seconds:
            expired.append(job_id)
        # Safety: even if picked up, expire if it's been in-flight for too long (e.g. browser crash).
        elif (not done) and picked_up and (now - created_at) > (ttl_seconds * 5):
            expired.append(job_id)
    for job_id in expired:
        _m()._USERSCRIPT_PROXY_JOBS.pop(job_id, None)


def _mark_userscript_proxy_inactive() -> None:
    """
    Mark the userscript-proxy as inactive.

    Do this when we detect proxy health/timeouts so strict-model routing stops preferring a proxy that is not
    responding. The proxy becomes active again once a real poll/push updates the timestamps.
    """
    _m().USERSCRIPT_PROXY_LAST_POLL_AT = 0.0
    _m().last_userscript_poll = 0.0


async def _finalize_userscript_proxy_job(job_id: str, *, error: Optional[str] = None, remove: bool = False) -> None:
    """
    Finalize a userscript-proxy job without touching proxy "last seen" timestamps.

    This is intentionally separate from `push_proxy_chunk()`: server-side timeouts must not keep the proxy
    marked as "active" because that would route future requests back into a dead proxy.
    """
    jid = str(job_id or "").strip()
    if not jid:
        return
    job = _m()._USERSCRIPT_PROXY_JOBS.get(jid)
    if not isinstance(job, dict):
        return

    if error and not job.get("error"):
        job["error"] = str(error)

    if job.get("_finalized"):
        if remove:
            _m()._USERSCRIPT_PROXY_JOBS.pop(jid, None)
        return

    job["_finalized"] = True
    job["done"] = True

    done_event = job.get("done_event")
    if isinstance(done_event, asyncio.Event):
        done_event.set()
    status_event = job.get("status_event")
    if isinstance(status_event, asyncio.Event):
        status_event.set()

    q = job.get("lines_queue")
    if isinstance(q, asyncio.Queue):
        try:
            q.put_nowait(None)
        except Exception:
            try:
                await q.put(None)
            except Exception:
                pass

    if remove:
        _m()._USERSCRIPT_PROXY_JOBS.pop(jid, None)


class UserscriptProxyStreamResponse:
    def __init__(self, job_id: str, timeout_seconds: int = 120):
        self.job_id = str(job_id)
        self._status_code: int = 200
        self._headers: dict = {}
        self._timeout_seconds = int(timeout_seconds or 120)
        self._method = "POST"
        self._url = "https://arena.ai/"

    @property
    def status_code(self) -> int:
        # Do not rely on a snapshot: proxy workers can report status after `__aenter__` returns.
        job = _m()._USERSCRIPT_PROXY_JOBS.get(self.job_id)
        if isinstance(job, dict):
            status = job.get("status_code")
            if isinstance(status, int):
                return int(status)
        return int(self._status_code or 0)

    @status_code.setter
    def status_code(self, value: int) -> None:
        try:
            self._status_code = int(value)
        except Exception:
            self._status_code = 0

    @property
    def headers(self) -> dict:
        job = _m()._USERSCRIPT_PROXY_JOBS.get(self.job_id)
        if isinstance(job, dict):
            headers = job.get("headers")
            if isinstance(headers, dict):
                return headers
        return self._headers

    @headers.setter
    def headers(self, value: dict) -> None:
        self._headers = value if isinstance(value, dict) else {}

    async def __aenter__(self):
        job = _m()._USERSCRIPT_PROXY_JOBS.get(self.job_id)
        if not isinstance(job, dict):
            self.status_code = 503
            return self
        # Give the proxy a short window to report the upstream HTTP status before we snapshot it, but don't
        # block if it has already started streaming lines (some proxy implementations report status late).
        status_event = job.get("status_event")
        should_wait_status = False
        if isinstance(status_event, asyncio.Event) and not status_event.is_set():
            should_wait_status = True
            try:
                if job.get("error"):
                    should_wait_status = False
            except Exception:
                pass
            done_event = job.get("done_event")
            if isinstance(done_event, asyncio.Event) and done_event.is_set():
                should_wait_status = False
            q = job.get("lines_queue")
            if isinstance(q, asyncio.Queue) and not q.empty():
                should_wait_status = False

        if should_wait_status:
            try:
                await asyncio.wait_for(
                    status_event.wait(),
                    timeout=min(15.0, float(max(1, self._timeout_seconds))),
                )
            except Exception:
                pass
        self._method = str(job.get("method") or "POST")
        self._url = str(job.get("url") or self._url)
        status = job.get("status_code")
        if isinstance(status, int):
            self.status_code = int(status)
        headers = job.get("headers")
        if isinstance(headers, dict):
            self.headers = headers
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        await self.aclose()
        return False

    async def aclose(self) -> None:
        # Do not eagerly delete completed jobs here.
        #
        # Callers may need to inspect `status_code`/`error` after the context exits (e.g. to decide whether to
        # fall back to Chrome fetch). Jobs are pruned by `_cleanup_userscript_proxy_jobs()` on a short TTL.
        return None

    async def aiter_lines(self):
        job = _m()._USERSCRIPT_PROXY_JOBS.get(self.job_id)
        if not isinstance(job, dict):
            return
        q = job.get("lines_queue")
        done_event = job.get("done_event")
        if not isinstance(q, asyncio.Queue) or not isinstance(done_event, asyncio.Event):
            return

        deadline = _m().time.time() + float(max(5, self._timeout_seconds))
        while True:
            if done_event.is_set() and q.empty():
                break
            remaining = deadline - _m().time.time()
            if remaining <= 0:
                job["error"] = job.get("error") or "userscript proxy timeout"
                job["done"] = True
                done_event.set()
                break
            timeout = max(0.25, min(2.0, remaining))
            try:
                item = await asyncio.wait_for(q.get(), timeout=timeout)
            except asyncio.TimeoutError:
                continue
            if item is None:
                break
            yield str(item)

    async def aread(self) -> bytes:
        job = _m()._USERSCRIPT_PROXY_JOBS.get(self.job_id)
        if not isinstance(job, dict):
            return b""
        q = job.get("lines_queue")
        if not isinstance(q, asyncio.Queue):
            return b""
        items: list[str] = []
        try:
            while True:
                item = q.get_nowait()
                if item is None:
                    break
                items.append(str(item))
        except Exception:
            pass
        return ("\n".join(items)).encode("utf-8")

    def raise_for_status(self) -> None:
        job = _m()._USERSCRIPT_PROXY_JOBS.get(self.job_id)
        if isinstance(job, dict) and job.get("error"):
            request = httpx.Request(self._method, self._url)
            response = httpx.Response(503, request=request, content=str(job.get("error")).encode("utf-8"))
            raise httpx.HTTPStatusError("Userscript proxy error", request=request, response=response)
        status = int(self.status_code or 0)
        if status == 0 or status >= 400:
            request = httpx.Request(self._method, self._url)
            response = httpx.Response(status or 502, request=request)
            raise httpx.HTTPStatusError(f"HTTP {status}", request=request, response=response)


_LMARENA_ORIGIN = "https://lmarena.ai"
_ARENA_ORIGIN = "https://arena.ai"
_ARENA_HOST_TO_ORIGIN = {
    "lmarena.ai": _LMARENA_ORIGIN,
    "www.lmarena.ai": _LMARENA_ORIGIN,
    "arena.ai": _ARENA_ORIGIN,
    "www.arena.ai": _ARENA_ORIGIN,
}


def _detect_arena_origin(url: Optional[str] = None) -> str:
    """
    Return the canonical origin (https://lmarena.ai or https://arena.ai) for a URL-like string.

    LMArena has historically used both domains. Browser automation can land on `arena.ai` even when the backend
    constructs `https://lmarena.ai/...` URLs, so cookie ops must follow the actual origin.
    """
    text = str(url or "").strip()
    if not text:
        return _ARENA_ORIGIN
    try:
        parts = urlsplit(text)
    except Exception:
        parts = None

    host = ""
    if parts and parts.scheme and parts.netloc:
        host = str(parts.netloc or "").split("@")[-1].split(":")[0].lower()
    if not host:
        host = text.split("/")[0].split("@")[-1].split(":")[0].lower()
    return _ARENA_HOST_TO_ORIGIN.get(host, _ARENA_ORIGIN)


def _arena_origin_candidates(url: Optional[str] = None) -> list[str]:
    """Return `[primary, secondary]` origins, preferring the detected origin but always including both."""
    primary = _detect_arena_origin(url)
    secondary = _LMARENA_ORIGIN if primary == _ARENA_ORIGIN else _ARENA_ORIGIN
    return [primary, secondary]


def _arena_auth_cookie_specs(token: str, *, page_url: Optional[str] = None) -> list[dict]:
    """
    Build host-only `arena-auth-prod-v1` cookie specs for both arena.ai and lmarena.ai.

    Using `url` (instead of `domain`) more closely matches how the site stores this cookie (host-only).
    """
    value = str(token or "").strip()
    if not value:
        return []
    specs: list[dict] = []
    for origin in _arena_origin_candidates(page_url):
        specs.append({"name": "arena-auth-prod-v1", "value": value, "url": origin, "path": "/"})
    return specs


def _provisional_user_id_cookie_specs(provisional_user_id: str, *, page_url: Optional[str] = None) -> list[dict]:
    """
    Build `provisional_user_id` cookie specs for both origins.

    LMArena sometimes stores this cookie as host-only and sometimes as a domain cookie; keep both in sync.
    """
    value = str(provisional_user_id or "").strip()
    if not value:
        return []
    specs: list[dict] = []
    for origin in _arena_origin_candidates(page_url):
        specs.append({"name": "provisional_user_id", "value": value, "url": origin, "path": "/"})
    for domain in (".lmarena.ai", ".arena.ai"):
        # When using domain, do NOT include path - they're mutually exclusive in Playwright
        specs.append({"name": "provisional_user_id", "value": value, "domain": domain})

    return specs


async def _get_arena_context_cookies(context, *, page_url: Optional[str] = None) -> list[dict]:
    """
    Fetch cookies for both arena.ai and lmarena.ai from a Playwright browser context.
    """
    urls = _arena_origin_candidates(page_url)
    all_raw: list[dict] = []
    for url in urls:
        try:
            chunk = await context.cookies(url)
            if isinstance(chunk, list):
                all_raw.extend(chunk)
        except Exception:
            pass
    
    seen: set[tuple[str, str, str]] = set()
    merged: list[dict] = []
    for c in all_raw:
        key = (str(c.get("name") or ""), str(c.get("domain") or ""), str(c.get("path") or ""))
        if key in seen:
            continue
        seen.add(key)
        merged.append(c)
    return merged


def _normalize_userscript_proxy_url(url: str) -> str:
    """
    Convert LMArena absolute URLs into same-origin paths for in-page fetch.

    The proxy page may be on arena.ai (after redirect from lmarena.ai); absolute
    cross-origin URLs can cause browser fetch to reject with a generic NetworkError (CORS).
    """
    text = str(url or "").strip()
    if not text:
        return ""
    if text.startswith("/"):
        return text
    try:
        parts = urlsplit(text)
    except Exception:
        return text
    if not parts.scheme or not parts.netloc:
        return text
    host = str(parts.netloc or "").split("@")[-1].split(":")[0].lower()
    if host not in {"lmarena.ai", "www.lmarena.ai", "arena.ai", "www.arena.ai"}:
        return text
    path = parts.path or "/"
    if parts.query:
        path = f"{path}?{parts.query}"
    return path


async def fetch_lmarena_stream_via_userscript_proxy(
    http_method: str,
    url: str,
    payload: dict,
    timeout_seconds: int = 120,
    auth_token: str = "",
) -> Optional[UserscriptProxyStreamResponse]:
    config = _m().get_config()
    _cleanup_userscript_proxy_jobs(config)

    job_id = str(uuid.uuid4())
    lines_queue: asyncio.Queue = asyncio.Queue()
    done_event: asyncio.Event = asyncio.Event()
    status_event: asyncio.Event = asyncio.Event()
    picked_up_event: asyncio.Event = asyncio.Event()

    proxy_url = _normalize_userscript_proxy_url(str(url))
    sitekey, action = _m().get_recaptcha_settings(config)
    job = {
        "created_at": _m().time.time(),
        "job_id": job_id,
        # Job lifecycle markers used by the server-side stream handler to apply timeouts correctly.
        # - phase: queued -> picked_up -> signup -> fetch
        # - picked_up_at_monotonic: set when any proxy worker/poller claims the job
        # - upstream_started_at_monotonic: set when the proxy begins processing the request (may include preflight)
        # - upstream_fetch_started_at_monotonic: set when the upstream HTTP fetch is initiated (after preflight)
        "phase": "queued",
        "picked_up_at_monotonic": None,
        "upstream_started_at_monotonic": None,
        "upstream_fetch_started_at_monotonic": None,
        "url": str(url),
        "method": str(http_method or "POST"),
        # Per-request auth token (do not mutate persisted config). The proxy worker uses this to set
        # the `arena-auth-prod-v1` cookie before executing the in-page fetch.
        "arena_auth_token": str(auth_token or "").strip(),
        "recaptcha_sitekey": sitekey,
        "recaptcha_action": action,
        "payload": {
            "url": proxy_url or str(url),
            "method": str(http_method or "POST"),
            "headers": {"Content-Type": "text/plain;charset=UTF-8"},
            "body": json.dumps(payload) if payload is not None else "",
        },
        "lines_queue": lines_queue,
        "done_event": done_event,
        "status_event": status_event,
        "picked_up_event": picked_up_event,
        "done": False,
        "status_code": 200,
        "headers": {},
        "error": None,
    }
    _m()._USERSCRIPT_PROXY_JOBS[job_id] = job
    await _get_userscript_proxy_queue().put(job_id)
    return UserscriptProxyStreamResponse(job_id, timeout_seconds=timeout_seconds)


async def fetch_lmarena_stream_via_chrome(
    http_method: str,
    url: str,
    payload: dict,
    auth_token: str,
    timeout_seconds: int = 120,
    headless: bool = False, # Default to Headful for better reliability
    max_recaptcha_attempts: int = 3,
) -> Optional[BrowserFetchStreamResponse]:
    """
    Fallback transport: perform the stream request via in-browser fetch (Chrome/Edge via Playwright).
    This tends to align cookies/UA/TLS with what LMArena expects and can reduce reCAPTCHA flakiness.
    """
    try:
        from playwright.async_api import async_playwright  # type: ignore
    except Exception:
        return None

    config = _m().get_config()
    recaptcha_sitekey, recaptcha_action = _m().get_recaptcha_settings(config)

    cookie_store = config.get("browser_cookies")
    cookie_map: dict[str, str] = {}
    if isinstance(cookie_store, dict):
        for name, value in cookie_store.items():
            if not name or not value:
                continue
            cookie_map[str(name)] = str(value)

    # Prefer the Chrome persistent profile's own Cloudflare/BM cookies when present.
    # We only inject missing cookies to avoid overwriting a valid cf_clearance/__cf_bm with stale values
    # coming from a different browser fingerprint.
    cf_clearance = str(config.get("cf_clearance") or cookie_map.get("cf_clearance") or "").strip()
    cf_bm = str(config.get("cf_bm") or cookie_map.get("__cf_bm") or "").strip()
    cfuvid = str(config.get("cfuvid") or cookie_map.get("_cfuvid") or "").strip()
    provisional_user_id = str(config.get("provisional_user_id") or cookie_map.get("provisional_user_id") or "").strip()
    grecaptcha_cookie = str(cookie_map.get("_GRECAPTCHA") or "").strip()

    desired_cookies: list[dict] = []
    # When using domain, do NOT include path - they're mutually exclusive in Playwright
    # We define cookies for both domains for seamless migration
    cookie_definitions = [
        (cf_clearance, "cf_clearance"),
        (cf_bm, "__cf_bm"),
        (cfuvid, "_cfuvid"),
        (provisional_user_id, "provisional_user_id"),
        (grecaptcha_cookie, "_GRECAPTCHA"),
    ]
    for value, name in cookie_definitions:
        if value:
            for _domain in (".lmarena.ai", ".arena.ai"):
                desired_cookies.append({"name": name, "value": value, "domain": _domain})
    
    if auth_token:
        desired_cookies.extend(_arena_auth_cookie_specs(auth_token))

    user_agent = _m().normalize_user_agent_value(config.get("user_agent"))

    fetch_url = _normalize_userscript_proxy_url(url)

    def _is_recaptcha_validation_failed(status: int, text: object) -> bool:
        if int(status or 0) != HTTPStatus.FORBIDDEN:
            return False
        if not isinstance(text, str) or not text:
            return False
        try:
            body = json.loads(text)
        except Exception:
            return False
        return isinstance(body, dict) and body.get("error") == "recaptcha validation failed"

    max_recaptcha_attempts = max(1, min(int(max_recaptcha_attempts), 10))

    profile_dir = Path(_m().CONFIG_FILE).with_name("chrome_grecaptcha")
    window_args = [] if headless else chrome_window_args(browser_window_mode(config))
    async with async_playwright() as p:
        chrome_path = _m().find_chrome_executable(
            config,
            playwright_executable=p.chromium.executable_path,
        )
        if not chrome_path:
            return None
        context = await p.chromium.launch_persistent_context(
            user_data_dir=str(profile_dir),
            executable_path=chrome_path,
            headless=bool(headless),
            user_agent=user_agent or None,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-first-run",
                "--no-default-browser-check",
                *window_args,
            ],
        )
        try:
            # Small stealth tweak: reduces bot-detection surface for reCAPTCHA v3 scoring.
            try:
                await context.add_init_script(
                    "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
                )
            except Exception:
                pass

            if desired_cookies:
                try:
                    existing_names: set[str] = set()
                    try:
                        existing = await _get_arena_context_cookies(context)
                        for c in existing or []:
                            name = c.get("name")
                            if name:
                                existing_names.add(str(name))
                    except Exception:
                        existing_names = set()

                    cookies_to_add: list[dict] = []
                    for c in desired_cookies:
                        name = str(c.get("name") or "")
                        if not name:
                            continue
                        # Always ensure the auth cookie matches the selected upstream token.
                        if name == "arena-auth-prod-v1":
                            cookies_to_add.append(c)
                            continue

                        # Do NOT overwrite/inject Cloudflare or reCAPTCHA cookies in the persistent profile.
                        # The profile manages these itself; injecting stale ones from config causes 403s.
                        if name in ("cf_clearance", "__cf_bm", "_GRECAPTCHA"):
                            continue

                        # Avoid overwriting existing Cloudflare/session cookies in the persistent profile.
                        if name in existing_names:
                            continue
                        cookies_to_add.append(c)

                    if cookies_to_add:
                        await context.add_cookies(cookies_to_add)
                except Exception:
                    pass

            page = await context.new_page()
            await page.goto("https://arena.ai/?mode=direct", wait_until="domcontentloaded", timeout=120000)

            # Best-effort: if we land on a Cloudflare challenge page, try clicking Turnstile before minting tokens.
            try:
                for i in range(10): # Up to 30 seconds
                    title = await page.title()
                    if "Just a moment" not in title:
                        break
                    _m().debug_print(f"  ⏳ Waiting for Cloudflare challenge in Chrome... (attempt {i+1}/10)")
                    await _m().click_turnstile(page)
                    await asyncio.sleep(3)
                try:
                    await page.wait_for_load_state("domcontentloaded", timeout=15000)
                except Exception:
                    pass
            except Exception:
                pass

            # Light warm-up (often improves reCAPTCHA v3 score vs firing immediately).
            try:
                await page.mouse.move(100, 100)
                await asyncio.sleep(0.5)
                await page.mouse.wheel(0, 200)
                await asyncio.sleep(1)
                await page.mouse.move(200, 300)
                await asyncio.sleep(0.5)
                await page.mouse.wheel(0, 300)
                await asyncio.sleep(2) # Reduced "Human" pause for faster response
            except Exception:
                pass

            # Persist updated cookies/UA from this browser context (helps keep auth + cf cookies fresh).
            try:
                fresh_cookies = await _get_arena_context_cookies(context, page_url=str(getattr(page, "url", "") or ""))
                _m()._capture_ephemeral_arena_auth_token_from_cookies(fresh_cookies)
                try:
                    ua_now = await page.evaluate("() => navigator.userAgent")
                except Exception:
                    ua_now = user_agent
                if _m()._upsert_browser_session_into_config(config, fresh_cookies, user_agent=ua_now):
                    _m().save_config(config)
            except Exception:
                pass

            async def _mint_recaptcha_v3_token() -> Optional[str]:
                await page.wait_for_function(
                    "window.grecaptcha && ("
                    "(window.grecaptcha.enterprise && typeof window.grecaptcha.enterprise.execute === 'function') || "
                    "typeof window.grecaptcha.execute === 'function'"
                    ")",
                    timeout=60000,
                )
                token = await page.evaluate(
                    """({sitekey, action}) => new Promise((resolve, reject) => {
                      const g = (window.grecaptcha?.enterprise && typeof window.grecaptcha.enterprise.execute === 'function')
                        ? window.grecaptcha.enterprise
                        : window.grecaptcha;
                      if (!g || typeof g.execute !== 'function') return reject('NO_GRECAPTCHA');
                      try {
                        g.execute(sitekey, { action }).then(resolve).catch((err) => reject(String(err)));
                      } catch (e) { reject(String(e)); }
                    })""",
                    {"sitekey": recaptcha_sitekey, "action": recaptcha_action},
                )
                if isinstance(token, str) and token:
                    return token
                return None

            async def _mint_recaptcha_v2_token() -> Optional[str]:
                """
                Best-effort: try to obtain a reCAPTCHA Enterprise v2 token (checkbox/invisible).
                LMArena falls back to v2 when v3 scoring is rejected.
                """
                try:
                    await page.wait_for_function(
                        "window.grecaptcha && window.grecaptcha.enterprise && typeof window.grecaptcha.enterprise.render === 'function'",
                        timeout=60000,
                    )
                except Exception:
                    return None

                token = await page.evaluate(
                    """({sitekey, timeoutMs}) => new Promise((resolve, reject) => {
                      const g = window.grecaptcha?.enterprise;
                      if (!g || typeof g.render !== 'function') return reject('NO_GRECAPTCHA_V2');
                      let settled = false;
                      const done = (fn, arg) => {
                        if (settled) return;
                        settled = true;
                        fn(arg);
                      };
                      try {
                        const el = document.createElement('div');
                        el.style.cssText = 'position:fixed;left:-9999px;top:-9999px;width:1px;height:1px;';
                        document.body.appendChild(el);
                        const timer = setTimeout(() => done(reject, 'V2_TIMEOUT'), timeoutMs || 60000);
                        const wid = g.render(el, {
                          sitekey,
                          size: 'invisible',
                          callback: (tok) => { clearTimeout(timer); done(resolve, tok); },
                          'error-callback': () => { clearTimeout(timer); done(reject, 'V2_ERROR'); },
                        });
                        try {
                          if (typeof g.execute === 'function') g.execute(wid);
                        } catch (e) {}
                      } catch (e) {
                        done(reject, String(e));
                      }
                    })""",
                    {"sitekey": _m().RECAPTCHA_V2_SITEKEY, "timeoutMs": 60000},
                )
                if isinstance(token, str) and token:
                    return token
                return None

            lines_queue: asyncio.Queue = asyncio.Queue()
            done_event: asyncio.Event = asyncio.Event()

            # Buffer for splitlines handling in browser
            async def _report_chunk(source, line: str):
                if line and line.strip():
                    await lines_queue.put(line)

            await page.expose_binding("reportChunk", _report_chunk)

            fetch_script = """async ({url, method, body, extraHeaders, timeoutMs}) => {
              const controller = new AbortController();
              const timer = setTimeout(() => controller.abort('timeout'), timeoutMs);
              try {
                const res = await fetch(url, {
                  method,
                  headers: { 
                    'content-type': 'text/plain;charset=UTF-8',
                    ...extraHeaders
                  },
                  body,
                  credentials: 'include',
                  signal: controller.signal,
                });
                const headers = {};
                try {
                  if (res.headers && typeof res.headers.forEach === 'function') {
                    res.headers.forEach((value, key) => { headers[key] = value; });
                  }
                } catch (e) {}

                // Send initial status and headers
                if (window.reportChunk) {
                    await window.reportChunk(JSON.stringify({ __type: 'meta', status: res.status, headers }));
                }

                if (res.body) {
                  const reader = res.body.getReader();
                  const decoder = new TextDecoder();
                  let buffer = '';
                  while (true) {
                    const { value, done } = await reader.read();
                    if (value) buffer += decoder.decode(value, { stream: true });
                    if (done) buffer += decoder.decode();
                    
                    const parts = buffer.split(/\\r?\\n/);
                    buffer = parts.pop() || '';
                    for (const line of parts) {
                        if (line.trim() && window.reportChunk) {
                            await window.reportChunk(line);
                        }
                    }
                    if (done) break;
                  }
                  if (buffer.trim() && window.reportChunk) {
                      await window.reportChunk(buffer);
                  }
                } else {
                  const text = await res.text();
                  if (window.reportChunk) await window.reportChunk(text);
                }
                return { __streaming: true };
              } catch (e) {
                return { status: 502, headers: {}, text: 'FETCH_ERROR:' + String(e) };
              } finally {
                clearTimeout(timer);
              }
            }"""

            result: dict = {"status": 0, "headers": {}, "text": ""}
            for attempt in range(max_recaptcha_attempts):
                # Clear queue for each attempt
                while not lines_queue.empty():
                    lines_queue.get_nowait()
                done_event.clear()

                current_recaptcha_token = ""
                # Mint a new token if not already present or if it's empty
                has_v2 = isinstance(payload, dict) and bool(payload.get("recaptchaV2Token"))
                has_v3 = isinstance(payload, dict) and bool(payload.get("recaptchaV3Token"))
                
                if isinstance(payload, dict) and not has_v2 and (attempt > 0 or not has_v3):
                    current_recaptcha_token = await _mint_recaptcha_v3_token()
                    if current_recaptcha_token:
                        payload["recaptchaV3Token"] = current_recaptcha_token

                extra_headers = {}
                token_for_headers = current_recaptcha_token
                if not token_for_headers and isinstance(payload, dict):
                    token_for_headers = str(payload.get("recaptchaV3Token") or "").strip()
                if token_for_headers:
                    extra_headers["X-Recaptcha-Token"] = token_for_headers
                    extra_headers["X-Recaptcha-Action"] = recaptcha_action

                body = json.dumps(payload) if payload is not None else ""
                
                # Start fetch task
                fetch_task = asyncio.create_task(page.evaluate(
                    fetch_script,
                    {
                        "url": fetch_url,
                        "method": http_method,
                        "body": body,
                        "extraHeaders": extra_headers,
                        "timeoutMs": int(timeout_seconds * 1000),
                    },
                ))

                # Wait for initial meta (status/headers) OR task completion
                meta = None
                while not fetch_task.done():
                    try:
                        # Peek at queue for meta
                        item = await asyncio.wait_for(lines_queue.get(), timeout=0.1)
                        if isinstance(item, str) and item.startswith('{"__type":"meta"'):
                            meta = json.loads(item)
                            break
                        else:
                            # Not meta, put it back (though it shouldn't happen before meta)
                            # Actually, LMArena might send data immediately.
                            # If it's not meta, it's likely already content.
                            # For safety, let's assume if it doesn't look like meta, status is 200.
                            if not item.startswith('{"__type":"meta"'):
                                await lines_queue.put(item)
                                meta = {"status": 200, "headers": {}}
                                break
                    except asyncio.TimeoutError:
                        continue
                
                if fetch_task.done() and meta is None:
                    # Give a brief moment for meta chunk to arrive in the queue (race condition)
                    try:
                        # Check if there's anything in the queue that might be the meta chunk
                        try:
                            item = lines_queue.get_nowait()
                            if isinstance(item, str) and item.startswith('{"__type":"meta"'):
                                meta = json.loads(item)
                            else:
                                # Put it back and use default meta
                                await lines_queue.put(item)
                                meta = {"status": 200, "headers": {}}
                        except asyncio.QueueEmpty:
                            # No items in queue, use default successful response
                            meta = {"status": 200, "headers": {}}
                        
                        if meta:
                            result = meta
                        else:
                            res = fetch_task.result()
                            if isinstance(res, dict) and not res.get("__streaming"):
                                result = res
                            else:
                                result = {"status": 502, "text": "FETCH_DONE_WITHOUT_META"}
                    except Exception as e:
                        result = {"status": 502, "text": f"FETCH_EXCEPTION: {e}"}
                elif meta:
                    result = meta
                
                status_code = int(result.get("status") or 0)

                # If upstream rate limits us, wait and retry inside the same browser session to avoid hammering.
                if status_code == HTTPStatus.TOO_MANY_REQUESTS and attempt < max_recaptcha_attempts - 1:
                    retry_after = None
                    if isinstance(result, dict) and isinstance(result.get("headers"), dict):
                        headers_map = result.get("headers") or {}
                        retry_after = headers_map.get("retry-after") or headers_map.get("Retry-After")
                    sleep_seconds = _m().get_rate_limit_sleep_seconds(
                        str(retry_after) if retry_after is not None else None,
                        attempt,
                    )
                    await _cancel_background_task(fetch_task)
                    await asyncio.sleep(sleep_seconds)
                    continue

                if not _is_recaptcha_validation_failed(status_code, result.get("text")):
                    # Success or non-recaptcha error. 
                    # If success, start a task to wait for fetch_task to finish and set done_event.
                    if status_code < 400:
                        # If the in-page script returned a buffered body (e.g. in unit tests/mocks where
                        # `reportChunk` isn't exercised), fall back to a plain buffered response.
                        try:
                            candidate_body = result.get("text") if isinstance(result, dict) else None
                        except Exception:
                            candidate_body = None
                        if isinstance(candidate_body, str) and candidate_body:
                            return BrowserFetchStreamResponse(
                                status_code=status_code,
                                headers=result.get("headers", {}) if isinstance(result, dict) else {},
                                text=candidate_body,
                                method=http_method,
                                url=url,
                            )

                        def _on_fetch_task_done(task: "asyncio.Task") -> None:
                            _consume_background_task_exception(task)
                            try:
                                done_event.set()
                            except Exception:
                                pass

                        try:
                            fetch_task.add_done_callback(_on_fetch_task_done)
                        except Exception:
                            pass
                        
                        return BrowserFetchStreamResponse(
                            status_code=status_code,
                            headers=result.get("headers", {}),
                            method=http_method,
                            url=url,
                            lines_queue=lines_queue,
                            done_event=done_event
                        )
                    await _cancel_background_task(fetch_task)
                    break

                await _cancel_background_task(fetch_task)
                if attempt < max_recaptcha_attempts - 1:
                    # ... retry logic ...
                    if isinstance(payload, dict) and not bool(payload.get("recaptchaV2Token")):
                        try:
                            v2_token = await _mint_recaptcha_v2_token()
                        except Exception:
                            v2_token = None
                        if v2_token:
                            payload["recaptchaV2Token"] = v2_token
                            payload.pop("recaptchaV3Token", None)
                            await asyncio.sleep(0.5)
                            continue

                    try:
                        await _m().click_turnstile(page)
                    except Exception:
                        pass

                    try:
                        await page.mouse.move(120 + (attempt * 10), 120 + (attempt * 10))
                        await page.mouse.wheel(0, 250)
                    except Exception:
                        pass
                    await asyncio.sleep(min(2.0 * (2**attempt), 15.0))

            response = BrowserFetchStreamResponse(
                int(result.get("status") or 0),
                result.get("headers") if isinstance(result, dict) else {},
                result.get("text") if isinstance(result, dict) else "",
                method=http_method,
                url=url,
            )
            return response
        except Exception as e:
            _m().debug_print(f"??? Chrome fetch transport failed: {e}")
            return None
        finally:
            await context.close()


async def fetch_via_proxy_queue(
    url: str,
    payload: dict,
    http_method: str = "POST",
    timeout_seconds: int = 120,
    streaming: bool = False,
    auth_token: str = "",
) -> UserscriptProxyStreamResponse | BrowserFetchStreamResponse | None:
    """Delegate a request to an active external userscript proxy."""
    proxy_stream = await fetch_lmarena_stream_via_userscript_proxy(
        http_method=http_method,
        url=url,
        payload=payload or {},
        timeout_seconds=timeout_seconds,
        auth_token=auth_token,
    )
    if proxy_stream is None:
        return None
    if streaming:
        return proxy_stream

    collected_lines: list[str] = []
    async with proxy_stream as response:
        async for line in response.aiter_lines():
            collected_lines.append(str(line))

    return BrowserFetchStreamResponse(
        status_code=getattr(proxy_stream, "status_code", 200),
        headers=getattr(proxy_stream, "headers", {}),
        text="\n".join(collected_lines),
        method=http_method,
        url=url,
    )


async def push_proxy_chunk(jid, d) -> None:
    _touch_userscript_poll()

    job_id = str(jid or "").strip()
    job = _m()._USERSCRIPT_PROXY_JOBS.get(job_id)
    if not isinstance(job, dict):
        return

    if isinstance(d, dict):
        fetch_started = d.get("upstream_fetch_started")
        if fetch_started is None:
            fetch_started = d.get("fetch_started")
        status = d.get("status")
        if fetch_started or isinstance(status, int):
            try:
                if not job.get("upstream_fetch_started_at_monotonic"):
                    job["upstream_fetch_started_at_monotonic"] = _m().time.monotonic()
            except Exception:
                pass

        if isinstance(status, int):
            job["status_code"] = int(status)
            status_event = job.get("status_event")
            if isinstance(status_event, asyncio.Event):
                status_event.set()
            if not job.get("_proxy_status_logged"):
                job["_proxy_status_logged"] = True
                _m().debug_print(f"Userscript proxy job {job_id[:8]} upstream status: {int(status)}")
        headers = d.get("headers")
        if isinstance(headers, dict):
            job["headers"] = headers
        error = d.get("error")
        if error:
            job["error"] = str(error)
            _m().debug_print(f"Userscript proxy job {job_id[:8]} error: {str(error)[:200]}")

        debug_obj = d.get("debug")
        if debug_obj and os.environ.get("LM_BRIDGE_PROXY_DEBUG"):
            try:
                dbg_text = json.dumps(debug_obj, ensure_ascii=False)
            except Exception:
                dbg_text = str(debug_obj)
            _m().debug_print(f"Userscript proxy debug {job_id[:8]}: {dbg_text[:300]}")

        buffer = str(job.get("_proxy_buffer") or "")
        raw_lines = d.get("lines") or []
        if isinstance(raw_lines, list):
            for raw in raw_lines:
                if raw is None:
                    continue
                # The in-page fetch script emits newline-delimited *lines* (without trailing "\n").
                # Join with an explicit newline so we can safely split/enqueue each line here.
                buffer += f"{raw}\n"

        # Safety: normalize and split regardless of whether JS already split lines.
        buffer = buffer.replace("\r\n", "\n").replace("\r", "\n")
        parts = buffer.split("\n")
        buffer = parts.pop() if parts else ""
        job["_proxy_buffer"] = buffer
        for part in parts:
            part = str(part).strip()
            if not part:
                continue
            await job["lines_queue"].put(part)

        if bool(d.get("done")):
            # Flush any remaining partial line.
            remainder = str(job.get("_proxy_buffer") or "").strip()
            if remainder:
                await job["lines_queue"].put(remainder)
            job["_proxy_buffer"] = ""

            job["done"] = True
            done_event = job.get("done_event")
            if isinstance(done_event, asyncio.Event):
                done_event.set()
            status_event = job.get("status_event")
            if isinstance(status_event, asyncio.Event):
                status_event.set()
            await job["lines_queue"].put(None)
            _m().debug_print(f"Userscript proxy job {job_id[:8]} done")
