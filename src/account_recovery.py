"""Managed Arena account inspection, persistence, and browser recovery."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Optional

from playwright.async_api import Browser, BrowserContext, Page, Playwright, async_playwright

try:
    import temp_mail as _temp_mail
except ModuleNotFoundError:
    from . import temp_mail as _temp_mail

try:
    from browser_window import browser_window_mode, chrome_window_args
except ModuleNotFoundError:
    from .browser_window import browser_window_mode, chrome_window_args

TempMailError = _temp_mail.TempMailError
build_provider_chain = _temp_mail.build_provider_chain


ConfigLoader = Callable[[], dict[str, Any]]
ConfigSaver = Callable[[dict[str, Any]], None]
DebugFn = Callable[[str], None]


@dataclass
class AuthInspection:
    authenticated: bool
    anonymous: bool
    expired: bool
    email: str = ""
    reason: str = "invalid_auth"


@dataclass
class RecoveryResult:
    ok: bool
    stage: str
    action: str
    retryable: bool
    authenticated: bool
    error_code: Optional[str] = None
    browser_mode: Optional[str] = None


@dataclass
class BrowserSession:
    context: BrowserContext
    mode: Literal["borrowed_cdp", "owned_persistent"]
    executable: str
    browser: Optional[Browser] = None
    _closed: bool = False

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.mode == "owned_persistent":
            await self.context.close()


class BrowserSessionAcquisitionError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


_RECOVERY_LOCK = asyncio.Lock()
_EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
_URL_TOKEN_RE = re.compile(r"([?&](?:token|code|sid_token)=)[^&\s]+", re.I)
_AUTH_TOKEN_RE = re.compile(r"base64-[A-Za-z0-9_\-+/=\.]+")


def redact_text(value: Any) -> str:
    text = str(value or "")
    text = _EMAIL_RE.sub("<redacted-email>", text)
    text = _URL_TOKEN_RE.sub(r"\1<redacted>", text)
    text = _AUTH_TOKEN_RE.sub("<redacted-auth-token>", text)
    text = re.sub(
        r"(?i)(password|authorization|cookie|cf_clearance|arena-auth-prod-v1|api\s+key|refresh_token|access_token|sid_token)\s*[:=]\s*[^\s,;]+",
        r"\1=<redacted>",
        text,
    )
    return text


def _b64decode_json(encoded: str) -> dict[str, Any]:
    payload = encoded[len("base64-") :] if encoded.startswith("base64-") else encoded
    payload += "=" * (-len(payload) % 4)
    decoded = base64.urlsafe_b64decode(payload.encode("ascii"))
    data = json.loads(decoded.decode("utf-8"))
    return data if isinstance(data, dict) else {}


def inspect_auth_cookie(token: str) -> AuthInspection:
    if not str(token or "").strip():
        return AuthInspection(False, False, False, reason="missing_auth")
    if not str(token).strip().startswith("base64-"):
        # Preserve compatibility with externally supplied legacy/opaque tokens.
        # Managed replacement accounts are still required to pass decoded checks.
        return AuthInspection(True, False, False, reason="legacy_opaque")
    try:
        payload = _b64decode_json(str(token).strip())
    except (ValueError, UnicodeError, json.JSONDecodeError, binascii.Error):
        return AuthInspection(False, False, False, reason="invalid_auth")

    raw_user = payload.get("user")
    user: dict[str, Any] = raw_user if isinstance(raw_user, dict) else {}
    anonymous = bool(user.get("is_anonymous", payload.get("is_anonymous", False)))
    role = str(user.get("role") or payload.get("role") or "").lower()
    audience = str(user.get("aud") or payload.get("aud") or "").lower()
    email = str(user.get("email") or "")
    expiry = payload.get("expires_at")
    expired = False
    try:
        expired = bool(expiry and float(expiry) <= time.time() + 30)
    except (TypeError, ValueError):
        expired = False
    authenticated = (
        bool(user)
        and not anonymous
        and not expired
        and (role == "authenticated" or audience == "authenticated")
    )
    reason = "authenticated" if authenticated else (
        "expired_auth" if expired else "anonymous_auth" if anonymous else "invalid_auth"
    )
    return AuthInspection(authenticated, anonymous, expired, email=email, reason=reason)


def _combine_split_auth(cookies: list[dict[str, Any]]) -> str:
    direct = next(
        (
            str(cookie.get("value") or "")
            for cookie in cookies
            if str(cookie.get("name") or "") == "arena-auth-prod-v1"
        ),
        "",
    )
    if direct:
        return direct
    parts = {
        str(cookie.get("name") or ""): str(cookie.get("value") or "")
        for cookie in cookies
    }
    return (parts.get("arena-auth-prod-v1.0", "") + parts.get("arena-auth-prod-v1.1", "")).strip()


def inspect_config_auth(config: dict[str, Any]) -> tuple[str, AuthInspection]:
    candidates = [
        str(config.get("auth_token") or ""),
        *[str(item or "") for item in config.get("auth_tokens", [])],
    ]
    browser_cookies = config.get("browser_cookies") or {}
    if isinstance(browser_cookies, dict):
        candidates.append(str(browser_cookies.get("arena-auth-prod-v1") or ""))
        candidates.append(
            str(browser_cookies.get("arena-auth-prod-v1.0") or "")
            + str(browser_cookies.get("arena-auth-prod-v1.1") or "")
        )
    cookie_jar = config.get("cookie_jar") or []
    if isinstance(cookie_jar, list):
        candidates.append(_combine_split_auth(cookie_jar))
    best = AuthInspection(False, False, False, reason="missing_auth")
    for candidate in candidates:
        inspection = inspect_auth_cookie(candidate)
        if inspection.authenticated:
            return candidate, inspection
        if inspection.reason != "missing_auth":
            best = inspection
    return "", best


def merge_authenticated_tokens(token: str, existing: list[Any]) -> list[str]:
    """Place a verified token first and discard decoded anonymous/expired tokens."""
    merged = [str(token).strip()]
    for item in existing:
        candidate = str(item or "").strip()
        if not candidate or candidate in merged:
            continue
        inspection = inspect_auth_cookie(candidate)
        if candidate.startswith("base64-") and not inspection.authenticated:
            continue
        merged.append(candidate)
    return merged


def classify_failure(status_code: int | None, detail: str = "") -> str:
    lowered = str(detail or "").lower()
    if status_code == 429:
        return "rate_limited"
    if status_code == 404 or "model" in lowered and "not found" in lowered:
        return "model_missing"
    if "recaptcha" in lowered or "challenge" in lowered:
        return "challenge_retryable"
    if status_code == 401:
        return "invalid_auth"
    if status_code == 403 and any(
        marker in lowered for marker in ("auth", "session", "login", "expired")
    ):
        return "invalid_auth"
    if status_code and status_code >= 500:
        return "upstream_error"
    if any(marker in lowered for marker in ("timeout", "dns", "connection")):
        return "network_error"
    return "unknown"


def _default_profile_dir(config: dict[str, Any]) -> Path:
    configured = str(
        os.environ.get("LM_CHROME_PROFILE_DIR")
        or config.get("chrome_fetch_user_data_dir")
        or ""
    ).strip()
    path = Path(configured) if configured else Path.cwd() / ".runtime" / "chrome-profile"
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _browser_executable(config: dict[str, Any], playwright_executable: str) -> str:
    configured = str(
        os.environ.get("LM_CHROME_EXECUTABLE")
        or config.get("chrome_fetch_executable")
        or ""
    ).strip()
    candidates = [
        Path(configured) if configured else None,
        Path(os.environ.get("PROGRAMFILES", r"C:\Program Files"))
        / "Google"
        / "Chrome"
        / "Application"
        / "chrome.exe",
        Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"))
        / "Google"
        / "Chrome"
        / "Application"
        / "chrome.exe",
        Path(os.environ.get("LOCALAPPDATA", ""))
        / "Google"
        / "Chrome"
        / "Application"
        / "chrome.exe",
        Path(os.environ.get("PROGRAMFILES", r"C:\Program Files"))
        / "Microsoft"
        / "Edge"
        / "Application"
        / "msedge.exe",
        Path(playwright_executable),
    ]
    for candidate in candidates:
        if candidate and candidate.is_file():
            return str(candidate.resolve())
    return playwright_executable


def _config_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


def _configured_cdp_port(config: dict[str, Any]) -> int:
    try:
        port = int(config.get("chrome_fetch_cdp_port") or 9333)
    except (TypeError, ValueError):
        port = 9333
    return port if 1 <= port <= 65535 else 9333


async def _probe_cdp(port: int, timeout: float = 1.0) -> bool:
    writer = None
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection("127.0.0.1", port), timeout=timeout
        )
        writer.write(
            (
                "GET /json/version HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{port}\r\n"
                "Connection: close\r\n\r\n"
            ).encode("ascii")
        )
        await asyncio.wait_for(writer.drain(), timeout=timeout)
        status_line = await asyncio.wait_for(reader.readline(), timeout=timeout)
        return status_line.startswith(b"HTTP/") and b" 200 " in status_line
    except (OSError, asyncio.TimeoutError):
        return False
    finally:
        if writer is not None:
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass


async def _acquire_browser_session(
    playwright: Playwright,
    config: dict[str, Any],
    *,
    profile_dir: Path,
    headless: bool,
) -> BrowserSession:
    """Borrow the bridge CDP context when live, otherwise own a new context."""
    executable = _browser_executable(config, playwright.chromium.executable_path)
    reuse_cdp = _config_bool(config.get("chrome_fetch_reuse_cdp"), True)
    cdp_port = _configured_cdp_port(config)
    window_mode = browser_window_mode(config)

    cdp_live = False
    if reuse_cdp:
        for attempt in range(3):
            if await _probe_cdp(cdp_port):
                cdp_live = True
                break
            if attempt < 2:
                await asyncio.sleep(0.2 * (attempt + 1))

    if cdp_live:
        last_error: Optional[Exception] = None
        for attempt in range(3):
            try:
                browser = await playwright.chromium.connect_over_cdp(
                    f"http://127.0.0.1:{cdp_port}", timeout=10000
                )
                if not browser.contexts:
                    raise RuntimeError("CDP browser has no reusable context")
                return BrowserSession(
                    context=browser.contexts[0],
                    mode="borrowed_cdp",
                    executable=executable,
                    browser=browser,
                )
            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    await asyncio.sleep(0.25 * (attempt + 1))
        raise BrowserSessionAcquisitionError(
            "cdp_attach_failed",
            f"Active Chrome CDP attachment failed: {type(last_error).__name__}",
        )

    context = await playwright.chromium.launch_persistent_context(
        user_data_dir=str(profile_dir),
        executable_path=executable,
        headless=headless,
        viewport={"width": 1280, "height": 900},
        args=[
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-blink-features=AutomationControlled",
            *chrome_window_args(window_mode),
        ],
    )
    return BrowserSession(
        context=context,
        mode="owned_persistent",
        executable=executable,
    )


def _artifact_dir() -> Path:
    path = Path(os.environ.get("LM_RECOVERY_ARTIFACT_DIR") or ".runtime/artifacts")
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _cookie_jar(cookies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    allowed = {
        "arena-auth-prod-v1",
        "arena-auth-prod-v1.0",
        "arena-auth-prod-v1.1",
        "cf_clearance",
        "__cf_bm",
        "provisional_user_id",
        "arena_visit_id",
    }
    result = []
    for cookie in cookies:
        if cookie.get("name") not in allowed:
            continue
        result.append(
            {
                key: cookie.get(key)
                for key in (
                    "name",
                    "value",
                    "domain",
                    "path",
                    "expires",
                    "httpOnly",
                    "secure",
                    "sameSite",
                )
                if key in cookie
            }
        )
    return result


async def capture_failure_artifact(
    page: Page, stage: str, detail: str, email: str = ""
) -> None:
    timestamp = int(time.time())
    base = _artifact_dir() / f"recovery-{timestamp}-{stage}"
    masks = [page.locator("input"), page.locator("textarea")]
    if email:
        masks.append(page.get_by_text(email, exact=False))
    try:
        await page.screenshot(path=str(base.with_suffix(".png")), full_page=True, mask=masks)
    except Exception:
        pass
    summary = {
        "stage": stage,
        "url": redact_text(page.url),
        "detail": redact_text(detail),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    base.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


class _ProfileLock:
    def __init__(self, profile_dir: Path, timeout: float = 180.0):
        self.path = profile_dir / ".lmarena-recovery.lock"
        self.timeout = timeout
        self.acquired = False

    async def __aenter__(self):
        deadline = asyncio.get_running_loop().time() + self.timeout
        while asyncio.get_running_loop().time() < deadline:
            try:
                fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode("ascii"))
                os.close(fd)
                self.acquired = True
                return self
            except FileExistsError:
                try:
                    if time.time() - self.path.stat().st_mtime > self.timeout * 2:
                        self.path.unlink(missing_ok=True)
                        continue
                except OSError:
                    pass
                await asyncio.sleep(0.5)
        raise TimeoutError("Timed out waiting for the Arena browser profile lock")

    async def __aexit__(self, exc_type, exc, tb):
        if self.acquired:
            try:
                self.path.unlink(missing_ok=True)
            except OSError:
                pass


async def _open_login_dialog(page: Page) -> None:
    await page.goto("https://arena.ai/", wait_until="domcontentloaded", timeout=120000)
    login = page.get_by_role("button", name="Log In", exact=True)
    try:
        await login.wait_for(state="visible", timeout=15000)
    except Exception:
        unnamed = page.locator("button").nth(1)
        await unnamed.click(timeout=10000)
        await login.wait_for(state="visible", timeout=15000)
    await login.click(timeout=15000)
    await page.get_by_role("heading", name="Log In or Create Account").wait_for(
        state="visible", timeout=15000
    )


async def _current_context_auth(context: BrowserContext) -> tuple[str, AuthInspection, list[dict[str, Any]]]:
    cookies = [dict(cookie) for cookie in await context.cookies(["https://arena.ai/"])]
    token = _combine_split_auth(cookies)
    return token, inspect_auth_cookie(token), cookies


async def _login(page: Page, context: BrowserContext, account: dict[str, Any]):
    email = str(account.get("address") or "")
    password = str(account.get("password") or "")
    if not email or not password:
        return "", AuthInspection(False, False, False, reason="missing_credentials"), []
    await _open_login_dialog(page)
    await page.get_by_placeholder("Your email").fill(email)
    await page.get_by_role("button", name="Continue with email").click()
    heading = page.get_by_role("heading", name="Log In to your account")
    await heading.wait_for(state="visible", timeout=20000)
    await page.get_by_role("textbox", name="Password").fill(password)
    await page.get_by_role("button", name="Log In", exact=True).click()
    await page.wait_for_url("https://arena.ai/**", timeout=60000)
    await page.wait_for_timeout(1500)
    return await _current_context_auth(context)


async def _signup(page: Page, context: BrowserContext, provider, mailbox):
    await _open_login_dialog(page)
    await page.get_by_placeholder("Your email").fill(mailbox.address)
    await page.get_by_role("button", name="Continue with email").click()
    await page.get_by_role("heading", name="Create Account").wait_for(
        state="visible", timeout=20000
    )
    name_field = page.get_by_role("textbox", name="Full Name (Optional)")
    if await name_field.count():
        await name_field.fill("Bridge User")
    await page.get_by_role("button", name="Create Account", exact=True).click()
    await page.get_by_role(
        "heading", name="Verify your email address to continue"
    ).wait_for(state="visible", timeout=30000)

    message_timeout = float(os.environ.get("LM_SIGNUP_EMAIL_TIMEOUT") or 180)
    message = await provider.wait_for_verification(mailbox, timeout=message_timeout)
    await page.goto(message.verification_url, wait_until="domcontentloaded", timeout=120000)
    await page.get_by_role("heading", name="Create password").wait_for(
        state="visible", timeout=30000
    )
    password_fields = page.get_by_role("textbox", name="Password")
    if await password_fields.count() < 2:
        password_fields = page.locator("input[type='password']")
    await password_fields.nth(0).fill(mailbox.password)
    await password_fields.nth(1).fill(mailbox.password)
    await page.get_by_role("button", name="Finish", exact=True).click()
    await page.wait_for_url("https://arena.ai/**", timeout=60000)
    await page.wait_for_timeout(1500)
    return await _current_context_auth(context)


def _persist(
    config_loader: ConfigLoader,
    config_saver: ConfigSaver,
    *,
    token: str,
    cookies: list[dict[str, Any]],
    profile_dir: Path,
    executable: str,
    account: Optional[dict[str, Any]],
) -> None:
    config = config_loader()
    history = list(config.get("managed_account_history") or [])
    current = config.get("managed_account")
    if current:
        history.insert(0, current)
    config["managed_account_history"] = history[:3]
    if account:
        config["managed_account"] = account
    config["auth_token"] = token
    config["auth_tokens"] = merge_authenticated_tokens(
        token, list(config.get("auth_tokens", []))
    )
    config["cookie_jar"] = _cookie_jar(cookies)
    config["browser_cookies"] = {
        str(item.get("name")): str(item.get("value") or "")
        for item in cookies
        if item.get("name")
    }
    config["chrome_fetch_user_data_dir"] = str(profile_dir)
    config["chrome_fetch_executable"] = executable
    config_saver(config)


async def ensure_authenticated_account(
    config_loader: ConfigLoader,
    config_saver: ConfigSaver,
    *,
    reason: str = "request",
    force_recovery: bool = False,
    debug: Optional[DebugFn] = None,
) -> RecoveryResult:
    log = debug or (lambda _message: None)
    async with _RECOVERY_LOCK:
        config = config_loader()
        token, inspection = inspect_config_auth(config)
        if inspection.authenticated and not force_recovery:
            return RecoveryResult(True, "inspect", "reuse", False, True)

        if str(os.environ.get("LM_AUTO_ACCOUNT_RECOVERY", "1")).lower() in {"0", "false", "no"}:
            return RecoveryResult(
                False, "disabled", "none", False, False, "recovery_disabled"
            )

        profile_dir = _default_profile_dir(config)
        headless = str(os.environ.get("LM_BROWSER_HEADLESS", "0")).lower() in {
            "1",
            "true",
            "yes",
        }
        log(redact_text(f"Managed auth recovery started reason={reason} stage=browser"))

        async with _ProfileLock(profile_dir):
            async with async_playwright() as playwright:
                try:
                    session = await _acquire_browser_session(
                        playwright,
                        config,
                        profile_dir=profile_dir,
                        headless=headless,
                    )
                except BrowserSessionAcquisitionError as exc:
                    log(redact_text(f"Managed auth browser acquisition failed: {exc}"))
                    return RecoveryResult(
                        False,
                        "browser",
                        "none",
                        True,
                        False,
                        exc.code,
                    )
                except Exception as exc:
                    log(
                        redact_text(
                            "Managed auth browser launch failed: "
                            f"{type(exc).__name__}: {exc}"
                        )
                    )
                    return RecoveryResult(
                        False,
                        "browser",
                        "none",
                        True,
                        False,
                        "browser_launch_failed",
                    )

                context = session.context
                executable = session.executable
                browser_mode = session.mode
                log(f"Managed auth recovery browser mode={browser_mode}")
                try:
                    page = context.pages[0] if context.pages else await context.new_page()

                    # Stage 1: allow the persistent profile/site to refresh its session.
                    try:
                        await page.goto(
                            "https://arena.ai/",
                            wait_until="domcontentloaded",
                            timeout=120000,
                        )
                        await page.wait_for_timeout(1500)
                        fresh_token, fresh_inspection, cookies = await _current_context_auth(context)
                        if fresh_inspection.authenticated:
                            _persist(
                                config_loader,
                                config_saver,
                                token=fresh_token,
                                cookies=cookies,
                                profile_dir=profile_dir,
                                executable=executable,
                                account=config.get("managed_account"),
                            )
                            return RecoveryResult(
                                True,
                                "refresh",
                                "refresh",
                                False,
                                True,
                                browser_mode=browser_mode,
                            )
                    except Exception as exc:
                        log(redact_text(f"Managed auth refresh failed: {exc}"))

                    # Stage 2: re-login with the generated account.
                    account = config.get("managed_account") or {}
                    if account:
                        try:
                            fresh_token, fresh_inspection, cookies = await _login(
                                page, context, account
                            )
                            if fresh_inspection.authenticated:
                                updated = dict(account)
                                updated["status"] = "active"
                                updated["last_verified_at"] = int(time.time())
                                _persist(
                                    config_loader,
                                    config_saver,
                                    token=fresh_token,
                                    cookies=cookies,
                                    profile_dir=profile_dir,
                                    executable=executable,
                                    account=updated,
                                )
                                return RecoveryResult(
                                    True,
                                    "login",
                                    "login",
                                    False,
                                    True,
                                    browser_mode=browser_mode,
                                )
                        except Exception as exc:
                            await capture_failure_artifact(
                                page,
                                "login",
                                str(exc),
                                str(account.get("address") or ""),
                            )
                            log(redact_text(f"Managed account login failed: {exc}"))

                    # Stage 3: create a replacement account using ordered providers.
                    provider_errors: list[str] = []
                    for provider in build_provider_chain():
                        mailbox = None
                        try:
                            mailbox = await provider.create_mailbox()
                            fresh_token, fresh_inspection, cookies = await _signup(
                                page, context, provider, mailbox
                            )
                            if not fresh_inspection.authenticated or fresh_inspection.anonymous:
                                raise RuntimeError("signup returned a non-authenticated session")
                            managed = {
                                "schema": 1,
                                "provider": provider.name,
                                "address": mailbox.address,
                                "password": mailbox.password,
                                "created_at": int(time.time()),
                                "last_verified_at": int(time.time()),
                                "status": "active",
                            }
                            _persist(
                                config_loader,
                                config_saver,
                                token=fresh_token,
                                cookies=cookies,
                                profile_dir=profile_dir,
                                executable=executable,
                                account=managed,
                            )
                            return RecoveryResult(
                                True,
                                "signup",
                                "signup",
                                False,
                                True,
                                browser_mode=browser_mode,
                            )
                        except Exception as exc:
                            provider_errors.append(f"{provider.name}:{type(exc).__name__}")
                            email = mailbox.address if mailbox else ""
                            await capture_failure_artifact(
                                page, f"signup-{provider.name}", str(exc), email
                            )
                            log(
                                redact_text(
                                    f"Managed signup provider={provider.name} failed: {exc}"
                                )
                            )
                            try:
                                await page.goto(
                                    "https://arena.ai/",
                                    wait_until="domcontentloaded",
                                    timeout=120000,
                                )
                            except Exception:
                                pass
                        finally:
                            if mailbox:
                                try:
                                    await provider.close_mailbox(mailbox)
                                except TempMailError:
                                    pass
                    return RecoveryResult(
                        False,
                        "signup",
                        "signup",
                        True,
                        False,
                        "all_providers_failed:" + ",".join(provider_errors),
                        browser_mode,
                    )
                finally:
                    await session.close()
