"""Chrome-backed reCAPTCHA discovery, minting, and token caching."""

import asyncio
import os
import re
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from .browser_window import browser_window_mode, chrome_window_args


def _m():
    """Late import of main module so tests can patch main.X and it is reflected here."""
    from . import main

    return main


def extract_recaptcha_params_from_text(text: str) -> tuple[Optional[str], Optional[str]]:
    if not isinstance(text, str) or not text:
        return None, None

    discovered_sitekey: Optional[str] = None
    discovered_action: Optional[str] = None

    # 1) Prefer direct matches from execute(sitekey,{action:"..."}) when present.
    if "execute" in text and "action" in text:
        patterns = [
            r'grecaptcha\.enterprise\.execute\(\s*["\'](?P<sitekey>[0-9A-Za-z_-]{8,200})["\']\s*,\s*\{\s*(?:action|["\']action["\'])\s*:\s*["\'](?P<action>[^"\']{1,80})["\']',
            r'grecaptcha\.execute\(\s*["\'](?P<sitekey>[0-9A-Za-z_-]{8,200})["\']\s*,\s*\{\s*(?:action|["\']action["\'])\s*:\s*["\'](?P<action>[^"\']{1,80})["\']',
            # Fallback for minified code that aliases grecaptcha to another identifier.
            r'\.execute\(\s*["\'](?P<sitekey>6[0-9A-Za-z_-]{8,200})["\']\s*,\s*\{\s*(?:action|["\']action["\'])\s*:\s*["\'](?P<action>[^"\']{1,80})["\']',
        ]
        for pattern in patterns:
            try:
                match = re.search(pattern, text)
            except re.error:
                continue
            if not match:
                continue
            sitekey = str(match.group("sitekey") or "").strip()
            action = str(match.group("action") or "").strip()
            if sitekey and action:
                return sitekey, action

    # 2) Discover sitekey from the enterprise.js/api.js render URL (common in HTML/JS chunks).
    # Example: https://www.google.com/recaptcha/enterprise.js?render=SITEKEY
    sitekey_patterns = [
        r"recaptcha/(?:enterprise|api)\.js\?render=(?P<sitekey>[0-9A-Za-z_-]{8,200})",
        r"(?:enterprise|api)\.js\?render=(?P<sitekey>[0-9A-Za-z_-]{8,200})",
    ]
    for pattern in sitekey_patterns:
        try:
            match = re.search(pattern, text)
        except re.error:
            continue
        if not match:
            continue
        sitekey = str(match.group("sitekey") or "").strip()
        if sitekey:
            discovered_sitekey = sitekey
            break

    # 3) Discover action from headers/constants in client-side code.
    if "recaptcha" in text.lower() or "X-Recaptcha-Action" in text or "x-recaptcha-action" in text:
        action_patterns = [
            r'X-Recaptcha-Action["\']\s*[:=]\s*["\'](?P<action>[^"\']{1,80})["\']',
            r'X-Recaptcha-Action["\']\s*,\s*["\'](?P<action>[^"\']{1,80})["\']',
            r'x-recaptcha-action["\']\s*[:=]\s*["\'](?P<action>[^"\']{1,80})["\']',
        ]
        for pattern in action_patterns:
            try:
                match = re.search(pattern, text)
            except re.error:
                continue
            if not match:
                continue
            action = str(match.group("action") or "").strip()
            if action:
                discovered_action = action
                break

    return discovered_sitekey, discovered_action


def get_recaptcha_settings(config: Optional[dict] = None) -> tuple[str, str]:
    cfg = config or _m().get_config()
    sitekey = str((cfg or {}).get("recaptcha_sitekey") or "").strip()
    action = str((cfg or {}).get("recaptcha_action") or "").strip()
    if not sitekey:
        sitekey = _m().RECAPTCHA_SITEKEY

    if not action:
        # Support both auth_tokens (list) and auth_token (legacy singular)
        auth_tokens = cfg.get("auth_tokens", []) if cfg else []
        # Backward compatibility: also check for singular auth_token
        singular_token = cfg.get("auth_token", "") if cfg else ""
        if singular_token and isinstance(auth_tokens, list) and not auth_tokens:
            auth_tokens = [singular_token]
        if isinstance(auth_tokens, list):
            auth_tokens = [str(t or "").strip() for t in auth_tokens if str(t or "").strip()]

        # Also check legacy auth_token field
        legacy_token = str(cfg.get("auth_token") or "").strip() if cfg else ""
        if legacy_token and legacy_token not in auth_tokens:
            auth_tokens.append(legacy_token)

        has_valid_token = any(_m().is_probably_valid_arena_auth_token(t) for t in auth_tokens)

        action = "chat_submit" if has_valid_token else "sign_up"

    return sitekey, action


def find_chrome_executable(
    config: Optional[dict] = None,
    *,
    playwright_executable: str = "",
) -> Optional[str]:
    cfg = config
    if cfg is None:
        try:
            cfg = _m().get_config()
        except Exception:
            cfg = {}

    environment_override = str(
        os.environ.get("LM_CHROME_EXECUTABLE")
        or os.environ.get("CHROME_PATH")
        or ""
    ).strip()
    config_override = str((cfg or {}).get("chrome_fetch_executable") or "").strip()

    candidates = [
        Path(environment_override) if environment_override else None,
        Path(config_override) if config_override else None,
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
        Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"))
        / "Microsoft"
        / "Edge"
        / "Application"
        / "msedge.exe",
        Path(playwright_executable) if str(playwright_executable or "").strip() else None,
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return str(candidate)

    for name in ("google-chrome", "chrome", "chromium", "chromium-browser", "msedge"):
        resolved = shutil.which(name)
        if resolved:
            return resolved

    return None


async def get_recaptcha_v3_token_with_chrome(config: dict) -> Optional[str]:
    try:
        from playwright.async_api import async_playwright  # type: ignore
    except Exception:
        return None

    profile_dir = Path(_m().CONFIG_FILE).with_name("chrome_grecaptcha")

    cf_clearance = str(config.get("cf_clearance") or "").strip()
    cf_bm = str(config.get("cf_bm") or "").strip()
    cfuvid = str(config.get("cfuvid") or "").strip()
    provisional_user_id = str(config.get("provisional_user_id") or "").strip()
    user_agent = _m().normalize_user_agent_value(config.get("user_agent"))
    recaptcha_sitekey, recaptcha_action = get_recaptcha_settings(config)

    cookies = []
    # When using domain, do NOT include path - they're mutually exclusive in Playwright
    if cf_clearance:
        cookies.append({"name": "cf_clearance", "value": cf_clearance, "domain": ".arena.ai"})
    if cf_bm:
        cookies.append({"name": "__cf_bm", "value": cf_bm, "domain": ".arena.ai"})
    if cfuvid:
        cookies.append({"name": "_cfuvid", "value": cfuvid, "domain": ".arena.ai"})
    if provisional_user_id:
        cookies.append(
            {"name": "provisional_user_id", "value": provisional_user_id, "domain": ".arena.ai"}
        )
    window_args = chrome_window_args(browser_window_mode(config))
    async with async_playwright() as p:
        chrome_path = find_chrome_executable(
            config,
            playwright_executable=p.chromium.executable_path,
        )
        if not chrome_path:
            return None
        context = await p.chromium.launch_persistent_context(
            user_data_dir=str(profile_dir),
            executable_path=chrome_path,
            headless=False,  # Headful for better reCAPTCHA score/warmup
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

            if cookies:
                try:
                    existing_names: set[str] = set()
                    try:
                        existing = await _m()._get_arena_context_cookies(context)
                        for c in existing or []:
                            name = c.get("name")
                            if name:
                                existing_names.add(str(name))
                    except Exception:
                        existing_names = set()

                    cookies_to_add: list[dict] = []
                    for c in cookies:
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
            await page.goto(
                "https://arena.ai/?mode=direct", wait_until="domcontentloaded", timeout=120000
            )

            # Best-effort: if we land on a Cloudflare challenge page, try clicking Turnstile.
            try:
                for _ in range(5):
                    title = await page.title()
                    if "Just a moment" not in title:
                        break
                    await _m().click_turnstile(page)
                    await asyncio.sleep(2)
            except Exception:
                pass

            # Light warm-up (often improves reCAPTCHA v3 score vs firing immediately).
            try:
                await page.mouse.move(100, 100)
                await page.mouse.wheel(0, 200)
                await asyncio.sleep(1)
                await page.mouse.move(200, 300)
                await page.mouse.wheel(0, 300)
                await asyncio.sleep(3)  # Increased "Human" pause
            except Exception:
                pass

            # Persist updated cookies/UA from this real browser context (often refreshes arena-auth-prod-v1).
            try:
                fresh_cookies = await _m()._get_arena_context_cookies(
                    context, page_url=str(getattr(page, "url", "") or "")
                )
                try:
                    ua_now = await page.evaluate("() => navigator.userAgent")
                except Exception:
                    ua_now = user_agent
                if _m()._upsert_browser_session_into_config(
                    config, fresh_cookies, user_agent=ua_now
                ):
                    _m().save_config(config)
            except Exception:
                pass

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
        except Exception as e:
            _m().debug_print(f"⚠️ Chrome reCAPTCHA retrieval failed: {e}")
            return None
        finally:
            await context.close()


async def get_recaptcha_v3_token() -> Optional[str]:
    """Mint and cache a fresh reCAPTCHA v3 token with Chrome."""
    config = _m().get_config()
    recaptcha_sitekey, recaptcha_action = get_recaptcha_settings(config)
    _m().debug_print(
        "Starting Chrome reCAPTCHA v3 retrieval "
        f"sitekey={recaptcha_sitekey[:20]}... action={recaptcha_action}"
    )
    try:
        token = await get_recaptcha_v3_token_with_chrome(config)
    except Exception as exc:
        _m().debug_print(f"Chrome reCAPTCHA retrieval raised {type(exc).__name__}: {exc}")
        return None
    if not token:
        _m().debug_print("Chrome reCAPTCHA retrieval returned no token.")
        return None
    _m().RECAPTCHA_TOKEN = token
    _m().RECAPTCHA_EXPIRY = datetime.now(timezone.utc) + timedelta(seconds=110)
    return token


async def refresh_recaptcha_token(force_new: bool = False):
    """Checks if the global reCAPTCHA token is expired and refreshes it if necessary."""

    current_time = datetime.now(timezone.utc)
    if force_new:
        _m().RECAPTCHA_TOKEN = None
        _m().RECAPTCHA_EXPIRY = current_time - timedelta(days=365)
    # Unit tests should never launch real browser automation. Tests that need a token patch
    # `refresh_recaptcha_token` / `get_recaptcha_v3_token` explicitly.
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return get_cached_recaptcha_token() or None
    # Check if token is expired (set a refresh margin of 10 seconds)
    if _m().RECAPTCHA_TOKEN is None or current_time > _m().RECAPTCHA_EXPIRY - timedelta(seconds=10):
        _m().debug_print("🔄 Recaptcha token expired or missing. Refreshing...")
        new_token = await get_recaptcha_v3_token()
        if new_token:
            _m().RECAPTCHA_TOKEN = new_token
            # reCAPTCHA v3 tokens typically last 120 seconds (2 minutes)
            _m().RECAPTCHA_EXPIRY = current_time + timedelta(seconds=120)
            _m().debug_print(
                f"✅ Recaptcha token refreshed, expires at {_m().RECAPTCHA_EXPIRY.isoformat()}"
            )
            return new_token
        else:
            _m().debug_print("❌ Failed to refresh recaptcha token.")
            # Set a short retry delay if refresh fails
            _m().RECAPTCHA_EXPIRY = current_time + timedelta(seconds=10)
            return None

    return _m().RECAPTCHA_TOKEN


def get_cached_recaptcha_token() -> str:
    """Return the current reCAPTCHA v3 token if it's still valid, without refreshing."""
    token = _m().RECAPTCHA_TOKEN
    if not token:
        return ""
    current_time = datetime.now(timezone.utc)
    if current_time > _m().RECAPTCHA_EXPIRY - timedelta(seconds=10):
        return ""
    return str(token)
