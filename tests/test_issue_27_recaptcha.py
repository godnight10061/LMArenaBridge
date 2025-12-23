import asyncio
import json
import os
import re
import secrets
import time
import unittest
from pathlib import Path


LMARENA_ORIGIN = "https://lmarena.ai"
RECAPTCHA_SITEKEY = "6Led_uYrAAAAAKjxDIF58fgFtX3t8loNAK85bW9I"
RECAPTCHA_ACTION = "chat_submit"


def uuid7() -> str:
    timestamp_ms = int(time.time() * 1000)
    rand_a = secrets.randbits(12)
    rand_b = secrets.randbits(62)

    uuid_int = timestamp_ms << 80
    uuid_int |= (0x7000 | rand_a) << 64
    uuid_int |= 0x8000000000000000 | rand_b

    hex_str = f"{uuid_int:032x}"
    return f"{hex_str[0:8]}-{hex_str[8:12]}-{hex_str[12:16]}-{hex_str[16:20]}-{hex_str[20:32]}"


async def maybe_click_turnstile(page) -> bool:
    selectors = [
        "#cf-turnstile",
        "iframe[src*='challenges.cloudflare.com']",
        "[style*='display: grid'] iframe",
    ]

    for selector in selectors:
        el = await page.query_selector(selector)
        if not el:
            continue
        box = await el.bounding_box()
        if not box:
            continue
        await page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        await asyncio.sleep(2)
        return True
    return False


async def wait_for_cloudflare(page, timeout_s: int = 120) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            title = await page.title()
        except Exception:
            title = ""
        if "Just a moment" not in title and "Attention Required" not in title:
            return
        await maybe_click_turnstile(page)
        await asyncio.sleep(2)
    raise TimeoutError(f"Timed out waiting for Cloudflare challenge (title={title!r})")


async def wait_for_arena_auth_cookie(page, timeout_s: int = 60) -> None:
    deadline = time.time() + timeout_s
    attempt = 0
    while time.time() < deadline:
        cookies = await page.context.cookies()
        if any(c.get("name") == "arena-auth-prod-v1" for c in cookies):
            return
        if attempt % 3 == 0:
            await maybe_click_turnstile(page)
        attempt += 1
        await asyncio.sleep(1)
    raise TimeoutError("Timed out waiting for arena-auth-prod-v1 cookie")


async def get_recaptcha_token_via_injected_script(page, timeout_s: int = 30) -> str:
    js = f"""
(() => {{
  const root = document.documentElement;
  root.dataset.__lm_bridge_recaptcha = '';
  root.dataset.__lm_bridge_recaptcha_err = '';

  const getGrecaptcha = () => window.grecaptcha?.enterprise || window.grecaptcha;
  const exec = () => {{
    const grecaptcha = getGrecaptcha();
    if (!grecaptcha) return;
    try {{
      grecaptcha.ready(() => {{
        grecaptcha.execute('{RECAPTCHA_SITEKEY}', {{ action: '{RECAPTCHA_ACTION}' }})
          .then((token) => {{ root.dataset.__lm_bridge_recaptcha = token; }})
          .catch((err) => {{ root.dataset.__lm_bridge_recaptcha_err = String(err); }});
      }});
    }} catch (e) {{
      root.dataset.__lm_bridge_recaptcha_err = 'SYNC_ERROR: ' + String(e);
    }}
  }};

  const start = Date.now();
  const timer = setInterval(() => {{
    if (getGrecaptcha()) {{
      clearInterval(timer);
      exec();
      return;
    }}
    if (Date.now() - start > {timeout_s * 1000}) {{
      clearInterval(timer);
      root.dataset.__lm_bridge_recaptcha_err = 'TIMEOUT_WAITING_FOR_GRECAPTCHA';
    }}
  }}, 250);
}})();
"""

    await page.evaluate(
        """(code) => {
          const s = document.createElement('script');
          s.textContent = code;
          document.documentElement.appendChild(s);
          s.remove();
        }""",
        js,
    )

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        token = await page.evaluate("() => document.documentElement.dataset.__lm_bridge_recaptcha")
        err = await page.evaluate("() => document.documentElement.dataset.__lm_bridge_recaptcha_err")
        if err:
            raise RuntimeError(err)
        if token:
            return token
        await asyncio.sleep(0.25)
    raise TimeoutError("Timed out polling for recaptcha token")


async def fetch_create_evaluation_first_chunk(page, payload: dict) -> dict:
    return await page.evaluate(
        """async (payload) => {
          const res = await fetch('/nextjs-api/stream/create-evaluation', {
            method: 'POST',
            headers: { 'content-type': 'text/plain;charset=UTF-8' },
            body: JSON.stringify(payload),
          });
          let first = '';
          try {
            const reader = res.body.getReader();
            const { value } = await reader.read();
            if (value) first = new TextDecoder().decode(value);
            await reader.cancel();
          } catch (e) {
            first = 'READ_ERROR:' + String(e);
          }
          return { status: res.status, first };
        }""",
        payload,
    )


class TestIssue27RecaptchaValidation(unittest.IsolatedAsyncioTestCase):
    @unittest.skipUnless(
        os.getenv("RUN_LMARENA_INTEGRATION") == "1",
        "Set RUN_LMARENA_INTEGRATION=1 to run this real external integration test.",
    )
    async def test_create_evaluation_does_not_fail_recaptcha(self) -> None:
        from camoufox.async_api import AsyncCamoufox

        headless = os.getenv("LMARENA_HEADLESS", "1") not in {"0", "false", "False"}

        async with AsyncCamoufox(headless=headless) as browser:
            page = await browser.new_page()
            await page.goto(LMARENA_ORIGIN, wait_until="domcontentloaded")
            await wait_for_cloudflare(page, timeout_s=120)
            await page.wait_for_selector("textarea", timeout=60000)

            match = None
            for _ in range(2):
                html = await page.content()
                match = re.search(
                    r'{\\"initialModels\\":(\\[.*?\\]),\\"initialModel[A-Z]Id',
                    html,
                    re.DOTALL,
                )
                if match:
                    break
                await asyncio.sleep(3)

            if match:
                models_json = match.group(1).encode().decode("unicode_escape")
                models = json.loads(models_json)
            else:
                models = json.loads(Path("models.json").read_text(encoding="utf-8"))

            self.assertTrue(models, "No models available (neither from page nor models.json)")
            model_id = models[0]["id"]

            await page.fill("textarea", "Hello")
            await page.keyboard.press("Enter")
            await wait_for_arena_auth_cookie(page, timeout_s=60)

            recaptcha_token = await get_recaptcha_token_via_injected_script(page, timeout_s=30)
            self.assertGreater(len(recaptcha_token), 200)

            payload = {
                "id": uuid7(),
                "mode": "direct",
                "modelAId": model_id,
                "userMessageId": uuid7(),
                "modelAMessageId": uuid7(),
                "userMessage": {
                    "content": "Hello from integration test",
                    "experimental_attachments": [],
                    "metadata": {},
                },
                "modality": "chat",
                "recaptchaV3Token": recaptcha_token,
            }

            result = await fetch_create_evaluation_first_chunk(page, payload)

            self.assertEqual(
                result["status"],
                200,
                msg=f"Expected 200 OK but got {result['status']} with first chunk: {result['first'][:200]}",
            )
            self.assertNotIn("recaptcha validation failed", result["first"])
