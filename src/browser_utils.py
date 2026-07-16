"""Shared browser interaction and async task lifecycle helpers."""

import asyncio
from typing import Optional


async def click_turnstile(page) -> bool:
    """
    Attempts to locate and click the Cloudflare Turnstile widget.
    Based on gpt4free logic.
    """
    from . import main as _main  # late import to avoid circularity

    _main.debug_print("  🖱️  Attempting to click Cloudflare Turnstile...")
    try:
        # Common selectors used by LMArena's Turnstile implementation
        selectors = [
            "#lm-bridge-turnstile",
            "#lm-bridge-turnstile iframe",
            "#cf-turnstile",
            'iframe[src*="challenges.cloudflare.com"]',
            '[style*="display: grid"] iframe',  # The grid style often wraps the checkbox
        ]

        for selector in selectors:
            try:
                # Playwright pages support `query_selector_all`, but our unit-test stubs may only implement
                # `query_selector`. Support both for robustness.
                query_all = getattr(page, "query_selector_all", None)
                if callable(query_all):
                    elements = await query_all(selector)
                else:
                    one = await page.query_selector(selector)
                    elements = [one] if one else []
            except Exception:
                try:
                    one = await page.query_selector(selector)
                    elements = [one] if one else []
                except Exception:
                    elements = []
            for element in elements or []:
                # If this is a Turnstile iframe, try clicking within the frame first.
                try:
                    frame = await element.content_frame()
                except Exception:
                    frame = None

                if frame is not None:
                    inner_selectors = [
                        "input[type='checkbox']",
                        "div[role='checkbox']",
                        "label",
                    ]
                    for inner_sel in inner_selectors:
                        try:
                            inner = await frame.query_selector(inner_sel)
                            if inner:
                                try:
                                    await inner.click(force=True)
                                except TypeError:
                                    await inner.click()
                                await asyncio.sleep(2)
                                return True
                        except Exception:
                            continue

                # If the OS window is hidden/occluded, Playwright may return no bounding box even when the element is
                # present. Try a direct element click first (force) before relying on geometry.
                try:
                    try:
                        await element.click(force=True)
                    except TypeError:
                        await element.click()
                    await asyncio.sleep(2)
                    return True
                except Exception:
                    pass

                # Get bounding box to click specific coordinates if needed
                try:
                    box = await element.bounding_box()
                except Exception:
                    box = None
                if box:
                    x = box["x"] + (box["width"] / 2)
                    y = box["y"] + (box["height"] / 2)
                    _main.debug_print(f"  🎯 Found widget at {x},{y}. Clicking...")
                    await page.mouse.click(x, y)
                    await asyncio.sleep(2)
                    return True
        return False
    except Exception as e:
        _main.debug_print(f"  ⚠️ Error clicking turnstile: {e}")
        return False


def _consume_background_task_exception(task: "asyncio.Task") -> None:
    try:
        task.exception()
    except asyncio.CancelledError:
        pass
    except Exception:
        pass


async def _cancel_background_task(
    task: Optional["asyncio.Task"], *, timeout_seconds: float = 1.0
) -> None:
    if task is None:
        return
    if task.done():
        _consume_background_task_exception(task)
        return

    task.cancel()
    try:
        await asyncio.wait_for(task, timeout=float(timeout_seconds))
    except Exception:
        pass

    if task.done():
        _consume_background_task_exception(task)
    else:
        try:
            task.add_done_callback(_consume_background_task_exception)
        except Exception:
            pass
