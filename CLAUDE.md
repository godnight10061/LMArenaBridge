# CLAUDE.md

Project guidance for assistants working on LM Arena Bridge.

## Commands

```powershell
# Both entry points are supported.
python src/main.py
python -m src.main

# Regression checks.
python -m pytest -q
python -m compileall -q src tests
python -m ruff check src tests

# Real bridge validation.
python src/live_gemini_check.py `
  --base-url http://127.0.0.1:8000/api/v1 `
  --model gemini-3.5-flash `
  --minimum-model-successes 5 `
  --model-candidate-limit 8
```

## Architecture

The FastAPI bridge exposes OpenAI-compatible endpoints and uses Chrome/Playwright
for Arena catalog discovery, rendered model requests, reCAPTCHA, and managed
account recovery.

- `src/main.py`: routes, streaming orchestration, startup, catalog normalization,
  and Chrome UI transport.
- `src/account_recovery.py`: authenticated-cookie inspection, Chrome/CDP session
  ownership, login, temporary-mail signup, and atomic account activation.
- `src/temp_mail.py`: zero-configuration `mail.tm` and Guerrilla Mail adapters.
- `src/recaptcha.py`: Chrome-backed reCAPTCHA discovery, minting, and caching.
- `src/transport.py`: direct Chrome fetch and optional external userscript-proxy
  transport support.
- `src/browser_window.py`: headed background/visible Chrome placement.
- `src/browser_utils.py`: Turnstile interaction and browser-task cleanup.
- `src/gemini_transcript.py`: deterministic OpenAI-message transcript rendering.
- `src/live_gemini_check.py`: real Gemini plus provider-diverse live validation.

## Runtime contracts

- `src.main` is the canonical module identity for both startup commands. Modules
  with `_m()` late imports must resolve that same object so process-wide locks and
  token state are not duplicated.
- Startup fetches Arena's current `initialModels` payload through Chrome, selects
  one preferred organized/selectable variant per `publicName`, and atomically
  writes the ignored `models.json` cache.
- `/v1/models` and `/api/v1/models` expose every organized public name once.
- A live CDP endpoint is borrowed instead of launching a competing persistent
  context against the same profile. Borrowed contexts are not closed by recovery.
- `LM_BROWSER_WINDOW_MODE=background` keeps Chrome headed, minimized, and
  off-screen. `visible` is the local diagnostic override.
- Managed recovery accepts only decoded authenticated, non-anonymous Arena state.
- The optional userscript proxy is active only while a real external poller is
  present. Startup does not create an internal proxy browser.

## Testing

Pure component tests cover catalog normalization, transcript construction, model
selection, authentication inspection, browser ownership, and window arguments.
Both entry points are exercised as real subprocesses. Browser/catalog behavior is
ultimately gated by the Windows live job, which makes real bridge calls to
`gemini-3.5-flash` and additional Arena models without mocked browser or upstream
responses.

Generated credentials, browser profiles, Trellis state, and model caches remain
outside source control.
