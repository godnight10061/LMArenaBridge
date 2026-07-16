"""Zero-configuration temporary-mail providers used by managed signup."""

from __future__ import annotations

import asyncio
import html
import os
import secrets
import string
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx


class TempMailError(RuntimeError):
    """Base error for normalized temporary-mail failures."""

    code = "temp_mail_error"


class TempMailUnavailable(TempMailError):
    code = "provider_unavailable"


class TempMailTimeout(TempMailError):
    code = "message_timeout"


class TempMailMalformedMessage(TempMailError):
    code = "malformed_message"


@dataclass
class Mailbox:
    provider: str
    address: str
    password: str = ""
    state: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationMessage:
    provider: str
    subject: str
    verification_url: str


class TempMailProvider(Protocol):
    name: str

    async def create_mailbox(self) -> Mailbox: ...

    async def wait_for_verification(
        self, mailbox: Mailbox, *, timeout: float
    ) -> VerificationMessage: ...

    async def close_mailbox(self, mailbox: Mailbox) -> None: ...


def _random_local_part(prefix: str = "lma") -> str:
    alphabet = string.ascii_lowercase + string.digits
    return prefix + "".join(secrets.choice(alphabet) for _ in range(14))


def _random_password() -> str:
    return secrets.token_urlsafe(24) + "A1!"


def extract_verification_url(*parts: str) -> str:
    """Extract Arena's verification callback without exposing provider formats."""
    import re

    content = html.unescape("\n".join(str(part or "") for part in parts))
    urls = re.findall(r"https?://[^\s\"'<>]+", content)
    cleaned = [url.rstrip(".,);]}") for url in urls]
    preferred = (
        "arena.ai/nextjs-api/callback/email",
        "arena.ai/auth/set-password",
    )
    for marker in preferred:
        for url in cleaned:
            if marker in url:
                return url
    raise TempMailMalformedMessage("Arena verification URL was not found")


class MailTmProvider:
    name = "mail.tm"

    def __init__(self, *, base_url: str | None = None, timeout: float = 20.0):
        self.base_url = (
            base_url or os.environ.get("LM_MAIL_TM_BASE_URL") or "https://api.mail.tm"
        ).rstrip("/")
        self.timeout = timeout

    async def create_mailbox(self) -> Mailbox:
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/domains?page=1")
                response.raise_for_status()
                domains = response.json().get("hydra:member") or []
                active = [item.get("domain") for item in domains if item.get("isActive")]
                if not active:
                    raise TempMailUnavailable("mail.tm returned no active domains")

                address = f"{_random_local_part()}@{active[0]}"
                password = _random_password()
                payload = {"address": address, "password": password}
                created = await client.post(f"{self.base_url}/accounts", json=payload)
                created.raise_for_status()
                token_response = await client.post(f"{self.base_url}/token", json=payload)
                token_response.raise_for_status()
                token = str(token_response.json().get("token") or "")
                if not token:
                    raise TempMailUnavailable("mail.tm returned no mailbox token")
                return Mailbox(
                    provider=self.name,
                    address=address,
                    password=password,
                    state={
                        "token": token,
                        "account_id": created.json().get("id"),
                        "base_url": self.base_url,
                    },
                )
        except TempMailError:
            raise
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            raise TempMailUnavailable("mail.tm mailbox creation failed") from exc

    async def wait_for_verification(
        self, mailbox: Mailbox, *, timeout: float
    ) -> VerificationMessage:
        deadline = asyncio.get_running_loop().time() + timeout
        headers = {"Authorization": f"Bearer {mailbox.state.get('token', '')}"}
        seen: set[str] = set()
        try:
            async with httpx.AsyncClient(timeout=self.timeout, headers=headers) as client:
                while asyncio.get_running_loop().time() < deadline:
                    response = await client.get(f"{self.base_url}/messages?page=1")
                    response.raise_for_status()
                    for item in response.json().get("hydra:member") or []:
                        message_id = str(item.get("id") or "")
                        if not message_id or message_id in seen:
                            continue
                        seen.add(message_id)
                        detail = await client.get(f"{self.base_url}/messages/{message_id}")
                        detail.raise_for_status()
                        payload = detail.json()
                        subject = str(payload.get("subject") or "")
                        combined_html = "\n".join(payload.get("html") or [])
                        try:
                            url = extract_verification_url(
                                str(payload.get("text") or ""), combined_html
                            )
                        except TempMailMalformedMessage:
                            continue
                        return VerificationMessage(self.name, subject, url)
                    await asyncio.sleep(3)
        except TempMailError:
            raise
        except (httpx.HTTPError, ValueError) as exc:
            raise TempMailUnavailable("mail.tm message polling failed") from exc
        raise TempMailTimeout("mail.tm verification message timed out")

    async def close_mailbox(self, mailbox: Mailbox) -> None:
        account_id = str(mailbox.state.get("account_id") or "")
        token = str(mailbox.state.get("token") or "")
        if not account_id or not token:
            return
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout, headers={"Authorization": f"Bearer {token}"}
            ) as client:
                await client.delete(f"{self.base_url}/accounts/{account_id}")
        except httpx.HTTPError:
            return


class GuerrillaMailProvider:
    name = "guerrillamail"

    def __init__(self, *, base_url: str | None = None, timeout: float = 20.0):
        self.base_url = (
            base_url
            or os.environ.get("LM_GUERRILLA_BASE_URL")
            or "https://api.guerrillamail.com/ajax.php"
        )
        self.timeout = timeout

    async def create_mailbox(self) -> Mailbox:
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    self.base_url,
                    params={"f": "get_email_address", "lang": "en"},
                )
                response.raise_for_status()
                payload = response.json()
                address = str(payload.get("email_addr") or "")
                sid_token = str(payload.get("sid_token") or "")
                if not address or not sid_token:
                    raise TempMailUnavailable(
                        "Guerrilla Mail returned incomplete mailbox state"
                    )
                return Mailbox(
                    provider=self.name,
                    address=address,
                    password=_random_password(),
                    state={"sid_token": sid_token, "base_url": self.base_url},
                )
        except TempMailError:
            raise
        except (httpx.HTTPError, ValueError) as exc:
            raise TempMailUnavailable("Guerrilla Mail mailbox creation failed") from exc

    async def wait_for_verification(
        self, mailbox: Mailbox, *, timeout: float
    ) -> VerificationMessage:
        deadline = asyncio.get_running_loop().time() + timeout
        sid_token = str(mailbox.state.get("sid_token") or "")
        seen: set[str] = set()
        sequence = 0
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                while asyncio.get_running_loop().time() < deadline:
                    response = await client.get(
                        self.base_url,
                        params={
                            "f": "check_email",
                            "seq": sequence,
                            "sid_token": sid_token,
                        },
                    )
                    response.raise_for_status()
                    payload = response.json()
                    sequence = int(payload.get("seq") or sequence)
                    for item in payload.get("list") or []:
                        message_id = str(item.get("mail_id") or "")
                        if not message_id or message_id in seen:
                            continue
                        seen.add(message_id)
                        detail = await client.get(
                            self.base_url,
                            params={
                                "f": "fetch_email",
                                "email_id": message_id,
                                "sid_token": sid_token,
                            },
                        )
                        detail.raise_for_status()
                        message = detail.json()
                        subject = str(message.get("mail_subject") or "")
                        try:
                            url = extract_verification_url(
                                str(message.get("mail_body") or ""),
                                str(message.get("mail_excerpt") or ""),
                            )
                        except TempMailMalformedMessage:
                            continue
                        return VerificationMessage(self.name, subject, url)
                    await asyncio.sleep(3)
        except TempMailError:
            raise
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            raise TempMailUnavailable("Guerrilla Mail message polling failed") from exc
        raise TempMailTimeout("Guerrilla Mail verification message timed out")

    async def close_mailbox(self, mailbox: Mailbox) -> None:
        return


def build_provider_chain() -> list[TempMailProvider]:
    configured = os.environ.get("LM_TEMP_MAIL_PROVIDERS", "mail.tm,guerrillamail")
    names = [part.strip().lower() for part in configured.split(",") if part.strip()]
    providers: list[TempMailProvider] = []
    for name in names:
        if name in {"mail.tm", "mailtm"}:
            providers.append(MailTmProvider())
        elif name in {"guerrillamail", "guerrilla"}:
            providers.append(GuerrillaMailProvider())
    return providers or [MailTmProvider(), GuerrillaMailProvider()]


async def create_mailbox_with_fallback() -> tuple[TempMailProvider, Mailbox]:
    errors: list[str] = []
    for provider in build_provider_chain():
        try:
            return provider, await provider.create_mailbox()
        except TempMailError as exc:
            errors.append(f"{provider.name}:{exc.code}")
    raise TempMailUnavailable(
        "All temporary-mail providers failed: " + ", ".join(errors)
    )
