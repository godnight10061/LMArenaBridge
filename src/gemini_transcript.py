"""Deterministic OpenAI message rendering for the Gemini browser transport."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Optional


DEFAULT_MAX_PROMPT_CHARACTERS = 113_567
_INSTRUCTION_ROLES = {"system", "developer"}


@dataclass(frozen=True)
class NormalizedMessage:
    index: int
    role: str
    content: str
    tool_calls: tuple[dict[str, Any], ...] = ()
    tool_call_id: str = ""


@dataclass(frozen=True)
class TranscriptResult:
    prompt: str
    source_message_count: int
    included_message_count: int
    omitted_message_count: int
    role_counts: dict[str, int]
    included_role_counts: dict[str, int]
    rendered_characters: int
    context_truncated: bool
    content_truncated: bool


def _json_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


def _content_text(content: Any, *, historical: bool = True) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (int, float, bool)):
        return str(content)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                item_type = str(item.get("type") or "")
                if item_type in {"text", "input_text", "output_text"}:
                    parts.append(str(item.get("text") or ""))
                elif item_type in {"image_url", "input_image", "image"}:
                    parts.append("[historical image omitted from Gemini transcript]")
                elif item.get("content") is not None:
                    parts.append(_content_text(item.get("content"), historical=historical))
            elif item is not None:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        if content.get("text") is not None:
            return str(content.get("text") or "")
        if content.get("content") is not None:
            return _content_text(content.get("content"), historical=historical)
        return _json_text(content)
    return str(content)


def _normalize_tool_calls(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, dict))


def normalize_messages(
    messages: Iterable[dict[str, Any]], *, latest_content: Optional[str] = None
) -> list[NormalizedMessage]:
    raw_messages = list(messages)
    normalized: list[NormalizedMessage] = []
    last_index = len(raw_messages) - 1
    for index, raw in enumerate(raw_messages):
        if not isinstance(raw, dict):
            continue
        role = str(raw.get("role") or "user")
        content = (
            latest_content
            if latest_content is not None and index == last_index
            else _content_text(raw.get("content"))
        )
        normalized.append(
            NormalizedMessage(
                index=index,
                role=role,
                content=content,
                tool_calls=_normalize_tool_calls(raw.get("tool_calls")),
                tool_call_id=str(raw.get("tool_call_id") or ""),
            )
        )
    return normalized


def _render_message(message: NormalizedMessage) -> str:
    role = message.role.upper()
    lines = [f"[{role}]"]
    if message.content:
        lines.append(message.content)
    for call in message.tool_calls:
        raw_function = call.get("function")
        function: dict[str, Any] = raw_function if isinstance(raw_function, dict) else {}
        call_id = str(call.get("id") or "")
        name = str(function.get("name") or call.get("name") or "")
        arguments = _json_text(function.get("arguments") or call.get("arguments") or {})
        lines.append(f"[TOOL_CALL id={call_id} name={name}]\n{arguments}")
    if message.tool_call_id:
        lines[0] = f"[{role} id={message.tool_call_id}]"
    return "\n".join(lines)


def _units(messages: list[NormalizedMessage]) -> list[list[NormalizedMessage]]:
    result: list[list[NormalizedMessage]] = []
    index = 0
    while index < len(messages):
        current = messages[index]
        unit = [current]
        index += 1
        if current.role == "assistant" and current.tool_calls:
            while index < len(messages) and messages[index].role == "tool":
                unit.append(messages[index])
                index += 1
        result.append(unit)
    return result


def _render(instructions: list[NormalizedMessage], units: list[list[NormalizedMessage]]) -> str:
    messages = [*instructions, *(message for unit in units for message in unit)]
    messages.sort(key=lambda message: message.index)
    blocks = [_render_message(message) for message in messages]
    return "\n\n".join(block for block in blocks if block).strip()


def _counts(messages: Iterable[NormalizedMessage]) -> dict[str, int]:
    return dict(Counter(message.role for message in messages))


def build_transcript(
    messages: Iterable[dict[str, Any]],
    *,
    latest_content: Optional[str] = None,
    max_characters: int = DEFAULT_MAX_PROMPT_CHARACTERS,
) -> TranscriptResult:
    normalized = normalize_messages(messages, latest_content=latest_content)
    instructions = [message for message in normalized if message.role in _INSTRUCTION_ROLES]
    conversation = [message for message in normalized if message.role not in _INSTRUCTION_ROLES]
    units = _units(conversation)
    source_counts = _counts(normalized)

    # Preserve the existing single-turn API behavior: a plain user prompt remains plain text.
    if not instructions and len(units) == 1 and len(units[0]) == 1:
        prompt = _render_message(units[0][0])
        if units[0][0].role == "user" and not units[0][0].tool_calls:
            prompt = units[0][0].content
        return TranscriptResult(
            prompt=prompt,
            source_message_count=len(normalized),
            included_message_count=len(normalized),
            omitted_message_count=0,
            role_counts=source_counts,
            included_role_counts=source_counts,
            rendered_characters=len(prompt),
            context_truncated=False,
            content_truncated=False,
        )

    selected: list[list[NormalizedMessage]] = []
    context_truncated = False
    content_truncated = False
    prompt = _render(instructions, units)
    if len(prompt) > max_characters:
        context_truncated = True
        for unit in reversed(units):
            candidate = [unit, *selected]
            candidate_prompt = _render(instructions, candidate)
            if len(candidate_prompt) <= max_characters or not selected:
                selected = candidate
            else:
                break
        prompt = _render(instructions, selected)

        # Keep the newest unit even when that single unit exceeds the bound; trim only its
        # textual content, preserving tool-call/result structure and explicit diagnostics.
        if len(prompt) > max_characters and selected:
            newest = selected[-1]
            trimmed_unit = [
                NormalizedMessage(
                    index=message.index,
                    role=message.role,
                    content=message.content,
                    tool_calls=message.tool_calls,
                    tool_call_id=message.tool_call_id,
                )
                for message in newest
            ]
            marker = "\n[content truncated]"
            while len(_render(instructions, [trimmed_unit])) > max_characters:
                excess = len(_render(instructions, [trimmed_unit])) - max_characters
                trim_index = next(
                    (
                        index
                        for index in range(len(trimmed_unit) - 1, -1, -1)
                        if trimmed_unit[index].content
                    ),
                    None,
                )
                if trim_index is None:
                    break
                message = trimmed_unit[trim_index]
                keep = max(0, len(message.content) - excess - len(marker))
                text = message.content[:keep] + marker if keep else ""
                trimmed_unit[trim_index] = (
                    NormalizedMessage(
                        index=message.index,
                        role=message.role,
                        content=text,
                        tool_calls=message.tool_calls,
                        tool_call_id=message.tool_call_id,
                    )
                )
                content_truncated = True
            selected = [trimmed_unit]
            prompt = _render(instructions, selected)
    else:
        selected = units

    included = [*instructions, *(message for unit in selected for message in unit)]
    return TranscriptResult(
        prompt=prompt,
        source_message_count=len(normalized),
        included_message_count=len(included),
        omitted_message_count=max(0, len(normalized) - len(included)),
        role_counts=source_counts,
        included_role_counts=_counts(included),
        rendered_characters=len(prompt),
        context_truncated=context_truncated,
        content_truncated=content_truncated,
    )
