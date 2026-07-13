"""JSON extraction and validation from LLM output text."""

from __future__ import annotations

import json
import re
import unicodedata

# Unicode 控制字符范围
_UNICODE_C0_START = 0x00
_UNICODE_C0_END = 0x1F
_UNICODE_DEL = 0x7F
_UNICODE_C1_START = 0x80
_UNICODE_C1_END = 0x9F


def _sanitize_control_chars(text: str) -> str:
    sanitized = []
    for ch in text:
        cp = ord(ch)
        if (
            _UNICODE_C0_START <= cp <= _UNICODE_C0_END
            or _UNICODE_C1_START <= cp <= _UNICODE_C1_END
            or unicodedata.category(ch).startswith("C")
        ):
            sanitized.append(" ")
        else:
            sanitized.append(ch)
    return "".join(sanitized)


def _extract_json_from_text(text: str) -> dict | None:
    """Extract JSON object from LLM output text (robust, fail-closed).

    Supports:
      - ```json ... ```  blocks
      - ``` ... ```       blocks (plain)
      - Raw {"...": ...}  objects

    Returns None on any parse failure.
    """
    if not text:
        return None
    text = text.strip()
    patterns = [
        r"```json\s*(\{.*?\})\s*```",
        r"```\s*(\{.*?\})\s*```",
        r"(\{.*\})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            continue
        candidate = match.group(1)
        candidate = _sanitize_control_chars(candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    candidate = _sanitize_control_chars(text)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def validate_json_output(
    raw_text: str,
    required_keys: list[str],
    fail_closed: bool = True,
) -> tuple[dict | None, str]:
    """Validate and parse LLM JSON output with fail-closed guarantees."""
    parsed = _extract_json_from_text(raw_text)
    if parsed is None:
        msg = f"JSON parse failed from: {raw_text[:200]}"
        return (None, msg) if fail_closed else (None, raw_text)

    if fail_closed:
        missing = [k for k in required_keys if k not in parsed]
        if missing:
            return None, f"Missing required keys {missing} in: {raw_text[:200]}"

    return parsed, ""
