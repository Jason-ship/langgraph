"""Shared command definitions used by all channel implementations.

Keeping the authoritative command set in one place ensures that channel
parsers and the ChannelManager dispatcher stay in sync automatically.
"""

from __future__ import annotations

KNOWN_CHANNEL_COMMANDS: frozenset[str] = frozenset(
    {
        "/bootstrap",
        "/goal",
        "/new",
        "/status",
        "/models",
        "/memory",
        "/help",
    }
)


def _is_leading_mention_token(token: str) -> bool:
    """Return whether *token* looks like a platform bot/user mention."""
    if not token:
        return False
    if token.startswith("<@") and token.endswith(">"):
        return True
    if token.startswith("@") and len(token) > 1:
        return True
    return False


def strip_leading_mentions(text: str) -> str:
    """Drop leading platform mention tokens (``@bot``, ``<@id>``)."""
    remainder = text
    while True:
        parts = remainder.split(maxsplit=1)
        if not parts or remainder[0].isspace() or not _is_leading_mention_token(parts[0]):
            break
        remainder = parts[1] if len(parts) > 1 else ""
    return remainder


def extract_connect_code(text: str) -> str | None:
    """Extract the one-time channel binding code from a connect command."""
    parts = text.strip().split()
    index = 0
    while index < len(parts) and _is_leading_mention_token(parts[index]):
        index += 1
    if index + 1 >= len(parts):
        return None
    command = parts[index].lower()
    if command == "/connect":
        return parts[index + 1]
    return None


def is_known_channel_command(text: str) -> bool:
    """Return whether text starts with a registered channel control command."""
    if not text.startswith("/"):
        return False
    return text.split(maxsplit=1)[0].lower() in KNOWN_CHANNEL_COMMANDS