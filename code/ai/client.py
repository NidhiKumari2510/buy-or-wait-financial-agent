"""
code/ai/client.py

Thin wrapper around the Anthropic API for structured-JSON extraction calls.
Two entry points: extract_json_from_text() and extract_json_from_image().
Both return (parsed_dict_or_None, usage_dict) so callers can track tokens
for evaluation/usage_report.md.

Model: claude-haiku-4-5-20251001 -- cheapest current Claude model.
Appropriate here because every call is short-text/single-image -> fixed
small JSON schema, not open-ended reasoning; deterministic code
(simulator.py, planner.py) does all the arithmetic and decision-making.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Optional

from anthropic import Anthropic

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 1024

_client: Optional[Anthropic] = None


def get_client() -> Optional[Anthropic]:
    """Returns None (not an error) if no API key is configured -- callers
    must handle that gracefully and skip AI-assisted steps rather than
    crash, so the pipeline degrades instead of failing without a key."""
    global _client
    if _client is not None:
        return _client
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    _client = Anthropic(api_key=api_key)
    return _client


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text[:-3]
        if text.lower().startswith("json"):
            text = text[4:]
    return text.strip()


def _usage_dict(response) -> dict:
    return {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


def extract_json_from_text(system_prompt: str, user_text: str) -> tuple[Optional[dict], Optional[dict]]:
    client = get_client()
    if client is None:
        return None, None
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": user_text}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    try:
        parsed = json.loads(_strip_code_fence(text))
    except (json.JSONDecodeError, IndexError):
        parsed = None
    return parsed, _usage_dict(response)


def extract_json_from_image(
    system_prompt: str, user_text: str, image_path: Path
) -> tuple[Optional[dict], Optional[dict]]:
    client = get_client()
    if client is None:
        return None, None
    image_bytes = image_path.read_bytes()
    image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": image_b64},
                    },
                    {"type": "text", "text": user_text},
                ],
            }
        ],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    try:
        parsed = json.loads(_strip_code_fence(text))
    except (json.JSONDecodeError, IndexError):
        parsed = None
    return parsed, _usage_dict(response)