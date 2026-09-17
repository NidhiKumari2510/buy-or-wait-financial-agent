"""
code/ai/client.py

Thin wrapper around the Gemini API for structured-JSON extraction calls.
Loads GEMINI_API_KEY from .env at the repo root automatically. Retries once
or twice on transient 503 UNAVAILABLE errors (Gemini free-tier capacity
spikes) before giving up and returning None, None -- callers already handle
None gracefully.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv, find_dotenv
from google import genai
from google.genai import types

_dotenv_path = find_dotenv(usecwd=True)
if _dotenv_path:
    load_dotenv(_dotenv_path)
else:
    print("[ai/client] WARNING: no .env file found via find_dotenv(). "
          "Make sure you're running commands from the repo root.")

MODEL = "gemini-3.6-flash"

_client: Optional["genai.Client"] = None


def get_client():
    global _client
    if _client is not None:
        return _client
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print(f"[ai/client] DEBUG: GEMINI_API_KEY not found after loading "
              f".env from: {_dotenv_path or 'NOT FOUND'}")
        return None
    _client = genai.Client(api_key=api_key)
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
    usage = response.usage_metadata
    return {
        "input_tokens": (usage.prompt_token_count or 0) if usage else 0,
        "output_tokens": (usage.candidates_token_count or 0) if usage else 0,
    }


def _call_with_retry(client, **kwargs):
    last_exc = None
    for attempt in range(3):
        try:
            return client.models.generate_content(**kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt < 2 and "503" in str(exc):
                wait = 2 * (attempt + 1)
                print(f"[ai/client] 503 UNAVAILABLE, retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise last_exc


def extract_json_from_text(system_prompt: str, user_text: str) -> tuple[Optional[dict], Optional[dict]]:
    client = get_client()
    if client is None:
        return None, None
    try:
        response = _call_with_retry(
            client,
            model=MODEL,
            contents=user_text,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
            ),
        )
        if not response.text:
            return None, _usage_dict(response)
        parsed = json.loads(_strip_code_fence(response.text))
        return parsed, _usage_dict(response)
    except Exception as exc:
        print(f"[ai/client] WARNING: text extraction call failed: {exc!r}")
        return None, None


def extract_json_from_image(
    system_prompt: str, user_text: str, image_path: Path
) -> tuple[Optional[dict], Optional[dict]]:
    client = get_client()
    if client is None:
        return None, None
    try:
        image_bytes = image_path.read_bytes()
        response = _call_with_retry(
            client,
            model=MODEL,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                user_text,
            ],
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
            ),
        )
        if not response.text:
            return None, _usage_dict(response)
        parsed = json.loads(_strip_code_fence(response.text))
        return parsed, _usage_dict(response)
    except Exception as exc:
        print(f"[ai/client] WARNING: image extraction call failed: {exc!r}")
        return None, None