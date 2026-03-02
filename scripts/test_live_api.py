#!/usr/bin/env python3
"""Smoke-test for the Gemini Live API integration.

Usage (from the backend/ directory):
    python -m scripts.test_live_api

Or from the project root:
    PYTHONPATH=backend python backend/scripts/test_live_api.py

Requires a valid .env (or environment variables) with at least:
    GOOGLE_CLOUD_PROJECT=<your-project>
    GOOGLE_CLOUD_LOCATION=us-central1
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

# ---------------------------------------------------------------------------
# Make sure the backend package is importable regardless of cwd
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_SCRIPT_DIR)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from app.config import Settings  # noqa: E402
from app.gemini_live_client import GeminiLiveClient  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TEST_MESSAGE = (
    "Hello, I just burned my hand on the stove. "
    "It is red and it hurts."
)
RECEIVE_TIMEOUT_SECONDS = 30  # max time to wait for the full response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_header(title: str) -> None:
    width = 60
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def _print_check(label: str, passed: bool) -> None:
    icon = "✅" if passed else "❌"
    print(f"  {icon}  {label}")


# ---------------------------------------------------------------------------
# Main test coroutine
# ---------------------------------------------------------------------------

async def main() -> None:
    _print_header("MedLens — Gemini Live API Smoke Test")

    # ---- Load config -------------------------------------------------------
    try:
        config = Settings()
    except Exception as exc:
        print(f"\n❌  Failed to load config: {exc}")
        print("    Make sure you have a .env file in backend/ or the")
        print("    required environment variables are set.\n")
        sys.exit(1)

    if not config.google_cloud_project:
        print("\n❌  GOOGLE_CLOUD_PROJECT is not set.")
        print("    Add it to your .env or export it as an env var.\n")
        sys.exit(1)

    print(f"\n  Project  : {config.google_cloud_project}")
    print(f"  Location : {config.google_cloud_location}")
    print(f"  Model    : {config.gemini_model}")

    # ---- Connect -----------------------------------------------------------
    client = GeminiLiveClient(config)

    _print_header("Connecting to Gemini Live API")
    try:
        await client.connect()
        print("  ✅  Connected successfully\n")
    except Exception as exc:
        print(f"\n  ❌  Connection failed: {exc}\n")
        print("  Troubleshooting:")
        print("   • Is GOOGLE_CLOUD_PROJECT correct?")
        print("   • Have you run `gcloud auth application-default login`?")
        print("   • Is the Vertex AI API enabled in your project?")
        print(f"   • Does the model '{config.gemini_model}' exist in your region?\n")
        sys.exit(1)

    # ---- Send test message -------------------------------------------------
    _print_header("Sending Test Message")
    print(f'  → "{TEST_MESSAGE}"\n')

    try:
        await client.send_text(TEST_MESSAGE)
    except Exception as exc:
        print(f"  ❌  Failed to send text: {exc}")
        await client.disconnect()
        sys.exit(1)

    # ---- Collect responses --------------------------------------------------
    _print_header("Receiving Response")

    text_chunks: list[str] = []
    audio_chunks: int = 0
    tool_calls: int = 0
    start = time.monotonic()

    try:
        async for chunk in client.receive_stream():
            elapsed = time.monotonic() - start

            if chunk["type"] == "text":
                text_chunks.append(chunk["text"])
                print(f"  [text]  {chunk['text']}")

            elif chunk["type"] == "audio":
                audio_chunks += 1
                size = len(chunk["data"])
                print(f"  [audio] chunk #{audio_chunks} — {size} bytes")

            elif chunk["type"] == "tool_call":
                tool_calls += 1
                print(f"  [tool]  {chunk['data']}")

            # Stop after timeout to avoid hanging forever
            if elapsed > RECEIVE_TIMEOUT_SECONDS:
                print(f"\n  ⏱  Timeout after {RECEIVE_TIMEOUT_SECONDS}s — stopping receive loop")
                break

    except Exception as exc:
        print(f"\n  ⚠️  Error during receive: {exc}")

    # ---- Full response text ------------------------------------------------
    full_response = " ".join(text_chunks).strip()

    _print_header("Full Text Response")
    if full_response:
        print(f"\n{full_response}\n")
    else:
        print("\n  (no text received — audio-only response)\n")

    # ---- Stats -------------------------------------------------------------
    _print_header("Response Stats")
    print(f"  Text chunks  : {len(text_chunks)}")
    print(f"  Audio chunks : {audio_chunks}")
    print(f"  Tool calls   : {tool_calls}")

    # ---- Verification checks -----------------------------------------------
    _print_header("Persona Verification")

    response_lower = full_response.lower()

    has_name = "muhammad" in response_lower or "dr." in response_lower
    _print_check("Dr. Muhammad name mentioned", has_name)

    disclaimer_keywords = ["not a replacement", "not a substitute", "professional", "emergency", "disclaimer"]
    has_disclaimer = any(kw in response_lower for kw in disclaimer_keywords)
    _print_check("Medical disclaimer present", has_disclaimer)

    burn_keywords = ["burn", "cool", "water", "cold", "running water", "ice"]
    has_burn_advice = any(kw in response_lower for kw in burn_keywords)
    _print_check("Relevant burn first-aid advice", has_burn_advice)

    has_response = bool(full_response) or audio_chunks > 0
    _print_check("Received a response (text or audio)", has_response)

    all_passed = has_name and has_disclaimer and has_burn_advice and has_response

    # ---- Disconnect --------------------------------------------------------
    _print_header("Disconnecting")
    try:
        await client.disconnect()
        print("  ✅  Disconnected cleanly\n")
    except Exception as exc:
        print(f"  ⚠️  Disconnect error: {exc}\n")

    # ---- Final result ------------------------------------------------------
    _print_header("Result")
    if all_passed:
        print("  🎉  ALL CHECKS PASSED — Dr. Muhammad is live!\n")
    else:
        print("  ⚠️  Some checks failed. Review the response above.")
        print("     Note: audio-only responses will fail text checks.\n")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    asyncio.run(main())
