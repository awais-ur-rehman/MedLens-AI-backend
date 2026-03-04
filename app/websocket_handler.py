"""Bidirectional WebSocket handler bridging Flutter ↔ Gemini Live API.

Message protocol
=================

**Client → Backend (JSON)**::

    {"type": "start_session"}
    {"type": "end_session"}
    {"type": "text", "content": "..."}
    {"type": "image_frame", "data": "<base64>"}
    {"type": "barge_in"}

**Client → Backend (Binary)**::

    Raw 16-bit PCM audio chunks @ 16 kHz

**Backend → Client**::

    Binary frames  → PCM audio from Dr. Muhammad
    JSON frames    → {"type": "transcript", "text": "..."} (text responses)
                     {"type": "care_summary", "data": {...}}  (session summary)
                     {"type": "session_started"}
                     {"type": "session_ended"}
                     {"type": "error", "message": "..."}
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from app.config import settings
from app.gemini_live_client import GeminiLiveClient
from app.tools.search_tool import GoogleSearchGroundingParser

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Safety keywords — if any appear in the model's text output the client UI
# should highlight them (we tag the message so Flutter can style it).
# ---------------------------------------------------------------------------
_ESCALATION_KEYWORDS = frozenset(
    [
        "call emergency",
        "call 911",
        "call an ambulance",
        "emergency services",
        "go to the er",
        "go to the emergency room",
    ]
)


def _check_escalation(text: str) -> bool:
    """Return True if the text contains an escalation trigger."""
    lower = text.lower()
    return any(kw in lower for kw in _ESCALATION_KEYWORDS)


# ---------------------------------------------------------------------------
# Gemini → Flutter forwarding task
# ---------------------------------------------------------------------------

async def _forward_gemini_to_client(
    gemini: GeminiLiveClient,
    ws: WebSocket,
    transcript_parts: list[str],
) -> None:
    """Background task: read from Gemini and push to the Flutter client.

    Runs until the Gemini stream ends or an error occurs.
    """
    try:
        async for chunk in gemini.receive_stream():
            chunk_type = chunk["type"]

            # ---- Audio → send as binary frame ------------------------------
            if chunk_type == "audio":
                await ws.send_bytes(chunk["data"])

            # ---- Text → safety check, then send as JSON --------------------
            elif chunk_type == "text":
                text = chunk["text"]
                transcript_parts.append(text)

                payload: dict[str, Any] = {
                    "type": "transcript",
                    "text": text,
                }
                if _check_escalation(text):
                    payload["escalation"] = True

                await ws.send_json(payload)

                # -- Forward Google Search grounding citations ---------------
                grounding_meta = chunk.get("grounding_metadata")
                if grounding_meta and GoogleSearchGroundingParser.has_grounding(
                    grounding_meta
                ):
                    citations = GoogleSearchGroundingParser.extract_citations(
                        grounding_meta
                    )
                    if citations:
                        await ws.send_json({
                            "type": "citation",
                            "sources": citations,
                        })

            # ---- Tool call → handle internally, respond to Gemini ----------
            elif chunk_type == "tool_call":
                fn_call = chunk["data"]
                tool_name = getattr(fn_call, "name", str(fn_call))
                logger.info("Tool call received: %s", tool_name)

                # Notify Flutter that the agent is thinking
                await ws.send_json({
                    "type": "agent_thinking",
                    "tool": tool_name,
                })

                # Google Search grounding is handled automatically by the
                # Live API — no manual tool response needed.  If custom tools
                # are added later, dispatch them here and call
                # gemini.send_tool_response([...]).

    except asyncio.CancelledError:
        logger.debug("Gemini forwarding task cancelled")
    except Exception:
        logger.exception("Error in Gemini → client forwarding")
        try:
            await ws.send_json({"type": "error", "message": "Stream interrupted"})
        except Exception:
            pass  # Client may already be gone


# ---------------------------------------------------------------------------
# Session summary generator
# ---------------------------------------------------------------------------

def _build_care_summary(transcript_parts: list[str]) -> dict[str, Any]:
    """Build a simple care summary from the collected transcript."""
    full_text = " ".join(transcript_parts).strip()

    return {
        "session_transcript": full_text,
        "word_count": len(full_text.split()) if full_text else 0,
        "had_escalation": _check_escalation(full_text),
        # TODO: Use Gemini Flash to generate a structured care summary
        #       with key findings, actions taken, and follow-up recommendations.
    }


# ---------------------------------------------------------------------------
# Main handler
# ---------------------------------------------------------------------------

async def handle_session(websocket: WebSocket) -> None:
    """Handle a single MedLens session over WebSocket.

    Called from the FastAPI route.  Manages the full lifecycle:
    client connect → Gemini session → forwarding → cleanup.
    """
    await websocket.accept()
    logger.info("Client WebSocket connected")

    gemini: GeminiLiveClient | None = None
    forward_task: asyncio.Task | None = None
    transcript_parts: list[str] = []

    try:
        while True:
            message = await websocket.receive()

            # ---- Binary frame → raw PCM audio ----------------------------
            if "bytes" in message and message["bytes"]:
                pcm_data: bytes = message["bytes"]
                if gemini and gemini.is_connected:
                    await gemini.send_audio(pcm_data)
                continue

            # ---- Text frame → JSON control message -----------------------
            raw_text: str | None = message.get("text")
            if not raw_text:
                continue

            try:
                data: dict[str, Any] = json.loads(raw_text)
            except json.JSONDecodeError:
                await websocket.send_json(
                    {"type": "error", "message": "Invalid JSON"}
                )
                continue

            msg_type = data.get("type", "")

            # ── start_session ─────────────────────────────────────────────
            if msg_type == "start_session":
                if gemini and gemini.is_connected:
                    await websocket.send_json(
                        {"type": "error", "message": "Session already active"}
                    )
                    continue

                try:
                    gemini = GeminiLiveClient(settings)
                    await gemini.connect()
                    transcript_parts.clear()

                    # Kick off background forwarding Gemini → Client
                    forward_task = asyncio.create_task(
                        _forward_gemini_to_client(gemini, websocket, transcript_parts)
                    )

                    await websocket.send_json({"type": "session_started"})
                    logger.info("Gemini Live session started")

                except Exception as exc:
                    logger.exception("Failed to start Gemini session")
                    await websocket.send_json(
                        {"type": "error", "message": f"Connection failed: {exc}"}
                    )
                    gemini = None

            # ── end_session ───────────────────────────────────────────────
            elif msg_type == "end_session":
                summary = _build_care_summary(transcript_parts)
                await websocket.send_json({"type": "care_summary", "data": summary})

                if forward_task and not forward_task.done():
                    forward_task.cancel()
                    try:
                        await forward_task
                    except asyncio.CancelledError:
                        pass

                if gemini:
                    await gemini.disconnect()
                    gemini = None

                forward_task = None
                transcript_parts.clear()

                await websocket.send_json({"type": "session_ended"})
                logger.info("Gemini Live session ended by client")

            # ── text ──────────────────────────────────────────────────────
            elif msg_type == "text":
                content = data.get("content", "")
                if gemini and gemini.is_connected and content:
                    await gemini.send_text(content)
                elif not gemini or not gemini.is_connected:
                    await websocket.send_json(
                        {"type": "error", "message": "No active session"}
                    )

            # ── image_frame ───────────────────────────────────────────────
            elif msg_type == "image_frame":
                b64_data = data.get("data", "")
                if gemini and gemini.is_connected and b64_data:
                    try:
                        jpeg_bytes = base64.b64decode(b64_data)
                        await gemini.send_image(jpeg_bytes)
                    except Exception:
                        logger.exception("Failed to decode/send image frame")
                        await websocket.send_json(
                            {"type": "error", "message": "Invalid image data"}
                        )
                elif not gemini or not gemini.is_connected:
                    await websocket.send_json(
                        {"type": "error", "message": "No active session"}
                    )

            # ── barge_in ──────────────────────────────────────────────────
            elif msg_type == "barge_in":
                if gemini and gemini.is_connected:
                    await gemini.send_barge_in()

            # ── unknown ──────────────────────────────────────────────────
            else:
                await websocket.send_json(
                    {"type": "error", "message": f"Unknown message type: {msg_type}"}
                )

    except WebSocketDisconnect:
        logger.info("Client WebSocket disconnected")

    except Exception:
        logger.exception("Unexpected error in WebSocket handler")

    finally:
        # ---- Cleanup --------------------------------------------------
        if forward_task and not forward_task.done():
            forward_task.cancel()
            try:
                await forward_task
            except asyncio.CancelledError:
                pass

        if gemini and gemini.is_connected:
            try:
                await gemini.disconnect()
            except Exception:
                logger.warning("Error disconnecting Gemini on cleanup", exc_info=True)

        logger.info("WebSocket handler fully cleaned up")
