"""Bidirectional WebSocket handler bridging Flutter ↔ Gemini Live API.

Message protocol
=================

**Client → Backend (JSON)**::

    {"type": "start_session"}
    {"type": "end_session"}
    {"type": "text", "content": "..."}
    {"type": "image_frame", "data": "<base64>"}
    {"type": "barge_in"}
    {"type": "end_of_turn"}

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
from datetime import datetime, timezone
from typing import Any

from google import genai
from google.genai import types

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
                logger.debug("Gemini → audio %d bytes", len(chunk["data"]))
                await ws.send_bytes(chunk["data"])

            # ---- Text → safety check, then send as JSON --------------------
            elif chunk_type == "text":
                text = chunk["text"]
                logger.info("Gemini → text: %s", text[:80])
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

            # ---- User Text → send as JSON ----------------------------------
            elif chunk_type == "user_text":
                text = chunk["text"]
                logger.info("Gemini heard user: %s", text[:80])
                transcript_parts.append("[User] " + text)
                await ws.send_json({
                    "type": "user_transcript",
                    "text": text,
                })

            # ---- Turn Complete ---------------------------------------------
            elif chunk_type == "turn_complete":
                logger.info("Gemini → turn_complete speaker=%s", chunk.get("speaker", "agent"))
                await ws.send_json({
                    "type": "turn_complete",
                    "speaker": chunk.get("speaker", "agent")
                })

            # ---- Tool call → handle internally, respond to Gemini ----------
            elif chunk_type == "tool_call":
                fn_call = chunk["data"]
                tool_name = getattr(fn_call, "name", str(fn_call))
                logger.info("Tool call received: %s", tool_name)

                if tool_name in ["request_camera", "request_live_camera"]:
                    logger.info("Forwarding %s request to Flutter", tool_name)

                    args = getattr(fn_call, "args", {})
                    if hasattr(args, "get"):
                        prompt = args.get("prompt", "Show me.")
                        duration = args.get("duration_seconds", 10)
                    else:
                        prompt = getattr(args, "prompt", "Show me.")
                        duration = getattr(args, "duration_seconds", 10)

                    payload = {
                        "type": tool_name,
                        "prompt": prompt,
                    }
                    if tool_name == "request_live_camera":
                        payload["duration_seconds"] = duration

                    await ws.send_json(payload)

                    # Respond to Gemini so it can continue
                    await gemini.send_tool_response([
                        types.FunctionResponse(
                            name=tool_name,
                            id=getattr(fn_call, "id", ""),
                            response={"result": "Request sent to user's device. Waiting for image."},
                        )
                    ])
                else:
                    await ws.send_json({
                        "type": "agent_thinking",
                        "tool": tool_name,
                    })

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

def _build_care_summary_fallback(transcript_parts: list[str]) -> dict[str, Any]:
    """Minimal fallback summary when Gemini Flash is unavailable."""
    return {
        "session_id": "",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "injury_type": "First Aid Consultation",
        "severity": "low",
        "patient_description": " ".join(transcript_parts[:3]).strip() or "Session completed.",
        "actions_taken": ["Please review the session transcript for details."],
        "medications_discussed": [],
        "follow_up_recommendations": ["Consult a healthcare provider if symptoms persist."],
        "warning_signs": ["Seek emergency care if symptoms worsen."],
        "disclaimer": (
            "This was first aid guidance only. Dr. Muhammad is not a substitute for "
            "professional medical care. If symptoms worsen or you are unsure, please "
            "contact a healthcare provider or call emergency services."
        ),
    }


async def _generate_care_summary(transcript_parts: list[str]) -> dict[str, Any]:
    """Generate a structured care summary using Gemini Flash.

    Produces JSON that matches Flutter's CareSummaryModel.fromJson expectations.
    Falls back gracefully if the model call fails.
    """
    full_transcript = " ".join(transcript_parts).strip()
    if not full_transcript:
        return _build_care_summary_fallback(transcript_parts)

    prompt = f"""You are a medical records assistant for a first aid app called MedLens.
A patient just completed a consultation with Dr. Muhammad, an AI first aid assistant.

CONSULTATION TRANSCRIPT:
{full_transcript}

Generate a structured care summary as a JSON object. Return ONLY valid JSON — no markdown, no code fences, no explanation.

Use exactly this structure:
{{
  "session_id": "",
  "timestamp": "{datetime.now(timezone.utc).isoformat()}",
  "injury_type": "One short phrase describing the injury (e.g. First-degree burn, Laceration, Sprained ankle)",
  "severity": "low",
  "patient_description": "One sentence describing what happened based on the transcript.",
  "actions_taken": [
    "First aid step 1 that was recommended",
    "First aid step 2 that was recommended"
  ],
  "medications_discussed": [],
  "follow_up_recommendations": [
    "When to see a doctor",
    "Any follow-up care needed"
  ],
  "warning_signs": [
    "Symptom A that means seek immediate care",
    "Symptom B that means seek immediate care"
  ],
  "disclaimer": "This was first aid guidance only. Dr. Muhammad is not a substitute for professional medical care. If symptoms worsen or you are unsure, please contact a healthcare provider or call emergency services."
}}

For severity: use "low" for minor cuts/burns/bruises, "medium" for moderate injuries requiring monitoring, "high" for serious injuries needing prompt medical attention."""

    try:
        client = genai.Client(
            vertexai=True,
            project=settings.google_cloud_project,
            location=settings.google_cloud_location,
        )
        response = await asyncio.wait_for(
            client.aio.models.generate_content(
                model=settings.gemini_flash_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                ),
            ),
            timeout=12.0,
        )
        return json.loads(response.text)
    except asyncio.TimeoutError:
        logger.warning("Care summary generation timed out — using fallback")
        return _build_care_summary_fallback(transcript_parts)
    except Exception:
        logger.exception("Gemini Flash care summary failed — using fallback")
        return _build_care_summary_fallback(transcript_parts)


# ---------------------------------------------------------------------------
# Main handler
# ---------------------------------------------------------------------------

async def handle_session(websocket: WebSocket) -> None:
    """Handle a single MedLens session over WebSocket."""
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
                    try:
                        await gemini.send_audio(pcm_data)
                    except Exception:
                        logger.exception("Gemini connection lost while sending audio")
                        gemini._session = None  # Mark as disconnected
                        await websocket.send_json({
                            "type": "error",
                            "message": "Connection to AI lost. Please end and restart the session.",
                        })
                else:
                    logger.warning("Audio received but no active Gemini session")
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

                    # Start background task forwarding Gemini → Flutter
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
                    continue

                # Trigger greeting in a separate try so failures don't kill the session.
                # The Live API won't speak first — we send a "." to prompt it.
                try:
                    await gemini.trigger_greeting()
                except Exception:
                    logger.warning("Greeting trigger failed — session still active", exc_info=True)

            # ── end_session ───────────────────────────────────────────────
            elif msg_type == "end_session":
                summary = await _generate_care_summary(transcript_parts)
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

            # ── activity_start ────────────────────────────────────────────
            # Sent when user taps mic to begin speaking (manual VAD mode).
            elif msg_type == "activity_start":
                logger.info("Client sent activity_start — forwarding ActivityStart to Gemini")
                if gemini and gemini.is_connected:
                    await gemini.send_activity_start()
                else:
                    logger.warning("activity_start received but no active Gemini session")

            # ── barge_in ──────────────────────────────────────────────────
            elif msg_type == "barge_in":
                if gemini and gemini.is_connected:
                    await gemini.send_barge_in()

            # ── end_of_turn ───────────────────────────────────────────────
            # Sent by Flutter when the user taps mic to stop speaking.
            elif msg_type == "end_of_turn":
                logger.info("Client sent end_of_turn — forwarding ActivityEnd to Gemini")
                if gemini and gemini.is_connected:
                    await gemini.send_end_of_turn()
                else:
                    logger.warning("end_of_turn received but no active Gemini session")

            # ── unknown ──────────────────────────────────────────────────
            else:
                logger.warning("Unknown message type: %s", msg_type)

    except WebSocketDisconnect:
        logger.info("Client WebSocket disconnected")

    except Exception:
        logger.exception("Unexpected error in WebSocket handler")

    finally:
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
