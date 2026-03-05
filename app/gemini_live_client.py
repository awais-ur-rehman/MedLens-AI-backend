"""Persistent WebSocket connection to the Gemini Live API.

Uses the google-genai SDK to maintain a bidirectional streaming session
with Gemini for real-time audio/text conversation as Dr. Muhammad.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncGenerator

from google import genai
from google.genai import types

from app.config import Settings
from app.prompts import DR_MUHAMMAD_PROMPT

logger = logging.getLogger(__name__)


class GeminiLiveClient:
    """Manages a persistent Live API session with Gemini."""

    def __init__(self, config: Settings) -> None:
        self._config = config
        self._client = genai.Client(
            vertexai=True,
            project=config.google_cloud_project,
            location=config.google_cloud_location,
        )
        self._session: Any | None = None
        self._session_ctx: Any | None = None
        logger.info(
            "GeminiLiveClient initialised  project=%s  model=%s",
            config.google_cloud_project,
            config.gemini_model,
        )

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open a Live API session with the Dr. Muhammad persona."""
        request_camera_decl = types.FunctionDeclaration(
            name="request_camera",
            description="Allows Dr. Muhammad to ask the user to open the camera to take a photo of their injury.",
            parameters={
                "type": "OBJECT",
                "properties": {
                    "prompt": {"type": "STRING", "description": "The prompt or reason to show the user (e.g. 'Show me the affected area')."}
                },
                "required": ["prompt"],
            }
        )
        request_live_camera_decl = types.FunctionDeclaration(
            name="request_live_camera",
            description="Allows Dr. Muhammad to ask the user to hold the camera steady for a live video feed.",
            parameters={
                "type": "OBJECT",
                "properties": {
                    "prompt": {"type": "STRING", "description": "The prompt to show the user."},
                    "duration_seconds": {"type": "INTEGER", "description": "The duration of the live feed in seconds (recommended: 10)."}
                },
                "required": ["prompt", "duration_seconds"],
            }
        )

        live_config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Orus",  # Calm, authoritative voice
                    )
                )
            ),
            # Transcribe both directions so we get text alongside audio
            output_audio_transcription=types.AudioTranscriptionConfig(),
            input_audio_transcription=types.AudioTranscriptionConfig(),
            system_instruction=DR_MUHAMMAD_PROMPT,
            tools=[types.Tool(
                google_search=types.GoogleSearch(),
                function_declarations=[request_camera_decl, request_live_camera_decl],
            )],
            # Disable automatic VAD so we control turn boundaries explicitly.
            # The client sends ActivityStart when the user begins speaking and
            # ActivityEnd when they finish. This prevents the session from dying
            # due to auto-VAD timing issues after the greeting turn completes.
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(
                    disabled=True,
                ),
            ),
        )

        self._session_ctx = self._client.aio.live.connect(
            model=self._config.gemini_model,
            config=live_config,
        )
        self._session = await self._session_ctx.__aenter__()
        logger.info("Live session connected")

    async def disconnect(self) -> None:
        """Gracefully close the Live API session."""
        if self._session_ctx is not None:
            try:
                await self._session_ctx.__aexit__(None, None, None)
            except Exception:
                logger.warning("Error closing Live session", exc_info=True)
            finally:
                self._session = None
                self._session_ctx = None
                logger.info("Live session disconnected")

    @property
    def is_connected(self) -> bool:
        return self._session is not None

    # ------------------------------------------------------------------
    # Sending input
    # ------------------------------------------------------------------

    async def send_audio(self, pcm_bytes: bytes) -> None:
        """Stream raw PCM audio (16 kHz, 16-bit mono) to the model."""
        if not self._session:
            raise RuntimeError("Session not connected")
        await self._session.send_realtime_input(
            audio=types.Blob(data=pcm_bytes, mime_type="audio/pcm;rate=16000"),
        )

    async def send_image(self, jpeg_bytes: bytes) -> None:
        """Send a JPEG image frame for visual analysis."""
        if not self._session:
            raise RuntimeError("Session not connected")
        await self._session.send_realtime_input(
            video=types.Blob(data=jpeg_bytes, mime_type="image/jpeg"),
        )

    async def send_text(self, text: str) -> None:
        """Send a text message via the realtime input channel.

        Uses send_realtime_input (same channel as audio) to avoid mixing
        API modes, which would break VAD-based audio turns.
        """
        if not self._session:
            raise RuntimeError("Session not connected")
        await self._session.send_realtime_input(text=text)
        await self._session.send_realtime_input(
            activity_end=types.ActivityEnd(),
        )

    async def send_activity_start(self) -> None:
        """Signal that the user has started speaking (manual VAD mode).

        Must be called before streaming user audio so Gemini knows to start
        processing it. Paired with send_end_of_turn (ActivityEnd) when done.
        """
        if not self._session:
            raise RuntimeError("Session not connected")
        logger.info("Sending ActivityStart — user started speaking")
        await self._session.send_realtime_input(
            activity_start=types.ActivityStart(),
        )

    async def send_barge_in(self) -> None:
        """Interrupt the model's current response (barge-in).

        In manual VAD mode, ActivityEnd immediately signals Gemini to stop
        generating. The user can then tap mic again to start a new turn.
        """
        if not self._session:
            raise RuntimeError("Session not connected")
        await self._session.send_realtime_input(
            activity_end=types.ActivityEnd(),
        )

    async def trigger_greeting(self) -> None:
        """Send an initial trigger so Dr. Muhammad starts with a greeting.

        Gemini Live doesn't auto-speak on connect — it waits for input.
        We use send_realtime_input (same channel as audio) to avoid mixing
        API modes, which breaks subsequent user messages.
        Text input is processed immediately — no ActivityEnd needed.
        """
        if not self._session:
            raise RuntimeError("Session not connected")
        logger.info("Triggering greeting via send_realtime_input(text='.')")
        await self._session.send_realtime_input(text=".")

    async def send_end_of_turn(self) -> None:
        """Signal that the user has finished speaking (manual VAD mode).

        Sends ActivityEnd to tell Gemini the user's turn is complete.
        Gemini then processes the buffered audio and generates a response.
        Only valid in manual VAD mode (automatic_activity_detection.disabled=True).
        """
        if not self._session:
            raise RuntimeError("Session not connected")
        logger.info("Sending ActivityEnd — user finished speaking")
        await self._session.send_realtime_input(
            activity_end=types.ActivityEnd(),
        )

    # ------------------------------------------------------------------
    # Receiving output
    # ------------------------------------------------------------------

    async def receive_stream(self) -> AsyncGenerator[dict[str, Any], None]:
        """Async generator yielding response chunks from the model.

        Loops over multiple turns. The SDK's session.receive() handles ONE turn
        (up to turn_complete) then exits — we restart it so the session stays
        alive for the full multi-turn conversation.

        Yields dicts with one of these shapes:
            {"type": "audio",     "data": bytes}
            {"type": "text",      "text": str}
            {"type": "user_text", "text": str}
            {"type": "turn_complete", "speaker": "agent"}
            {"type": "tool_call", "data": <function call object>}
        """
        if not self._session:
            raise RuntimeError("Session not connected")

        # CRITICAL: session.receive() ends after each turn_complete (by design).
        # We wrap it in a while loop to keep reading across multiple turns.
        while self._session:
            async for message in self._session.receive():
                # --- Audio chunk ---
                raw_audio = getattr(message, "data", None)
                if raw_audio:
                    yield {"type": "audio", "data": raw_audio}
                    # Don't access .text on audio messages — avoids SDK warning
                    # about non-text parts. Fall through to check server_content.
                else:
                    # --- Text chunk (only for non-audio messages) ---
                    if getattr(message, "text", None):
                        yield {"type": "text", "text": message.text}

                # Always check server_content regardless of audio presence.
                server_content = getattr(message, "server_content", None)
                if server_content:
                    # --- Output Audio Transcription ---
                    out_trans = getattr(server_content, "output_transcription", None)
                    if out_trans and out_trans.text:
                        yield {"type": "text", "text": out_trans.text}

                    # --- Input Audio Transcription ---
                    inp_trans = getattr(server_content, "input_transcription", None)
                    if inp_trans and inp_trans.text:
                        yield {"type": "user_text", "text": inp_trans.text}

                    # --- Tool / function call ---
                    model_turn = getattr(server_content, "model_turn", None)
                    if model_turn and model_turn.parts:
                        for part in model_turn.parts:
                            fn_call = getattr(part, "function_call", None)
                            if fn_call:
                                yield {"type": "tool_call", "data": fn_call}

                    # --- Turn Complete ---
                    turn_complete = getattr(server_content, "turn_complete", None)
                    if turn_complete:
                        yield {"type": "turn_complete", "speaker": "agent"}

            logger.debug("Gemini turn ended — restarting receive() for next turn")

    # ------------------------------------------------------------------
    # Tool response
    # ------------------------------------------------------------------

    async def send_tool_response(
        self, function_responses: list[types.FunctionResponse]
    ) -> None:
        """Return a tool's result to the model so it can continue."""
        if not self._session:
            raise RuntimeError("Session not connected")
        await self._session.send_tool_response(
            function_responses=function_responses,
        )
