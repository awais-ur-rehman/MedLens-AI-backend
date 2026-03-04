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
            tools=[types.Tool(google_search=types.GoogleSearch())],
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
        """Send a text message as user content."""
        if not self._session:
            raise RuntimeError("Session not connected")
        await self._session.send_client_content(
            turns=types.Content(
                role="user",
                parts=[types.Part.from_text(text)],
            ),
            turn_complete=True,
        )

    async def send_barge_in(self) -> None:
        """Interrupt the model's current response (barge-in)."""
        if not self._session:
            raise RuntimeError("Session not connected")
        await self._session.send_client_content(
            turns=types.Content(role="user", parts=[]),
            turn_complete=True,
        )

    # ------------------------------------------------------------------
    # Receiving output
    # ------------------------------------------------------------------

    async def receive_stream(self) -> AsyncGenerator[dict[str, Any], None]:
        """Async generator yielding response chunks from the model.

        Yields dicts with one of these shapes:
            {"type": "audio",     "data": bytes}
            {"type": "text",      "text": str}
            {"type": "tool_call", "data": <function call object>}
        """
        if not self._session:
            raise RuntimeError("Session not connected")

        async for message in self._session.receive():
            # --- Audio chunk ---
            if getattr(message, "data", None):
                yield {"type": "audio", "data": message.data}

            # --- Text chunk (if TEXT modality was used) ---
            if getattr(message, "text", None):
                yield {"type": "text", "text": message.text}

            server_content = getattr(message, "server_content", None)
            if server_content:
                # --- Output Audio Transcription (Text representation of voice) ---
                out_trans = getattr(server_content, "output_transcription", None)
                if out_trans and out_trans.text:
                    yield {"type": "text", "text": out_trans.text}

                # --- Input Audio Transcription (Optional: what the user said) ---
                inp_trans = getattr(server_content, "input_transcription", None)
                if inp_trans and inp_trans.text:
                    # We can yield this as user_text if UI wants to show what was heard
                    yield {"type": "user_text", "text": inp_trans.text}

                # --- Tool / function call ---
                model_turn = getattr(server_content, "model_turn", None)
                if model_turn and model_turn.parts:
                    for part in model_turn.parts:
                        fn_call = getattr(part, "function_call", None)
                        if fn_call:
                            yield {"type": "tool_call", "data": fn_call}

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
