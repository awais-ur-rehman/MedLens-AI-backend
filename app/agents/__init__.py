"""AgentPipeline — wires all ADK agents together and exposes high-level actions.

This is the single entry-point used by the WebSocket handler and any other
backend code that needs to run the Dr. Muhammad multi-agent system.
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Any

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.runners import InMemoryRunner
from google.genai import types

from app.agents.triage_director import create_triage_director
from app.agents.safety_guardian import create_safety_guardian
from app.agents.summary_generator import create_summary_generator
from app.config import settings

logger = logging.getLogger(__name__)


class AgentPipeline:
    """Orchestrator for the MedLens multi-agent system.

    Creates and manages the lifecycle of:
    - Triage Director (root) with Visual Assessor + Protocol Advisor sub-agents
    - Safety Guardian (parallel filter)
    - Summary Generator (sequential pipeline)

    Also maintains ``session_context`` — a running log of all assessments,
    protocols, medications, and transcript entries gathered during a session.
    """

    def __init__(self) -> None:
        # ---- Create agents ----
        self.triage_director: LlmAgent = create_triage_director()
        self.safety_guardian: LlmAgent = create_safety_guardian()
        self.summary_generator: SequentialAgent = create_summary_generator()

        # ---- Runners (in-memory for now) ----
        self.triage_runner = InMemoryRunner(
            agent=self.triage_director,
            app_name="medlens_triage",
        )
        self.safety_runner = InMemoryRunner(
            agent=self.safety_guardian,
            app_name="medlens_safety",
        )
        self.summary_runner = InMemoryRunner(
            agent=self.summary_generator,
            app_name="medlens_summary",
        )

        # ---- Persistent session context ----
        self.session_context: dict[str, Any] = {
            "assessments": [],
            "protocols": [],
            "medications": [],
            "transcript": [],
            "escalations": [],
            "citations": [],
        }

    # ------------------------------------------------------------------
    #  High-level actions
    # ------------------------------------------------------------------

    async def assess_visual(self, frame_bytes: bytes) -> dict[str, Any]:
        """Run the Visual Assessor → Protocol Advisor chain on a camera frame.

        Returns the combined result dict with assessment and protocol.
        """
        b64_image = base64.b64encode(frame_bytes).decode("utf-8")

        # Build user content with the image
        user_content = types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(
                    data=frame_bytes,
                    mime_type="image/jpeg",
                ),
                types.Part.from_text(
                    "Please assess this injury image and provide the "
                    "appropriate first-aid protocol."
                ),
            ],
        )

        result: dict[str, Any] = {}

        try:
            session = await self.triage_runner.session_service.create_session(
                app_name="medlens_triage",
                user_id="session_user",
            )

            async for event in self.triage_runner.run_async(
                session_id=session.id,
                user_id="session_user",
                new_message=user_content,
            ):
                if event.is_final_response():
                    for part in event.content.parts:
                        if part.text:
                            try:
                                result = json.loads(part.text)
                            except json.JSONDecodeError:
                                result = {"raw_text": part.text}

            # Track in session context
            if result:
                self.session_context["assessments"].append(result)

        except Exception as e:
            logger.error("Visual assessment failed: %s", e)
            result = {"error": str(e)}

        return result

    async def safety_check(self, text: str) -> dict[str, Any]:
        """Run the Safety Guardian on an output text.

        Returns a dict with safe/modified_text/escalation_needed fields.
        """
        user_content = types.Content(
            role="user",
            parts=[types.Part.from_text(
                f"Please review this text for safety:\n\n{text}"
            )],
        )

        result: dict[str, Any] = {"safe": True, "modified_text": text}

        try:
            session = await self.safety_runner.session_service.create_session(
                app_name="medlens_safety",
                user_id="session_user",
            )

            async for event in self.safety_runner.run_async(
                session_id=session.id,
                user_id="session_user",
                new_message=user_content,
            ):
                if event.is_final_response():
                    for part in event.content.parts:
                        if part.text:
                            try:
                                result = json.loads(part.text)
                            except json.JSONDecodeError:
                                result = {"safe": True, "modified_text": text}

        except Exception as e:
            logger.error("Safety check failed: %s", e)
            # Fail-open: allow the text through but log the error
            result = {"safe": True, "modified_text": text, "error": str(e)}

        return result

    async def generate_summary(self) -> dict[str, Any]:
        """Run the Summary Generator with the full session context.

        Returns a structured CareSummary dict.
        """
        context_text = json.dumps(self.session_context, indent=2, default=str)

        user_content = types.Content(
            role="user",
            parts=[types.Part.from_text(
                f"Generate a care summary for this session:\n\n{context_text}"
            )],
        )

        result: dict[str, Any] = {}

        try:
            session = await self.summary_runner.session_service.create_session(
                app_name="medlens_summary",
                user_id="session_user",
            )

            # Pre-populate session state with context
            session.state.update(self.session_context)

            async for event in self.summary_runner.run_async(
                session_id=session.id,
                user_id="session_user",
                new_message=user_content,
            ):
                if event.is_final_response():
                    for part in event.content.parts:
                        if part.text:
                            try:
                                result = json.loads(part.text)
                            except json.JSONDecodeError:
                                result = {"raw_text": part.text}

        except Exception as e:
            logger.error("Summary generation failed: %s", e)
            result = {"error": str(e)}

        return result

    async def handle_tool_call(
        self, tool_name: str, tool_args: dict[str, Any]
    ) -> dict[str, Any]:
        """Route an external tool call to the correct handler.

        This is called from the WebSocket handler when Gemini Live issues a
        tool_call that needs to be resolved server-side.
        """
        handlers = {
            "search_medical_knowledge": self._handle_search,
            "assess_visual": self._handle_visual,
        }

        handler = handlers.get(tool_name)
        if handler:
            return await handler(tool_args)

        logger.warning("Unknown tool call: %s", tool_name)
        return {"error": f"Unknown tool: {tool_name}"}

    # ------------------------------------------------------------------
    #  Context tracking helpers
    # ------------------------------------------------------------------

    def add_transcript(self, speaker: str, text: str) -> None:
        """Append a message to the session transcript."""
        self.session_context["transcript"].append({
            "speaker": speaker,
            "text": text,
        })

    def add_escalation(self, reason: str) -> None:
        """Record an escalation event."""
        self.session_context["escalations"].append(reason)

    def add_citation(self, citation: dict[str, Any]) -> None:
        """Record a citation."""
        self.session_context["citations"].append(citation)

    # ------------------------------------------------------------------
    #  Private tool handlers
    # ------------------------------------------------------------------

    async def _handle_search(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle a search_medical_knowledge tool call."""
        from app.tools.search import search_medical_knowledge

        query = args.get("query", "")
        return search_medical_knowledge(query)

    async def _handle_visual(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle an assess_visual tool call."""
        image_b64 = args.get("image", "")
        frame_bytes = base64.b64decode(image_b64)
        return await self.assess_visual(frame_bytes)
