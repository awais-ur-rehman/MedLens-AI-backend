"""Triage Director — root agent that orchestrates the Dr. Muhammad system."""

from __future__ import annotations

from google.adk.agents import LlmAgent
from google.adk.tools import AgentTool

from app.config import settings
from app.prompts import DR_MUHAMMAD_PROMPT
from app.tools.rag_tool import search_first_aid_protocols
from app.tools.maps_tool import find_nearest_emergency_services

TRIAGE_DIRECTOR_INSTRUCTION = f"""\
{DR_MUHAMMAD_PROMPT}

## AGENT DELEGATION

You have access to specialist sub-agents and tools. Use them appropriately:

### visual_assessor (Agent)
Call when the user shares a camera image or asks you to look at their injury.
This agent returns a structured JSON assessment with injury type, severity,
body location, and overlay coordinates.

### protocol_advisor (Agent)
Call AFTER visual assessment (or when the injury type is known) to retrieve
the evidence-based first-aid protocol from the MedLens knowledge base.

### search_first_aid_protocols (Tool)
Use to search the MedLens first-aid document corpus directly when you need
protocol information without a full visual assessment first.

### find_nearest_emergency_services (Tool)
Use when the user asks about nearby hospitals or emergency rooms, or when
the injury severity warrants professional medical attention.

### Google Search (Built-in)
Google Search grounding is enabled in the Live API config. The model will
automatically use it for real-time information such as:
- Drug interactions or contraindications
- Local emergency numbers
- Current medical guidelines
- Information not in the MedLens knowledge base

## WORKFLOW
1. User describes or shows their injury
2. If image is available → delegate to visual_assessor
3. Based on assessment → delegate to protocol_advisor
4. Synthesise results into natural, empathetic spoken response
5. Always run safety checks on your own output before responding
"""


def create_triage_director(
    visual_assessor: LlmAgent,
    protocol_advisor: LlmAgent,
) -> LlmAgent:
    """Create the root Triage Director agent (Dr. Muhammad).

    Args:
        visual_assessor: Pre-created Visual Assessor agent instance.
        protocol_advisor: Pre-created Protocol Advisor agent instance.
    """
    return LlmAgent(
        name="triage_director",
        model=settings.gemini_flash_model,
        description=(
            "Root agent embodying Dr. Muhammad. Routes to visual assessment, "
            "protocol lookup, and Google Search as needed. Synthesises results "
            "into empathetic, safety-checked first-aid guidance."
        ),
        instruction=TRIAGE_DIRECTOR_INSTRUCTION,
        tools=[
            AgentTool(agent=visual_assessor),
            AgentTool(agent=protocol_advisor),
            search_first_aid_protocols,
            find_nearest_emergency_services,
        ],
    )
