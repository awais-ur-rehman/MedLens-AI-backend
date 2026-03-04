"""Triage Director — root agent that orchestrates the Dr. Muhammad system."""

from google.adk.agents import LlmAgent
from google.adk.tools import google_search

from app.agents.visual_assessor import create_visual_assessor
from app.agents.protocol_advisor import create_protocol_advisor
from app.config import settings
from app.prompts import DR_MUHAMMAD_PROMPT
from app.tools.search import search_medical_knowledge

TRIAGE_DIRECTOR_INSTRUCTION = f"""\
{DR_MUHAMMAD_PROMPT}

## AGENT DELEGATION

You have access to specialist sub-agents. Use them appropriately:

### visual_assessor
Call when the user shares a camera image or asks you to look at their injury.
This agent returns a structured JSON assessment with injury type, severity,
body location, and overlay coordinates.

### protocol_advisor
Call AFTER visual assessment (or when the injury type is known) to retrieve
the evidence-based first-aid protocol from the MedLens knowledge base.

### google_search
Use for real-time information such as:
- Drug interactions or contraindications
- Local emergency numbers
- Current medical guidelines
- Information not in the MedLens knowledge base

### search_medical_knowledge
Use to search the MedLens first-aid document corpus directly.

## WORKFLOW
1. User describes or shows their injury
2. If image is available → delegate to visual_assessor
3. Based on assessment → delegate to protocol_advisor
4. Synthesise results into natural, empathetic spoken response
5. Always run safety checks on your own output before responding
"""


def create_triage_director() -> LlmAgent:
    """Create the root Triage Director agent (Dr. Muhammad)."""
    visual_assessor = create_visual_assessor()
    protocol_advisor = create_protocol_advisor()

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
            google_search,
            search_medical_knowledge,
        ],
        sub_agents=[
            visual_assessor,
            protocol_advisor,
        ],
    )
