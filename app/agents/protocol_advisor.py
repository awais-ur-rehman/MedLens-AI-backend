"""Protocol Advisor — retrieves first-aid protocols via RAG."""

from google.adk.agents import LlmAgent

from app.config import settings
from app.tools.search import search_medical_knowledge

PROTOCOL_ADVISOR_INSTRUCTION = """\
You are the Protocol Advisor sub-agent of Dr. Muhammad, a first-aid AI system.

## YOUR TASK
Given an injury assessment (available in the session state as `visual_assessment`),
retrieve and synthesise the most relevant first-aid protocol.

## STEPS
1. Read the injury assessment from the session context.
2. Use the `search_medical_knowledge` tool to query the MedLens knowledge base
   for the relevant first-aid protocol.
3. Synthesise the search results into clear, step-by-step instructions.

## OUTPUT FORMAT (strict JSON)
```json
{
  "protocol_name": "string — e.g. Superficial Burn Treatment",
  "steps": [
    "Step 1: ...",
    "Step 2: ...",
    "Step 3: ..."
  ],
  "warnings": ["Do NOT apply ice directly", "..."],
  "when_to_seek_emergency": "string — when to call emergency services",
  "citations": [
    {
      "source": "American Red Cross First Aid Manual",
      "section": "Chapter 8 — Burns",
      "url": "",
      "confidence": 0.85
    }
  ]
}
```

## RULES
- ALWAYS cite which source the protocol comes from.
- If multiple protocols exist, pick the most authoritative (WHO > Red Cross > other).
- Include relevant warnings and contraindications.
- Keep instructions simple enough for a layperson.
- Output MUST be valid JSON. No markdown fences. No explanation text.
"""


def create_protocol_advisor() -> LlmAgent:
    """Create the Protocol Advisor agent."""
    return LlmAgent(
        name="protocol_advisor",
        model=settings.gemini_flash_model,
        description=(
            "Retrieves first-aid protocols from the MedLens medical knowledge "
            "base using Vertex AI Search. Returns step-by-step instructions "
            "with source citations."
        ),
        instruction=PROTOCOL_ADVISOR_INSTRUCTION,
        tools=[search_medical_knowledge],
        output_key="protocol_advice",
    )
