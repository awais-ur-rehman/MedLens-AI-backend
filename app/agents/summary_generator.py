"""Summary Generator — produces a structured care summary at session end."""

from google.adk.agents import LlmAgent, SequentialAgent

from app.config import settings

# ---------------------------------------------------------------------------
# Step 1: Collect and structure raw session data
# ---------------------------------------------------------------------------

COLLECTOR_INSTRUCTION = """\
You are the Session Data Collector.

## YOUR TASK
Read the full session context from state and compile the raw data into a
structured intermediate format.

The session state contains:
- `transcript`: list of conversation messages
- `visual_assessment`: the most recent injury assessment JSON
- `protocol_advice`: the protocol advice JSON
- `safety_result`: the safety check results

## OUTPUT FORMAT (strict JSON)
```json
{
  "injury_type": "from visual assessment",
  "severity": "from visual assessment",
  "patient_description": "Brief summary of what happened based on transcript",
  "actions_discussed": ["List of actions mentioned in the transcript"],
  "medications_mentioned": ["Any medications discussed"],
  "protocols_referenced": ["First aid protocols used"],
  "escalation_was_triggered": true/false,
  "session_duration_context": "Brief note on how the session went"
}
```

Output MUST be valid JSON only.
"""

# ---------------------------------------------------------------------------
# Step 2: Generate the final care summary
# ---------------------------------------------------------------------------

SUMMARY_WRITER_INSTRUCTION = """\
You are the Care Summary Writer.

## YOUR TASK
Using the collected session data (from state key `collected_data`),
generate the final patient-facing care summary.

## OUTPUT FORMAT (strict JSON)
```json
{
  "session_id": "placeholder — will be filled by the pipeline",
  "timestamp": "ISO 8601 — use current time",
  "injury_type": "string",
  "severity": "low | medium | high",
  "patient_description": "What happened in the patient's own words",
  "actions_taken": [
    "Step-by-step list of first aid actions taken or recommended"
  ],
  "medications_discussed": [
    "Any medications mentioned (OTC or otherwise)"
  ],
  "follow_up_recommendations": [
    "When to see a doctor",
    "Warning signs to watch for",
    "Follow-up care steps"
  ],
  "warning_signs": [
    "Specific symptoms that mean the patient should seek immediate care"
  ],
  "disclaimer": "This was first aid guidance only. Dr. Muhammad is not a substitute for professional medical care. If symptoms worsen or you are unsure, please contact a healthcare provider or call emergency services."
}
```

## RULES
- Be empathetic and clear — this is for the patient to keep.
- Include ALL actions discussed, even if some were not performed.
- Warning signs should be specific to the injury type.
- Always include the disclaimer.
- Output MUST be valid JSON only.
"""


def create_summary_generator() -> SequentialAgent:
    """Create the Summary Generator as a SequentialAgent pipeline."""
    collector = LlmAgent(
        name="session_data_collector",
        model=settings.gemini_flash_model,
        description="Collects and structures raw session data for summarisation.",
        instruction=COLLECTOR_INSTRUCTION,
        output_key="collected_data",
    )

    writer = LlmAgent(
        name="care_summary_writer",
        model=settings.gemini_flash_model,
        description="Generates the final structured care summary JSON.",
        instruction=SUMMARY_WRITER_INSTRUCTION,
        output_key="care_summary",
    )

    return SequentialAgent(
        name="summary_generator",
        description=(
            "Two-step pipeline that collects session data then generates a "
            "structured care summary for the patient."
        ),
        sub_agents=[collector, writer],
    )
