# MedLens AI — Backend

> **See it. Speak it. Save it.**
> FastAPI backend powering real-time voice-and-vision first aid guidance via Gemini Live.

## Architecture

```
Flutter App
    │
    │  WebSocket (wss://)
    │  ├── Binary frames  → 16 kHz PCM mic audio (Flutter → Gemini)
    │  ├── JSON frames    → control messages, image frames, barge-in
    │  ├── Binary frames  ← 24 kHz PCM speech (Gemini → Flutter speaker)
    │  └── JSON frames    ← transcript, overlays, citations, care summary
    ▼
FastAPI  /ws/session  (websocket_handler.py)
    │
    │  Gemini Live API  (gemini_live_client.py)
    │  Model: gemini-live-2.5-flash-native-audio
    │  Voice: Orus  •  Manual VAD  •  send_realtime_input only
    ▼
Gemini 2.5 Flash Live
    ├── Native audio I/O (bidirectional PCM)
    ├── Tool calls: request_camera, request_live_camera
    ├── Google Search grounding (citation extraction)
    └── [[OVERLAY:{...}]] annotations (parsed server-side)

Post-session:
    Gemini Flash  ←  _generate_care_summary()  → structured JSON → Flutter
```

## Project Structure

```
backend/
├── main.py                        # FastAPI entry: GET /health, WS /ws/session
├── requirements.txt
├── Dockerfile
├── .env.example
└── app/
    ├── config.py                  # Pydantic settings (reads .env)
    ├── gemini_live_client.py      # Gemini Live API client
    │                              #   connect / send_realtime_input /
    │                              #   send_image / receive_stream (while loop)
    ├── websocket_handler.py       # Flutter ↔ Gemini bidirectional bridge
    │                              #   • Routes all client messages
    │                              #   • Parses [[OVERLAY:...]] from text
    │                              #   • Generates care summary on end_session
    ├── prompts/
    │   └── __init__.py            # DR_MUHAMMAD_PROMPT — 12-section system prompt
    │                              #   persona, tools, safety, overlay format,
    │                              #   CALM MODE, escalation, session scripts
    ├── agents/                    # ADK multi-agent pipeline (not wired in prod)
    │   ├── triage_director.py
    │   ├── visual_assessor.py
    │   ├── protocol_advisor.py
    │   ├── safety_guardian.py
    │   └── summary_generator.py
    └── tools/
        ├── search_tool.py         # Google Search grounding citation parser
        ├── rag_tool.py            # Vertex AI Search (Discovery Engine)
        └── maps_tool.py           # Emergency services locator
```

## WebSocket Protocol

### Client → Backend (JSON)

| Message | Fields | Description |
|---------|--------|-------------|
| `start_session` | — | Open Gemini Live session, begin greeting |
| `end_session` | — | Trigger care summary generation |
| `text` | `content` | Send a text message to Dr. Muhammad |
| `image_frame` | `data` (base64 JPEG) | Send a captured photo for analysis |
| `barge_in` | — | Interrupt Dr. Muhammad mid-speech |
| `end_of_turn` | — | Signal user finished speaking |
| `activity_start` | — | Explicit VAD start (push-to-talk) |

### Client → Backend (Binary)

Raw 16-bit PCM audio at 16 kHz, mono, little-endian.

### Backend → Client (JSON)

| Message | Key Fields | Description |
|---------|-----------|-------------|
| `session_started` | — | Gemini session open, greeting incoming |
| `transcript` | `text`, `escalation?` | Dr. Muhammad's speech chunk |
| `user_transcript` | `text` | What Gemini heard the user say |
| `turn_complete` | `speaker` | Agent/user turn finished |
| `overlay` | `data` | Visual annotation for camera preview |
| `citation` | `sources[]` | Google Search grounding sources |
| `assessment` | `data` | Injury severity assessment |
| `agent_thinking` | `tool` | Tool call in progress indicator |
| `care_summary` | `data` | Structured post-session summary |
| `session_ended` | — | Session closed |
| `error` | `message` | Error description |
| `request_camera` | `prompt` | Dr. Muhammad wants to see the injury |
| `request_live_camera` | `prompt`, `duration_seconds` | Dr. Muhammad wants a live feed |

### Backend → Client (Binary)

Raw 16-bit PCM audio at 24 kHz, mono, little-endian — Dr. Muhammad's voice.

## Visual Overlay System

Dr. Muhammad outputs overlay annotations embedded in his text responses:

```
[[OVERLAY:{"type":"highlight","region":"right hand","instruction":"Press here firmly","severity":"high"}]]
```

The backend:
1. Extracts all `[[OVERLAY:{...}]]` matches via regex
2. Strips them from the transcript text (users never see raw JSON)
3. Sends each as `{"type": "overlay", "data": {...}}` to Flutter
4. Flutter renders a coloured bounding box + label on the live camera preview

## Care Summary

When the user ends a session, `_generate_care_summary()` calls Gemini Flash with the full transcript and produces structured JSON matching `CareSummaryModel`:

```json
{
  "session_id": "uuid",
  "timestamp": "ISO-8601",
  "injury_type": "First-degree burn",
  "severity": "low",
  "patient_description": "...",
  "actions_taken": ["Run cool water for 10 min", "..."],
  "medications_discussed": ["Ibuprofen"],
  "follow_up_recommendations": ["See a doctor if blistering occurs"],
  "warning_signs": ["Increased redness", "Fever"],
  "disclaimer": "..."
}
```

## Setup

### Prerequisites

- Python 3.11+
- Google Cloud project with Vertex AI API enabled
- `gcloud auth application-default login` (local dev)

### Install

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
```

| Variable | Description | Example |
|----------|-------------|---------|
| `GOOGLE_CLOUD_PROJECT` | GCP project ID | `medlens-489020` |
| `GOOGLE_CLOUD_LOCATION` | Region | `us-central1` |
| `GEMINI_MODEL` | Live API model ID | `gemini-live-2.5-flash-native-audio` |
| `GEMINI_FLASH_MODEL` | Summary model | `gemini-2.5-flash` |
| `VERTEX_SEARCH_DATASTORE` | Discovery Engine datastore | `medlens-first-aid-docs` |
| `GCS_BUCKET` | Cloud Storage bucket | `medlens-sessions` |

### Run locally

```bash
uvicorn main:app --reload --port 8080
```

### Docker

```bash
docker build -t medlens-backend .
docker run -p 8080:8080 --env-file .env medlens-backend
```

### Cloud Run (deployed)

```
wss://medlens-backend-nw7kauj2aa-uc.a.run.app
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | `{"status": "healthy", "model": "..."}` |
| `/ws/session` | WebSocket | Full session bridge (audio + video + text) |

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` + `uvicorn` | Web framework + ASGI server |
| `google-genai` | Gemini Live API client (send_realtime_input) |
| `google-cloud-aiplatform` | Vertex AI integration |
| `pydantic-settings` | Typed environment config |
| `python-multipart` | WebSocket binary frame handling |

## Critical Implementation Notes

- **`send_realtime_input` only** — never mix `send_client_content` in the same session. Breaks multi-turn from the second message onward.
- **`receive_stream()` loop** — wraps `session.receive()` in a `while self._session:` loop. The Gemini SDK exits after each `turn_complete`; the while loop is what enables multi-turn conversations.
- **Audio format** — mic input must be 16 kHz PCM. Playback output is 24 kHz PCM. Mismatches produce garbled audio.
- **Care summary timeout** — `_generate_care_summary()` has a 12-second timeout with graceful fallback to a minimal summary structure.
