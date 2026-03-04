# MedLens AI — Backend

FastAPI backend powering the MedLens real-time first-aid companion. Bridges the Flutter app to Gemini Live API and orchestrates a multi-agent system via Google ADK.

## Architecture

```
Flutter App ←—WebSocket—→ FastAPI ←—Live API—→ Gemini 2.5 Flash
                              │
                       ADK Agent Pipeline
                              │
                 ┌─────────────┼─────────────┐
                 │             │             │
          Triage Director   Safety     Summary
          (Dr. Muhammad)   Guardian   Generator
                 │
          ┌──────┴──────┐
          │             │
    Visual Assessor  Protocol Advisor
                         │
                    Vertex AI Search
                    (RAG — first-aid docs)
```

## Project Structure

```
backend/
├── main.py                       # FastAPI entry point (health + WebSocket)
├── requirements.txt
├── Dockerfile
├── .env.example
├── app/
│   ├── config.py                 # Pydantic settings (env vars)
│   ├── gemini_live_client.py     # Gemini Live API WebSocket client
│   ├── websocket_handler.py      # Flutter ↔ Gemini bidirectional bridge
│   ├── prompts/
│   │   ├── __init__.py           # DR_MUHAMMAD_PROMPT (11 sections)
│   │   └── system_instructions.py
│   ├── agents/
│   │   ├── __init__.py           # AgentPipeline orchestrator
│   │   ├── triage_director.py    # Root agent (Dr. Muhammad persona)
│   │   ├── visual_assessor.py    # Camera frame → injury JSON
│   │   ├── protocol_advisor.py   # RAG → step-by-step protocol
│   │   ├── safety_guardian.py    # Output safety filter
│   │   └── summary_generator.py  # SequentialAgent → care summary
│   └── tools/
│       ├── rag_tool.py           # Vertex AI Search (Discovery Engine)
│       ├── search_tool.py        # Google Search grounding parser
│       ├── maps_tool.py          # Emergency services locator
│       ├── search.py             # Legacy search wrapper
│       ├── image_analysis.py     # Placeholder
│       └── storage.py            # Placeholder
├── grounding_data/               # First-aid protocol documents for RAG
└── scripts/
    └── test_live_api.py          # Smoke test for Gemini Live API
```

## Setup

### Prerequisites

- Python 3.11+
- Google Cloud project with these APIs enabled:
  - Vertex AI API
  - Discovery Engine API
  - Cloud Storage API
- Authenticated via `gcloud auth application-default login`

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
# Edit .env with your values:
```

| Variable | Description | Example |
|---|---|---|
| `GOOGLE_CLOUD_PROJECT` | GCP project ID | `medlens-489020` |
| `GOOGLE_CLOUD_LOCATION` | Region | `us-central1` |
| `GEMINI_MODEL` | Live API model | `gemini-live-2.5-flash-native-audio` |
| `GEMINI_FLASH_MODEL` | ADK agent model | `gemini-2.5-flash` |
| `VERTEX_SEARCH_DATASTORE` | Discovery Engine datastore | `medlens-first-aid-docs` |
| `VERTEX_SEARCH_APP` | Discovery Engine search app | `medlens-search-app` |
| `GCS_BUCKET` | Cloud Storage bucket | `medlens-sessions` |

### Run

```bash
uvicorn main:app --reload --port 8080
```

### Docker

```bash
docker build -t medlens-backend .
docker run -p 8080:8080 --env-file .env medlens-backend
```

## API Endpoints

| Endpoint | Type | Description |
|---|---|---|
| `GET /health` | HTTP | Health check → `{"status": "healthy"}` |
| `WS /ws/session` | WebSocket | Real-time session (audio, video, text) |

### WebSocket Protocol

**Client → Backend (JSON):**
```json
{"type": "start_session"}
{"type": "end_session"}
{"type": "text", "content": "I burned my hand"}
{"type": "image_frame", "data": "<base64 JPEG>"}
{"type": "barge_in"}
```

**Client → Backend (Binary):** Raw PCM audio (16-bit LE, 16 kHz, mono)

**Backend → Client (JSON):**
```json
{"type": "session_started"}
{"type": "transcript", "text": "...", "escalation": true}
{"type": "citation", "sources": [{"source": "...", "url": "..."}]}
{"type": "agent_thinking", "tool": "search_first_aid_protocols"}
{"type": "care_summary", "data": {...}}
{"type": "session_ended"}
{"type": "error", "message": "..."}
```

**Backend → Client (Binary):** PCM audio from Dr. Muhammad (16-bit LE, 24 kHz, mono)

## Agent Pipeline

| Agent | Type | Role |
|---|---|---|
| **Triage Director** | `LlmAgent` | Root agent — Dr. Muhammad persona, routes to specialists |
| **Visual Assessor** | `LlmAgent` | Analyses camera frames → structured injury JSON |
| **Protocol Advisor** | `LlmAgent` | RAG search → step-by-step first-aid protocol with citations |
| **Safety Guardian** | `LlmAgent` | Checks outputs for dangerous advice, injects disclaimers |
| **Summary Generator** | `SequentialAgent` | Session data collector → care summary writer pipeline |

## Key Dependencies

| Package | Purpose |
|---|---|
| `fastapi` + `uvicorn` | Web framework + ASGI server |
| `google-genai` | Gemini Live API client |
| `google-adk` | Agent Development Kit (LlmAgent, SequentialAgent, AgentTool) |
| `google-cloud-discoveryengine` | Vertex AI Search for RAG |
| `pydantic-settings` | Typed config from environment |
