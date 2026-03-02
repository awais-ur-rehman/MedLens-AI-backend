"""MedLens AI - FastAPI Backend Entry Point."""

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from app.websocket_handler import handle_session

app = FastAPI(
    title="MedLens AI",
    description="AI-powered medical lens analysis service",
    version="0.1.0",
)

# CORS middleware — allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "medlens-ai"}


@app.websocket("/ws/session")
async def websocket_session(websocket: WebSocket):
    """WebSocket endpoint for real-time AI session communication."""
    await handle_session(websocket)

