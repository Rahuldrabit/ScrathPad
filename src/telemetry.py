import json
import asyncio
from fastapi import WebSocket, WebSocketDisconnect
from typing import List, Dict

class LocalTelemetryManager:
    def __init__(self):
        # Maps session_id -> List of active WebSocket connections
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        if session_id not in self.active_connections:
            self.active_connections[session_id] = []
        self.active_connections[session_id].append(websocket)
        print(f"[TELEMETRY] Listener attached to session: {session_id}")

    def disconnect(self, session_id: str, websocket: WebSocket):
        if session_id in self.active_connections:
            if websocket in self.active_connections[session_id]:
                self.active_connections[session_id].remove(websocket)
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]
        print(f"[TELEMETRY] Listener detached from session: {session_id}")

    async def broadcast(self, session_id: str, message: dict):
        """
        Sends real-time JSON payloads to all clients watching the session.
        """
        if session_id in self.active_connections:
            payload = json.dumps(message)
            # Send concurrently to all open connections
            await asyncio.gather(
                *[conn.send_text(payload) for conn in self.active_connections[session_id]],
                return_exceptions=True
            )

# Initialize the global manager instance
telemetry_manager = LocalTelemetryManager()
