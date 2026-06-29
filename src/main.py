import json
import asyncio
from fastapi import FastAPI, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from typing import Dict, List

# Core internal system imports
from database import initialize_database, get_db_connection
from schema import AgentUpdateRequest, MemoryViewResponse, PageExtractionPayload, SessionInitRequest
from engine import commit_page_data_to_sqlite, compile_graph_memory_to_markdown

app = FastAPI(title="Scratchpad Context Middleware", version="2.0.0")

class LocalTelemetryManager:
    def __init__(self):
        # Maps session_id -> List of active WebSocket connections
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        if session_id not in self.active_connections:
            self.active_connections[session_id] = []
        self.active_connections[session_id].append(websocket)

    def disconnect(self, session_id: str, websocket: WebSocket):
        if session_id in self.active_connections:
            self.active_connections[session_id].remove(websocket)
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]

    async def broadcast(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            payload = json.dumps(message)
            # Safe concurrent broadcasting to all attached listeners
            await asyncio.gather(
                *[conn.send_text(payload) for conn in self.active_connections[session_id]],
                return_exceptions=True
            )

# Instantiate the global real-time event dispatcher
telemetry_manager = LocalTelemetryManager()

@app.on_event("startup")
async def startup_event():
    initialize_database()
    print("[MIDDLEWARE] Local SQLite GraphRAG tables ready.")

@app.post("/v1/session/init")
async def initialize_session(payload: SessionInitRequest):
    """
    Establishes a new execution tracking window and saves the 
    high-level Master Plan for the multi-agent swarm.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT OR REPLACE INTO sessions (session_id, master_plan, global_status)
            VALUES (?, ?, 'EXECUTING')
        """, (payload.session_id, payload.master_plan))
        conn.commit()
        return {"status": "INITIALIZED", "session_id": payload.session_id}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to initialize session: {str(e)}")
    finally:
        conn.close()

@app.get("/v1/session/{session_id}/memory", response_model=MemoryViewResponse)
async def get_session_memory_view(session_id: str):
    """
    Compiles the relational data matrix out of SQLite directly into
    a GitHub-Flavored Markdown text layer for the agents to read.
    """
    try:
        # DB reads are fast, but running inside threadpool keeps event loop pristine
        markdown_text = await run_in_threadpool(compile_graph_memory_to_markdown, session_id)
        return MemoryViewResponse(session_id=session_id, markdown_view=markdown_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/agent/update")
async def process_agent_memory_update(payload: AgentUpdateRequest):
    """
    Handles incoming extractions, processes them through string-submatch gates,
    and updates tracking metrics across the open telemetry cluster.
    """
    raw_tokens = len(payload.raw_active_chunk) // 4
    
    extraction_payload = PageExtractionPayload(
        extracted_triplets=payload.extracted_triplets,
        unresolved_variables_mutations=payload.unresolved_variables_mutations,
        is_chunk_completely_exhausted=payload.is_chunk_completely_exhausted
    )
    
    # CRITICAL PATCH: Offloads blocking synchronous SQLite I/O operations to worker threads
    saved_triplets_count = await run_in_threadpool(
        commit_page_data_to_sqlite,
        session_id=payload.session_id,
        agent_id=payload.agent_id,
        raw_chunk=payload.raw_active_chunk,
        extraction_data=extraction_payload
    )
    
    rejected_count = len(payload.extracted_triplets) - saved_triplets_count
    estimated_compressed_tokens = saved_triplets_count * 12 
    efficiency_delta = 1.0 - (estimated_compressed_tokens / max(raw_tokens, 1))
    
    # Broadcast alerts if hallucinated data was stripped
    if rejected_count > 0:
        await telemetry_manager.broadcast(payload.session_id, {
            "event": "scratchpad_retry",
            "telemetry": {"active_agent": payload.agent_id, "failure_reason": "CITATION_MISMATCH"},
            "error_details": {"rejected_facts_count": rejected_count}
        })

    # Broadcast computational compression updates
    await telemetry_manager.broadcast(payload.session_id, {
        "event": "scratchpad_processing_chunk",
        "telemetry": {"active_agent": payload.agent_id, "status": "CLEARED" if saved_triplets_count > 0 else "FAILED"},
        "token_metrics": {"compression_efficiency_delta": round(max(efficiency_delta, 0.0), 4)}
    })

    if len(payload.extracted_triplets) > 0 and saved_triplets_count == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Update rejected. All submitted triplets failed verification."
        )
        
    return {
        "status": "SUCCESS",
        "verified_triplets_committed": saved_triplets_count,
        "rejected_triplets_count": rejected_count
    }

@app.websocket("/v1/session/{session_id}/telemetry")
async def websocket_telemetry_stream(websocket: WebSocket, session_id: str):
    """
    Maintains persistent, real-time connectivity between the backend state engine
    and dashboard nodes monitoring client agent actions.
    """
    await telemetry_manager.connect(session_id, websocket)
    try:
        while True:
            # Keeps the socket connection alive without consuming infinite CPU resources
            await websocket.receive_text()
    except WebSocketDisconnect:
        telemetry_manager.disconnect(session_id, websocket)

from pydantic import BaseModel
from client import ScratchpadMiddleware

scratchpad = ScratchpadMiddleware()

class TurnInput(BaseModel):
    session_id: str
    messy_input: str

class DrillDownInput(BaseModel):
    session_id: str
    edge_id: str

@app.post("/v1/middleware/process")
async def process_turn_endpoint(payload: TurnInput):
    """Ingests messy input and returns a clean, token-bounded Markdown graph."""
    try:
        clean_context = await run_in_threadpool(scratchpad.process_turn, payload.session_id, payload.messy_input)
        return {"session_id": payload.session_id, "scratchpad_view": clean_context}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/middleware/drill-down")
async def drill_down_endpoint(payload: DrillDownInput):
    """Exposes granular L1 details for an agent tracking specific macro structures."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT source_entity, relationship, target_entity, citation_quote 
            FROM knowledge_graph 
            WHERE session_id = ? AND parent_node_id = ?
        """, (payload.session_id, payload.edge_id))
        
        details = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return {"parent_node_id": payload.edge_id, "granular_history": details}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))