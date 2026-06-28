# Technical Design Document: Localized GraphRAG Multi-Agent Scratchpad Engine
**Version:** 2.0.0 (Final Architecture Blueprint)
**Target Environment:** Localhost (CPU-bound Middleware + VRAM-bound SLM)
**Core Paradigms:** Database-Backed Markdown Views, Token Knapsack Optimization, Asynchronous Agent Operations, Deterministic Hallucination Guardrails.
## 1. High-Level System Topography
The system isolates the unpredictable reasoning of local Small Language Models (SLMs) from the strict structural requirements of continuous memory. It achieves this by placing a rigid, API-driven middleware between the agents and the database.
```text
+-----------------------------------------------------------------------------------+
|                            MULTI-AGENT SWARM (Clients)                            |
|  [Researcher Agent]          [Auditor Agent]               [Coder Agent]          |
|         |                           |                            |                |
|    (Agent SDK: Sync-Act-Commit Loop over HTTP/REST)              |                |
+---------|---------------------------|----------------------------|----------------+
          |                           |                            |
          v                           v                            v
+-----------------------------------------------------------------------------------+
|                        FASTAPI MIDDLEWARE (CPU-Bound)                             |
|                                                                                   |
|  1. ROUTING LAYER: Maps requests to isolated session and agent IDs.               |
|                                                                                   |
|  2. CONTEXT ENGINE: Dynamically calculates model token budgets.                   |
|                                                                                   |
|  3. VIEW GENERATOR: Translates backend graphs into Markdown Scratchpads.          |
|                                                                                   |
|  4. VERIFICATION GATE: Strips hallucinations via exact-substring checks.          |
|                                                                                   |
|  5. TELEMETRY MANAGER: Broadcasts real-time events via WebSockets.                |
+-----------------------------|------------------------|----------------------------+
                              |                        |
             (Read/Write)     |                        |   (Real-Time Stream)
                              v                        v
+---------------------------------------+  +----------------------------------------+
|      SQLITE (WAL MODE) MEMORY         |  |       OBSERVABILITY DASHBOARD          |
|  - Global Session Tracker             |  |  - Tracks Token Compression Efficiency |
|  - GraphRAG Triplet Store             |  |  - Logs Hallucination Rejections       |
|  - Unresolved Variables Matrix        |  |  - Monitors Agent Execution States     |
+---------------------------------------+  +----------------------------------------+

```
## 2. The Data Schema & Memory Model
The architecture abandons flat file persistence. Instead, it relies on atomic database structures that map directly to Knowledge Graph primitives.
 * **Global Sessions:** Tracks the high-level project status and the Master Plan (a list of sequential steps guiding the swarm).
 * **Knowledge Graph (Triplets):** The core memory engine. Every fact is stored as a specific relationship: [SOURCE_ENTITY] -> (relationship) -> [TARGET_ENTITY]. Every edge must include the exact source citation.
 * **Unresolved Variables:** A state-tracking matrix that acts as a "to-do list" for missing context. Agents mark variables as "MISSING" or "RESOLVED."
## 3. The End-to-End Execution Pipeline (Sync-Act-Commit)
When an individual agent takes a turn, it must follow a strict three-phase lifecycle dictated by the architecture.
### Phase 1: SYNC (Context Retrieval)
 1. The Agent SDK requests the current state of the world from the Middleware.
 2. The View Generator queries the SQLite Knowledge Graph for the specific session.
 3. The Middleware compiles the raw database triplets into a human/LLM-readable GitHub-Flavored Markdown string (The "Scratchpad View").
 4. The Markdown View is returned to the Agent.
### Phase 2: ACT (Local Inference)
 1. The Agent appends the Markdown View to its system prompt.
 2. The Agent processes raw data (e.g., executing a web search, reading a codebase file).
 3. The Agent calls the local SLM, instructing it to extract new Graph Triplets and identify newly resolved/unresolved variables based strictly on the raw data.
 4. The SLM outputs the extraction as a strictly formatted JSON payload (enforced by Pydantic schemas).
### Phase 3: COMMIT (Verification & Storage)
 1. The Agent SDK sends the newly extracted JSON and the raw text it analyzed back to the Middleware.
 2. The Verification Gate intercepts the payload.
 3. For every triplet submitted, the engine checks if the citation_quote exists exactly within the raw text.
 4. If the citation is verified, the triplet is transactionally written to SQLite.
 5. If the citation is hallucinated, the triplet is dropped. If all triplets are dropped, the update is rejected, forcing the agent into a retry loop.
## 4. The Context & Token Optimization Engine
To prevent "Context Window Exceeded" errors and manage massive payloads (e.g., 50,000+ tokens of raw logs), the system utilizes a dynamic pacing engine.
 1. **Tokenizer Registry:** The middleware loads exact HuggingFace tokenizers matching the downstream local SLM to eliminate guessing.
 2. **Budget Math:** It dynamically calculates: T_{budget} = T_{window} - \left( T_{system} + T_{graph\_memory} + T_{safety\_buffer} \right).
 3. **Adaptive Chunking:** The engine slices massive raw inputs into exact T_{budget} sized blocks with a configurable overlap percentage to preserve logical continuity between boundaries.
 4. **Pagination Enforcement:** The background loop force-feeds these chunks to the agents one by one, ensuring the SLM only reasons over safe, predictable byte sizes.
## 5. The Verification & Self-Correction Loop (Actor-Critic-Refine)
The pipeline is inherently self-healing. When the SLM makes a reasoning or formatting error, the architecture catches it before it corrupts the persistent memory.
 1. **The Trigger:** An agent submits a payload where the extracted citation is slightly modified (hallucinated) by the SLM.
 2. **The Intercept:** The Verification Gate denies the database transaction.
 3. **The Critique:** The middleware replies to the agent with an HTTP 422 error containing a precise critique (e.g., "Fact X lacks an exact source quote in the raw text.").
 4. **The Refine:** The Agent SDK automatically appends this critique to the SLM's active context and triggers a retry, allowing the SLM to fix its own mistake without developer intervention.
## 6. Real-Time Telemetry & Observability
Because the memory extraction happens asynchronously and invisibly in the background, the architecture requires a dedicated telemetry stream for observability.
 * **Protocol:** Local WebSockets attached to the specific session ID.
 * **Compression Metrics:** Upon every successful commit, the engine broadcasts the ratio of raw tokens ingested versus the token weight of the resulting Knowledge Graph Triplets (measuring information density).
 * **Retry Alerts:** If the Actor-Critic circuit is triggered, an alert is broadcast to the stream detailing the exact hallucination caught by the system.
 * **Agent Status:** Tracks which agent is currently locking the database and performing operations.
## 7. Failure Handling & Resilience Matrix
The architecture is designed to survive the chaos of multi-agent concurrency and unreliable SLM outputs.
 * **Concurrency Race Conditions:** If two agents attempt to write to the Knowledge Graph simultaneously, SQLite WAL mode ensures one transaction succeeds while the other is briefly queued, guaranteeing atomic writes with zero data corruption.
 * **Schema Hallucinations:** If the SLM fails to output valid JSON (breaking the Pydantic schema), the Agent SDK catches the parse error locally and triggers a self-correction prompt before ever hitting the Middleware API.
 * **Mid-Execution Crashes:** Because the memory is stored in SQLite and not in RAM, if the host machine restarts mid-process, the swarm can resume exactly where it left off by querying the Unresolved Variables matrix and the most recent Graph Triplets.

## 8. View Generation Example (SLM Prompt Injection)
This is the "View" generated by the FastAPI backend querying the SQLite database, appended to the raw data chunk, and passed to the model during the **ACT** phase.

```markdown
# SYSTEM ROLE & INSTRUCTIONS
You are an autonomous analytical agent operating within a multi-agent swarm. Your core function is to read the `ACTIVE CONTEXT WINDOW`, extract factual network relationships (Knowledge Graph Triplets), and update the `UNRESOLVED VARIABLES MATRIX`.

**CRITICAL DIRECTIVES:**
1. You must base all extractions STRICTLY on the text provided in the `ACTIVE CONTEXT WINDOW`.
2. Every extracted relationship must include an exact, verbatim `citation_quote` from the raw text. If you alter even one word of the quote, the system verification gate will reject your memory update.
3. Your final output must be exclusively a valid JSON object matching the requested schema. Do not output conversational text, markdown formatting (outside of the JSON block), or internal thoughts.

---

# GLOBAL SESSION STATE
**Session ID:** project_x_v2
**Active Agent:** Security_Auditor_Agent
**Global Status:** EXECUTING

## 1. THE MASTER PLAN
- [COMPLETED] Step 1: Map the overall repository directory structure.
- [IN_PROGRESS] Step 2: Identify all authentication routing files and token validation logic.
- [PENDING] Step 3: Audit token lifecycle for missing expiration handlers.
- [PENDING] Step 4: Generate final security report.

## 2. KNOWLEDGE GRAPH MEMORY (Verified Facts)
*The following facts have been verified by the swarm. Use them as context to understand the system, but do not re-extract them.*

* `[SRC/MAIN.PY]` --(initializes)--> `[FASTAPI_APP]` 
  └── Source Citation: "app = FastAPI(title='CoreService')"
* `[FASTAPI_APP]` --(mounts_router)--> `[AUTH_ROUTER]` 
  └── Source Citation: "app.include_router(auth.router, prefix='/v1/auth')"
* `[AUTH_ROUTER]` --(validates_via)--> `[JWT_MIDDLEWARE]` 
  └── Source Citation: "Depends(JWTBearer())"

## 3. UNRESOLVED VARIABLES MATRIX
*The swarm is actively looking for the following information. If you find the answers in the Active Context Window, update their status to RESOLVED.*

- [?] `JWT_SECRET_KEY_SOURCE` (Status: MISSING)
- [?] `TOKEN_EXPIRATION_TIME` (Status: MISSING)
- [?] `DATABASE_CONNECTION_POOL` (Status: RESOLVED)

---

# ACTIVE CONTEXT WINDOW
**[SYSTEM ALERT: ACTIVE PROCESSING - PAGE 3 OF 12]**
**Immediate Task:** Scan the following raw file chunk for authentication logic, extract new triplets, and resolve missing variables.

**[BEGIN RAW TEXT CHUNK]**
```python
# File: src/auth/jwt_handler.py
import time
import jwt
from typing import Dict
from core.config import settings

JWT_ALGORITHM = "HS256"
# Note: Token lifespan is hardcoded here pending migration to config DB
TOKEN_LIFESPAN = 3600  # 1 hour expiration

def sign_jwt(user_id: str) -> Dict[str, str]:
    """Generates the JWT token for an authenticated user."""
    payload = {
        "user_id": user_id,
        "expires": time.time() + TOKEN_LIFESPAN
    }
    # Security: Secret key is pulled directly from environment variables via pydantic settings
    token = jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    
    return {"access_token": token}

```
**[END RAW TEXT CHUNK]**
# OUTPUT FORMAT INSTRUCTIONS
Generate your response as a JSON object strictly matching this Pydantic schema:
{
"extracted_triplets": [
{
"source_entity": "string (UPPERCASE)",
"relationship": "string (lowercase_with_underscores)",
"target_entity": "string (UPPERCASE)",
"citation_quote": "string (Exact substring from raw text)"
}
],
"unresolved_variables_mutations": {
"VARIABLE_NAME": "RESOLVED"
},
"is_chunk_completely_exhausted": true
}
```

