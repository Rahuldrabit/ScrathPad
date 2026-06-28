from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class GraphTriplet(BaseModel):
    source_entity: str = Field(..., description="Subject noun, class, or env var in uppercase.")
    relationship: str = Field(..., description="Active verb indicating connection (lowercase_with_underscores).")
    target_entity: str = Field(..., description="Object noun, config, or destination in uppercase.")
    citation_quote: str = Field(..., description="Exact substring from raw code validating this connection.")

class PageExtractionPayload(BaseModel):
    extracted_triplets: List[GraphTriplet]
    unresolved_variables_mutations: Dict[str, str]
    is_chunk_completely_exhausted: bool

class AgentUpdateRequest(BaseModel):
    agent_id: str
    session_id: str
    raw_active_chunk: str
    extracted_triplets: List[GraphTriplet]
    unresolved_variables_mutations: Dict[str, str]
    is_chunk_completely_exhausted: bool

class MemoryViewResponse(BaseModel):
    session_id: str
    markdown_view: str