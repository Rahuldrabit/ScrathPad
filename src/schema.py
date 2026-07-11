from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class SessionInitRequest(BaseModel):
    session_id: str
    master_plan: Optional[str] = None
    user_query: Optional[str] = None

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

class GraphTripletSchema(BaseModel):
    source_entity: str = Field(..., description="The canonical uppercase macro-entity.")
    relationship: str = Field(..., description="The connecting verb or system dependency.")
    target_entity: str = Field(..., description="The target uppercase macro-entity.")
    citation_quote: str = Field(default="Generated via structural compression.", description="System generated reference.")

class L2CompressionPayload(BaseModel):
    reasoning_justification: str = Field(..., description="Chain-of-thought explaining the community synthesis.")
    source_edge_ids_used: List[str] = Field(..., description="L1 edge_ids collapsed into this summary.")
    extracted_l2_triplets: List[GraphTripletSchema] = Field(..., description="The macro triplets.")

class L1ExtractionPayload(BaseModel):
    triplets: List[GraphTriplet]