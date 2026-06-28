import httpx
from typing import List, Dict

class ScratchpadAgentClient:
    def __init__(self, agent_id: str, session_id: str, base_url: str = "http://localhost:8000"):
        self.agent_id = agent_id
        self.session_id = session_id
        self.base_url = base_url
        self.client = httpx.AsyncClient(base_url=base_url)

    async def get_memory_view(self) -> str:
        response = await self.client.get(f"/v1/session/{self.session_id}/memory")
        response.raise_for_status()
        return response.json().get("markdown_view")

    async def update_memory(self, raw_chunk: str, triplets: List[Dict], variables: Dict[str, str], is_done: bool = False):
        payload = {
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "raw_active_chunk": raw_chunk,
            "extracted_triplets": triplets,
            "unresolved_variables_mutations": variables,
            "is_chunk_completely_exhausted": is_done
        }
        
        response = await self.client.post("/v1/agent/update", json=payload)
        response.raise_for_status()
        return response.json()

    async def close(self):
        await self.client.aclose()