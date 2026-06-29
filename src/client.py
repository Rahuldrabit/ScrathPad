import uuid
from typing import Dict, Any, Optional
from inference import UniversalInferenceEngine
from database import get_db_connection
from engine import canonicalize_entity

class ScratchpadMiddleware:
    def __init__(self, token_window: int = 8192):
        self.inference_engine = UniversalInferenceEngine()
        self.token_window = token_window
        self.t_system_safety_buffer = 1500 

    def process_turn(self, session_id: str, messy_input: str) -> str:
        """
        Universal wrapper: Ingests raw messy text, extracts structured 
        knowledge, commits it, and returns a clean, token-bounded context.
        """
        self.commit_messy_input(session_id, messy_input)
        return self.get_clean_context(session_id)

    def commit_messy_input(self, session_id: str, raw_text: str):
        """Extracts triplets from messy text, runs verification, and commits."""
        system_prompt = "Extract clear entity-relationship triplets from the messy, unstructured text."
        
        from schema import L1ExtractionPayload 
        try:
            extracted_payload = self.inference_engine.generate_structured(
                prompt=raw_text,
                system_prompt=system_prompt,
                response_schema=L1ExtractionPayload
            )
            
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("BEGIN TRANSACTION;")
            
            for triplet in extracted_payload.triplets:
                src = canonicalize_entity(cursor, session_id, triplet.source_entity)
                tgt = canonicalize_entity(cursor, session_id, triplet.target_entity)
                
                # Verification Gate
                if triplet.citation_quote in raw_text:
                    cursor.execute("""
                        INSERT INTO knowledge_graph 
                        (edge_id, session_id, source_entity, relationship, target_entity, citation_quote, is_active)
                        VALUES (?, ?, ?, ?, ?, ?, TRUE)
                    """, (str(uuid.uuid4()), session_id, src, triplet.relationship, tgt, triplet.citation_quote))
            
            conn.commit()
        except Exception as e:
            if 'conn' in locals():
                conn.rollback()
            print(f"[MIDDLEWARE ERROR] Ingestion failed: {e}")
        finally:
            if 'conn' in locals():
                conn.close()

    def get_clean_context(self, session_id: str) -> str:
        """Queries the active hierarchical graph and builds the Markdown scratchpad."""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT source_entity, relationship, target_entity, hierarchy_level, edge_id
            FROM knowledge_graph 
            WHERE session_id = ? AND is_active = TRUE 
            ORDER BY relevance_score DESC
        """, (session_id,))
        
        active_rows = cursor.fetchall()
        conn.close()
        
        markdown_buffer = "### ACTIVE KNOWLEDGE SCRATCHPAD\n"
        for row in active_rows:
            if row['hierarchy_level'] == 2:
                markdown_buffer += f"* [{row['source_entity']}] --({row['relationship']})--> [{row['target_entity']}] `[COMPRESSED | ID: {row['edge_id']}]`\n"
            else:
                markdown_buffer += f"* [{row['source_entity']}] --({row['relationship']})--> [{row['target_entity']}]\n"
                
        return markdown_buffer
