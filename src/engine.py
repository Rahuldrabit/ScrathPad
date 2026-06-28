import uuid
from database import get_db_connection
from schema import PageExtractionPayload

def verify_citation(raw_chunk: str, citation: str) -> bool:
    """Anti-hallucination guardrail. Ensures exact matches only."""
    if not citation or citation.strip() == "":
        return False
    return citation.strip() in raw_chunk

async def commit_page_data_to_sqlite(session_id: str, agent_id: str, raw_chunk: str, extraction_data: PageExtractionPayload) -> int:
    """Runs the deterministic verification engine and bulk upserts verified facts."""
    conn = get_db_connection()
    cursor = conn.cursor()
    verified_count = 0
    
    try:
        cursor.execute("BEGIN TRANSACTION;")
        
        # Ensure session exists to prevent foreign key constraints from failing
        cursor.execute("INSERT OR IGNORE INTO sessions (session_id) VALUES (?)", (session_id,))
        
        for triplet in extraction_data.extracted_triplets:
            if not verify_citation(raw_chunk, triplet.citation_quote):
                print(f"[GUARDRAIL] Rejected hallucinated triplet: {triplet.source_entity} -> {triplet.target_entity}")
                continue 
                
            edge_id = str(uuid.uuid4())
            cursor.execute("""
                INSERT OR REPLACE INTO knowledge_graph 
                (edge_id, session_id, agent_id, source_entity, relationship, target_entity, citation_quote)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (edge_id, session_id, agent_id, triplet.source_entity.upper().strip(), 
                  triplet.relationship.lower().strip(), triplet.target_entity.upper().strip(), 
                  triplet.citation_quote.strip()))
            
            verified_count += 1
            
        for var_name, status in extraction_data.unresolved_variables_mutations.items():
            var_id = f"{session_id}_{var_name}"
            if status.upper() == "RESOLVED":
                cursor.execute("DELETE FROM unresolved_variables WHERE variable_id = ?", (var_id,))
            else:
                cursor.execute("""
                    INSERT OR IGNORE INTO unresolved_variables (variable_id, session_id, variable_name, status)
                    VALUES (?, ?, ?, ?)
                """, (var_id, session_id, var_name.upper().strip(), status.upper().strip()))
                
        conn.commit()
    except Exception as e:
        conn.execute("ROLLBACK;")
        print(f"[DATABASE ERROR] Transaction aborted: {e}")
        raise e
    finally:
        conn.close()
        
    return verified_count

def compile_graph_memory_to_markdown(session_id: str) -> str:
    """Assembles a clean GitHub-flavored Markdown view from SQLite."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT source_entity, relationship, target_entity, citation_quote 
        FROM knowledge_graph 
        WHERE session_id = ?
        ORDER BY extracted_at ASC
    """, (session_id,))
    
    graph_rows = cursor.fetchall()
    
    cursor.execute("""
        SELECT variable_name, status 
        FROM unresolved_variables 
        WHERE session_id = ?
    """, (session_id,))
    
    var_rows = cursor.fetchall()
    conn.close()
    
    md_output = ["## 1. KNOWLEDGE GRAPH MEMORY (Verified Facts)"]
    if not graph_rows:
        md_output.append("*(No active knowledge graph nodes established for this session)*\n")
    else:
        for row in graph_rows:
            md_output.append(
                f"* `[{row['source_entity']}]` --({row['relationship']})--> `[{row['target_entity']}]` \n"
                f"  └── Source Citation: \"{row['citation_quote']}\""
            )
            
    md_output.append("\n## 2. UNRESOLVED VARIABLES MATRIX")
    if not var_rows:
        md_output.append("*(No active variables currently tracked)*")
    else:
        for row in var_rows:
            md_output.append(f"- [?] `{row['variable_name']}` (Status: {row['status']})")
        
    return "\n".join(md_output)