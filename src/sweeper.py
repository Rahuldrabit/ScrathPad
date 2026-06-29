import asyncio
import sqlite3
import networkx as nx
from community import community_louvain # from python-louvain
from database import get_db_connection

# Configuration
DEGREE_THRESHOLD = 15
SWEEPER_INTERVAL_SECONDS = 10

async def call_slm_for_compression(triplets: list) -> list:
    """
    Mock function: Replace this with your actual local SLM API call.
    Pass the raw triplets and ask the SLM to return 1-3 high-level summary triplets.
    """
    # Example simulation of an SLM response
    return [{
        "source_entity": "L2_SUMMARY_NODE",
        "relationship": "abstracts",
        "target_entity": "GRANULAR_CLUSTER",
        "citation_quote": "Generated via asynchronous SLM compression."
    }]

def run_graph_compression():
    """Synchronous CPU-bound function executed by the daemon."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Step 1: Find Hubs (Nodes exceeding the degree threshold)
        cursor.execute("""
            SELECT entity, COUNT(*) as degree FROM (
                SELECT source_entity as entity FROM knowledge_graph WHERE is_active = TRUE
                UNION ALL
                SELECT target_entity as entity FROM knowledge_graph WHERE is_active = TRUE
            ) GROUP BY entity HAVING degree >= ?
        """, (DEGREE_THRESHOLD,))
        
        hubs = [row['entity'] for row in cursor.fetchall()]
        
        if not hubs:
            return # Graph is healthy, no compression needed
            
        print(f"[SWEEPER] Detected dense hubs requiring compression: {hubs}")
        
        # Step 2: Load the active graph into NetworkX
        cursor.execute("SELECT edge_id, source_entity, target_entity FROM knowledge_graph WHERE is_active = TRUE")
        edges = cursor.fetchall()
        
        G = nx.Graph()
        for edge in edges:
            G.add_edge(edge['source_entity'], edge['target_entity'], edge_id=edge['edge_id'])
            
        # Step 3: Run Louvain Community Detection
        # partition is a dict mapping node -> community_id
        partition = community_louvain.best_partition(G)
        
        # Step 4: Group edges by community to isolate the subgraphs
        communities = {}
        for u, v, data in G.edges(data=True):
            c_u, c_v = partition[u], partition[v]
            if c_u == c_v: # Only compress tightly bound internal edges
                if c_u not in communities:
                    communities[c_u] = []
                communities[c_u].append(data['edge_id'])
                
        # Step 5: Compress large communities
        for community_id, edge_ids in communities.items():
            if len(edge_ids) >= DEGREE_THRESHOLD:
                print(f"[SWEEPER] Compressing community {community_id} with {len(edge_ids)} edges...")
                
                # Fetch full triplet data for the SLM
                placeholders = ','.join('?' for _ in edge_ids)
                cursor.execute(f"SELECT * FROM knowledge_graph WHERE edge_id IN ({placeholders})", edge_ids)
                raw_triplets = cursor.fetchall()
                
                # ---> CALL LOCAL SLM HERE <---
                # (You would typically run this via an async client, but for simplicity we mock it)
                # new_l2_triplets = await call_slm_for_compression(raw_triplets)
                
                # Simulate the SLM returning an L2 summary
                new_l2_triplets = [{"source_entity": "CLUSTER_SUMMARY", "relationship": "contains", "target_entity": "MULTIPLE_ENTITIES", "citation_quote": "SLM Summary"}]
                
                # Step 6: The Database Swap (Atomic Transaction)
                cursor.execute("BEGIN TRANSACTION;")
                
                import uuid
                l2_parent_id = str(uuid.uuid4())
                
                # 6a: Insert new L2 triplet
                for t in new_l2_triplets:
                    cursor.execute("""
                        INSERT INTO knowledge_graph 
                        (edge_id, source_entity, relationship, target_entity, citation_quote, hierarchy_level, is_active)
                        VALUES (?, ?, ?, ?, ?, 2, TRUE)
                    """, (l2_parent_id, t['source_entity'], t['relationship'], t['target_entity'], t['citation_quote']))
                
                # 6b: Deactivate old granular triplets and link them to the parent
                cursor.execute(f"""
                    UPDATE knowledge_graph 
                    SET is_active = FALSE, parent_node_id = ? 
                    WHERE edge_id IN ({placeholders})
                """, [l2_parent_id] + edge_ids)
                
                conn.commit()
                print(f"[SWEEPER] Successfully compressed community {community_id}.")
                
    except Exception as e:
        conn.rollback()
        print(f"[SWEEPER ERROR] Failed to compress graph: {e}")
    finally:
        conn.close()

async def graph_maintenance_daemon():
    """
    The infinite async loop that runs in the background.
    It yields control back to FastAPI via asyncio.sleep.
    """
    print("[DAEMON] Graph Maintenance Sweeper initialized.")
    while True:
        try:
            # Sleep first to give agents time to work
            await asyncio.sleep(SWEEPER_INTERVAL_SECONDS)
            
            # Run the heavy CPU/SQLite logic in a threadpool so it doesn't block FastAPI WebSockets
            from fastapi.concurrency import run_in_threadpool
            await run_in_threadpool(run_graph_compression)
            
        except asyncio.CancelledError:
            print("[DAEMON] Sweeper shutting down.")
            break
        except Exception as e:
            print(f"[DAEMON FATAL] {e}")
            await asyncio.sleep(SWEEPER_INTERVAL_SECONDS) # Prevent rapid error looping