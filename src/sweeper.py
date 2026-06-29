import asyncio
import sqlite3
import uuid
import networkx as nx
from community import community_louvain # from python-louvain
from database import get_db_connection
from fastapi.concurrency import run_in_threadpool

from inference import UniversalInferenceEngine
from schema import L2CompressionPayload

class GraphSweeperDaemon:
    def __init__(self, check_interval=10):
        self.lock = asyncio.Lock()
        self.check_interval = check_interval
        self.degree_threshold = 15
        self.min_community_size = 10
        self.engine = UniversalInferenceEngine()

    async def start_loop(self):
        print("[DAEMON] Graph Maintenance Sweeper initialized.")
        while True:
            try:
                await asyncio.sleep(self.check_interval)
                if not self.lock.locked():
                    async with self.lock:
                        await run_in_threadpool(self.execute_maintenance_sweep)
            except asyncio.CancelledError:
                print("[DAEMON] Sweeper shutting down.")
                break
            except Exception as e:
                print(f"[DAEMON FATAL] {e}")
                await asyncio.sleep(self.check_interval)

    def compress_louvain_community(self, community_id, edge_ids, raw_triplets):
        markdown_triplets = "\n".join([f"* {t['source_entity']} -> {t['relationship']} -> {t['target_entity']} (citation: {t['citation_quote']})" for t in raw_triplets])
        system_instruction = "Collapse these granular L1 triplets into a dense macro L2 summary."
        prompt_content = f"Target Community Subgraph:\n{markdown_triplets}"
        
        structured_payload: L2CompressionPayload = self.engine.generate_structured(
            prompt=prompt_content,
            system_prompt=system_instruction,
            response_schema=L2CompressionPayload
        )
        return structured_payload

    def execute_maintenance_sweep(self):
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Load active graph edges partitioned by session
            cursor.execute("SELECT edge_id, session_id, source_entity, target_entity FROM knowledge_graph WHERE is_active = TRUE")
            edges = cursor.fetchall()
            
            if not edges:
                return
                
            # Group edges by session to prevent cross-session clustering
            session_graphs = {}
            for edge in edges:
                sid = edge['session_id']
                if sid not in session_graphs:
                    session_graphs[sid] = nx.Graph()
                session_graphs[sid].add_edge(edge['source_entity'], edge['target_entity'], edge_id=edge['edge_id'])
                
            for session_id, G in session_graphs.items():
                # Check if any node in this session exceeds degree threshold
                degrees = [d for n, d in G.degree()]
                if not degrees or max(degrees) < self.degree_threshold:
                    continue
                    
                print(f"[SWEEPER] Detected dense hubs in session {session_id}")
                
                # Run Louvain Community Detection
                partition = community_louvain.best_partition(G, random_state=42)
                
                # Group edges
                communities = {}
                for u, v, data in G.edges(data=True):
                    c_u, c_v = partition[u], partition[v]
                    if c_u == c_v: 
                        if c_u not in communities:
                            communities[c_u] = []
                        communities[c_u].append(data['edge_id'])
                        
                # Compress
                for community_id, edge_ids in communities.items():
                    if len(edge_ids) >= self.min_community_size:
                        print(f"[SWEEPER] Compressing community {community_id} with {len(edge_ids)} edges...")
                        
                        placeholders = ','.join('?' for _ in edge_ids)
                        cursor.execute(f"SELECT * FROM knowledge_graph WHERE edge_id IN ({placeholders})", edge_ids)
                        raw_triplets = cursor.fetchall()
                        
                        try:
                            l2_payload = self.compress_louvain_community(community_id, edge_ids, raw_triplets)
                            
                            cursor.execute("BEGIN TRANSACTION;")
                            
                            first_l2_id = None
                            
                            for i, t in enumerate(l2_payload.extracted_l2_triplets):
                                l2_id = str(uuid.uuid4())
                                if i == 0:
                                    first_l2_id = l2_id
                                    
                                cursor.execute("""
                                    INSERT INTO knowledge_graph 
                                    (edge_id, session_id, source_entity, relationship, target_entity, citation_quote, hierarchy_level, is_active)
                                    VALUES (?, ?, ?, ?, ?, ?, 2, TRUE)
                                """, (l2_id, session_id, t.source_entity, t.relationship, t.target_entity, t.citation_quote))
                            
                            if first_l2_id:
                                cursor.execute(f"""
                                    UPDATE knowledge_graph 
                                    SET is_active = FALSE, parent_node_id = ? 
                                    WHERE edge_id IN ({placeholders})
                                """, [first_l2_id] + edge_ids)
                            
                            conn.commit()
                            print(f"[SWEEPER] Successfully compressed community {community_id}.")
                        except Exception as e:
                            print(f"[SWEEPER ERROR] Failed to compress community {community_id}: {e}")
                            conn.rollback()
                            
        except Exception as e:
            conn.rollback()
            print(f"[SWEEPER ERROR] Graph maintenance failed: {e}")
        finally:
            conn.close()

# Keep backward compatibility if anything else imports graph_maintenance_daemon
daemon_instance = GraphSweeperDaemon()
graph_maintenance_daemon = daemon_instance.start_loop