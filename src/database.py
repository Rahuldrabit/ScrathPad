import sqlite3
import os

DB_PATH = "scratchpad_memory.db"

def get_db_connection():
    """Initializes a connection with WAL mode for high concurrency."""
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.row_factory = sqlite3.Row
    return conn

def initialize_database():
    """Creates the necessary schemas if they do not exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Global Session Tracker
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            user_query TEXT,
            master_plan TEXT,
            global_status TEXT DEFAULT 'PLANNING'
        )
    """)
    
    # 2. Knowledge Graph Triplet Store
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_graph (
            edge_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            agent_id TEXT,
            source_entity TEXT NOT NULL,
            relationship TEXT NOT NULL,
            target_entity TEXT NOT NULL,
            citation_quote TEXT,
            hierarchy_level INTEGER DEFAULT 1,
            parent_node_id TEXT DEFAULT NULL,
            relevance_score REAL DEFAULT 1.0,
            is_active BOOLEAN DEFAULT 1,
            extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(session_id) REFERENCES sessions(session_id),
            UNIQUE(session_id, source_entity, relationship, target_entity) 
            ON CONFLICT REPLACE
        )
    """)
    
    # Add composite index for fast SYNC paths
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_active_session_relevance 
        ON knowledge_graph (session_id, is_active, relevance_score DESC)
    """)
    
    # 3. Unresolved Variables Matrix
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS unresolved_variables (
            variable_id TEXT PRIMARY KEY,
            session_id TEXT,
            variable_name TEXT NOT NULL,
            status TEXT DEFAULT 'MISSING',
            FOREIGN KEY(session_id) REFERENCES sessions(session_id),
            UNIQUE(session_id, variable_name) ON CONFLICT IGNORE
        )
    """)
    
    conn.commit()
    conn.close()
    _migrate_knowledge_graph()


def _migrate_knowledge_graph():
    """
    Idempotent migration: add type / provenance columns to
    knowledge_graph if they don't exist yet. SQLite ALTER TABLE doesn't
    support IF NOT EXISTS for ADD COLUMN, so we check PRAGMA first.

    Safe to call on every initialize_database() — the existence check
    makes it a no-op once the migration has run.

    Columns:
      - source_type / target_type: EntityType literal (SERVICE, FILE, …)
      - extractor:                who wrote this row (agent, middleware,
                                  sweeper_l2, observation, plan)
      - pass_number:              which extraction pass (0 for single-pass,
                                  1..N for multi-pass consensus)
      - raw_citation_score:       the Layer 4 score at commit time, useful
                                  for "this row is on the edge of being
                                  rejected — what happened?"
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(knowledge_graph)")
    existing = {row[1] for row in cursor.fetchall()}
    for col, default in (
        ("source_type", "UNKNOWN"),
        ("target_type", "UNKNOWN"),
        ("extractor", "unknown"),
        ("pass_number", "0"),
        ("raw_citation_score", "0.0"),
    ):
        if col not in existing:
            cursor.execute(
                f"ALTER TABLE knowledge_graph ADD COLUMN {col} TEXT DEFAULT '{default}'"
            )
    conn.commit()
    conn.close()

if __name__ == "__main__":
    initialize_database()
    print("Database initialized successfully.")