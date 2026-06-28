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
            session_id TEXT,
            agent_id TEXT,
            source_entity TEXT NOT NULL,
            relationship TEXT NOT NULL,
            target_entity TEXT NOT NULL,
            citation_quote TEXT NOT NULL,
            extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(session_id) REFERENCES sessions(session_id),
            UNIQUE(session_id, source_entity, relationship, target_entity) 
            ON CONFLICT REPLACE
        )
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

if __name__ == "__main__":
    initialize_database()
    print("Database initialized successfully.")