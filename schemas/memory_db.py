# schemas/memory_db.py
import uuid
import os
from psycopg import connect
from core.routing.tool_vector_db import LangChainFastEmbedBridge
import numpy as np

DB_URI = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/cozmo_db")


class LongTermMemoryManager:
    def __init__(self):
        self.embedder = LangChainFastEmbedBridge()
        self._ensure_table_exists()

    def _ensure_table_exists(self):
        """Creates the user_profile_memories table with REAL[] arrays and user isolation. Runs safe DDL schema migrations."""
        with connect(DB_URI) as conn:
            with conn.cursor() as cur:
                # 1. Create table if not exists with correct schema
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS user_profile_memories (
                        id VARCHAR(36) PRIMARY KEY,
                        user_id VARCHAR(50) NOT NULL DEFAULT 'cozmo_owner',
                        fact TEXT NOT NULL,
                        embedding REAL[] NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                # 2. Schema migration: check and add user_id column
                cur.execute("""
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_name='user_profile_memories' AND column_name='user_id';
                """)
                if not cur.fetchone():
                    print("Migrating database user_profile_memories: adding user_id column...")
                    cur.execute("ALTER TABLE user_profile_memories ADD COLUMN user_id VARCHAR(50) NOT NULL DEFAULT 'cozmo_owner';")
                
                # 3. Schema migration: check and add updated_at column
                cur.execute("""
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_name='user_profile_memories' AND column_name='updated_at';
                """)
                if not cur.fetchone():
                    cur.execute("ALTER TABLE user_profile_memories ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;")
                
                # 4. Schema migration: check and add embedding column
                cur.execute("""
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_name='user_profile_memories' AND column_name='embedding';
                """)
                if not cur.fetchone():
                    print("Migrating database user_profile_memories: adding embedding REAL[] column...")
                    cur.execute("ALTER TABLE user_profile_memories ADD COLUMN embedding REAL[];")
                    
                    # Generate embeddings for legacy text-only rows on the fly
                    cur.execute("SELECT id, fact FROM user_profile_memories WHERE embedding IS NULL;")
                    legacy_rows = cur.fetchall()
                    if legacy_rows:
                        print(f"Retroactively embedding {len(legacy_rows)} legacy personal facts...")
                        for db_id, fact in legacy_rows:
                            embedding_vector = self.embedder.embed_query(fact)
                            cur.execute(
                                "UPDATE user_profile_memories SET embedding = %s WHERE id = %s;",
                                (embedding_vector, db_id)
                            )
                    
                    # Apply NOT NULL constraint after migration completes
                    cur.execute("ALTER TABLE user_profile_memories ALTER COLUMN embedding SET NOT NULL;")

    def save_memory(self, fact: str, user_id: str = "cozmo_owner"):
        """Saves a unique personal fact to the database. Overwrites semantically conflicting facts."""
        new_embedding = np.array(self.embedder.embed_query(fact))
        existing_memories = self._get_all_memories_for_user(user_id)

        if existing_memories:
            for db_id, db_fact, db_embedding_list in existing_memories:
                db_embedding = np.array(db_embedding_list)

                # Vectorized Cosine Similarity: (A . B) / (||A|| * ||B||)
                dot_prod = np.dot(new_embedding, db_embedding)
                norm_new = np.linalg.norm(new_embedding)
                norm_db = np.linalg.norm(db_embedding)

                similarity = dot_prod / (norm_new * norm_db) if (norm_new > 0 and norm_db > 0) else 0.0

                # If similarity is very high (e.g. > 0.82), update/overwrite the old entry
                if similarity > 0.82:
                    with connect(DB_URI) as conn:
                        with conn.cursor() as cur:
                            cur.execute(
                                """
                                UPDATE user_profile_memories 
                                SET fact = %s, embedding = %s, updated_at = CURRENT_TIMESTAMP 
                                WHERE id = %s;
                                """,
                                (fact, new_embedding.tolist(), db_id)
                            )
                    return

        # If it's a completely new fact, insert it
        with connect(DB_URI) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO user_profile_memories (id, user_id, fact, embedding) 
                    VALUES (%s, %s, %s, %s);
                    """,
                    (str(uuid.uuid4()), user_id, fact, new_embedding.tolist())
                )

    def _get_all_memories_for_user(self, user_id: str):
        """Helper to retrieve all raw facts and precomputed vectors for a user."""
        with connect(DB_URI) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id, fact, embedding FROM user_profile_memories WHERE user_id = %s;", (user_id,))
                return cur.fetchall()

    def retrieve_relevant_memories(self, user_query: str, user_id: str = "cozmo_owner", limit: int = 3) -> list[str]:
        """Calculates relevance scoring inside Python using precomputed Postgres vectors and NumPy."""
        all_rows = self._get_all_memories_for_user(user_id)
        if not all_rows:
            return []

        # Embed user's query exactly once per turn
        query_embedding = np.array(self.embedder.embed_query(user_query))
        norm_query = np.linalg.norm(query_embedding)
        if norm_query == 0:
            return []

        scored_memories = []

        for _, fact, db_embedding_list in all_rows:
            db_embedding = np.array(db_embedding_list)
            norm_db = np.linalg.norm(db_embedding)
            if norm_db == 0:
                continue

            similarity = np.dot(query_embedding, db_embedding) / (norm_query * norm_db)
            scored_memories.append((fact, similarity))

        # Sort by highest similarity score
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        return [mem[0] for mem in scored_memories[:limit]]


long_term_memory = LongTermMemoryManager()