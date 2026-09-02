import math
from db import get_conn, encode_embedding, decode_embedding
from llm import embed, extract_facts as llm_extract_facts

CONTRADICTION_SIM_THRESHOLD = 0.86  # cosine sim above this + different key -> ask LLM to judge
RETRIEVAL_TOP_K = 6


def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def store_message(session_id: str, turn: int, role: str, content: str):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, turn, role, content) VALUES (?, ?, ?, ?)",
            (session_id, turn, role, content),
        )


def get_active_fact_by_key(conn, session_id: str, key: str):
    row = conn.execute(
        "SELECT * FROM facts WHERE session_id=? AND fact_key=? AND status='active'",
        (session_id, key),
    ).fetchone()
    return row


def supersede(conn, old_id: int, new_id: int):
    conn.execute(
        "UPDATE facts SET status='superseded', superseded_by=? WHERE id=?",
        (new_id, old_id),
    )


def process_and_store_facts(session_id: str, turn: int, user_message: str) -> list[dict]:
    extracted = llm_extract_facts(user_message)
    stored = []

    with get_conn() as conn:
        for fact in extracted:
            key, value, category = fact.get("key"), fact.get("value"), fact.get("category", "other")
            if not key or not value:
                continue

            vec = embed(f"{key}: {value}")

            # Tier 1: exact key match -> direct supersede.
            existing = get_active_fact_by_key(conn, session_id, key)

            cur = conn.execute(
                """INSERT INTO facts
                   (session_id, fact_key, fact_value, category, embedding, source_turn)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (session_id, key, value, category, encode_embedding(vec), turn),
            )
            new_id = cur.lastrowid

            if existing is not None:
                supersede(conn, existing["id"], new_id)
            else:
                key_prefix = key.split(":")[0]

                candidates = conn.execute(
                    "SELECT * FROM facts WHERE session_id=? AND status='active' AND id != ?",
                    (session_id, new_id),
                ).fetchall()
                for cand in candidates:
                    if cand["fact_key"] == key:
                        continue
                    cand_prefix = cand["fact_key"].split(":")[0]
                    if ":" in key and ":" in cand["fact_key"] and cand_prefix == key_prefix:
                        continue  # siblings under the same set-type key — not a contradiction
                    sim = cosine_sim(vec, decode_embedding(cand["embedding"]))
                    if sim >= CONTRADICTION_SIM_THRESHOLD:
                        supersede(conn, cand["id"], new_id)
                        break

            stored.append({"key": key, "value": value, "category": category})

    return stored


def retrieve_relevant_facts(session_id: str, query: str, top_k: int = RETRIEVAL_TOP_K) -> list[str]:
    query_vec = embed(query)

    with get_conn() as conn:
        rows = conn.execute(
            "SELECT fact_key, fact_value, embedding FROM facts WHERE session_id=? AND status='active'",
            (session_id,),
        ).fetchall()

    scored = []
    for row in rows:
        sim = cosine_sim(query_vec, decode_embedding(row["embedding"]))
        scored.append((sim, row["fact_key"], row["fact_value"]))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]

    return [f"{key}: {value}" for _, key, value in top]


def get_recent_messages(session_id: str, limit: int = 12) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id=? ORDER BY turn DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
