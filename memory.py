import math
from db import get_conn, encode_embedding, decode_embedding
from llm import embed, extract_facts as llm_extract_facts

CONTRADICTION_SIM_THRESHOLD = 0.86  # cosine sim above this + different key -> ask LLM to judge
RETRIEVAL_TOP_K = 6

# Values too generic/uninformative to be allowed to overwrite a more specific
# prior value. An extractor emitting one of these for an existing key is
# treated as "couldn't determine it" rather than "here's the new truth" —
# e.g. job_title going from "backend engineer" to "unknown" should not erase
# the specific title.
_GENERIC_VALUES = {
    "unknown", "n/a", "na", "none", "unclear", "tbd", "not sure",
    "not specified", "unspecified", "",
}

# Explicit map of "<topic>_status" key prefixes to the specific other key
# prefixes they invalidate when they change. This replaces fuzzy topic-word
# overlap, which was too blunt (e.g. any "job_*" key would get superseded by
# job_status just for sharing the word "job", even keys like job_location
# that a status change doesn't necessarily invalidate).
#
# Extend this table deliberately rather than relying on string heuristics —
# if a new status key is introduced, decide explicitly what it invalidates.
_STATUS_INVALIDATES = {
    "job_status": ["job_title"],
    "relationship_status": [],  # relationship_status is authoritative on its own;
                                 # it does not by itself invalidate relationship_length
                                 # (that's a historical fact about a past relationship).
    "dietary_status": ["dietary_goal"],
}


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


def _is_generic(value: str) -> bool:
    return value.strip().lower() in _GENERIC_VALUES


# Words too generic to establish that two keys share a topic on their own
# (e.g. without this, "job_status" and "relationship_status" would look like
# the same topic just because both contain "status").
_TOPIC_STOPWORDS = {"status", "of", "the", "a", "an", "current", "is", "state"}


def _key_prefix(key: str) -> str:
    return key.split(":")[0]


def _key_entity(key: str) -> str | None:
    # For set-type keys like "pet_name:max", returns the entity id ("max").
    # Returns None for simple keys with no entity component.
    parts = key.split(":", 1)
    return parts[1].lower() if len(parts) == 2 else None


def _topic_words(key: str) -> set[str]:
    base = _key_prefix(key).lower()
    return {w for w in base.split("_") if w and w not in _TOPIC_STOPWORDS}


def process_and_store_facts(session_id: str, turn: int, user_message: str) -> list[dict]:
    extracted = llm_extract_facts(user_message)
    stored = []

    with get_conn() as conn:
        for fact in extracted:
            key, value, category = fact.get("key"), fact.get("value"), fact.get("category", "other")
            if not key or not value:
                continue

            vec = embed(f"{key}: {value}")

            # Tier 1: exact key match -> direct supersede, UNLESS the new
            # value is too generic to responsibly overwrite a specific prior
            # value (e.g. "unknown" replacing "backend engineer"). In that
            # case we still record the generic fact for visibility, but we
            # do NOT retire the old one, so retrieval keeps surfacing the
            # more specific, still-plausible fact.
            existing = get_active_fact_by_key(conn, session_id, key)

            cur = conn.execute(
                """INSERT INTO facts
                   (session_id, fact_key, fact_value, category, embedding, source_turn)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (session_id, key, value, category, encode_embedding(vec), turn),
            )
            new_id = cur.lastrowid

            if existing is not None and not (_is_generic(value) and not _is_generic(existing["fact_value"])):
                supersede(conn, existing["id"], new_id)

            # Tier 2: look for other active facts that this new fact invalidates,
            # even when the key doesn't match exactly.
            #
            # Only consider facts from strictly earlier turns. Facts inserted
            # earlier in THIS SAME extraction batch (same turn) are visible in
            # the table already since each insert commits immediately, but a
            # sibling fact from the same message should not be able to
            # supersede another sibling fact from that same message purely
            # because of insert order — e.g. "job_title: fintech" and
            # "job_status:quit: yes" extracted from one message shouldn't
            # race to invalidate each other.
            key_prefix = _key_prefix(key)
            key_entity = _key_entity(key)
            topic_words = _topic_words(key)
            is_status_update = key_prefix.endswith("_status")
            invalidates = _STATUS_INVALIDATES.get(key_prefix, [])

            candidates = conn.execute(
                """SELECT * FROM facts
                   WHERE session_id=? AND status='active' AND id != ? AND source_turn < ?""",
                (session_id, new_id, turn),
            ).fetchall()
            for cand in candidates:
                cand_key = cand["fact_key"]
                if cand_key == key:
                    continue
                cand_prefix = _key_prefix(cand_key)
                cand_entity = _key_entity(cand_key)

                if ":" in key and ":" in cand_key and cand_prefix == key_prefix:
                    continue  # siblings under the same set-type key — not a contradiction

                if key_entity is not None and cand_entity == key_entity and cand_prefix != key_prefix:
                    # Same entity, different attribute (e.g. pet_name:max vs
                    # pet_type:max). These describe different facets of the
                    # same thing and should never be silently superseded by
                    # each other just because their embedded text happens to
                    # be similar (e.g. both extracted as "golden retriever").
                    # If one of them is actually mislabeled, that's an
                    # extraction bug to fix upstream, not something this
                    # layer should paper over by deleting data.
                    continue

                # Explicit status-invalidation: a "<topic>_status" fact is a
                # definitive state change for specific, named keys only —
                # not anything that merely shares a topic word.
                if is_status_update and cand_prefix in invalidates:
                    supersede(conn, cand["id"], new_id)
                    continue

                sim = cosine_sim(vec, decode_embedding(cand["embedding"]))
                if sim >= CONTRADICTION_SIM_THRESHOLD:
                    supersede(conn, cand["id"], new_id)

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