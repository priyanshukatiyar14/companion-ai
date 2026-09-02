import os
import sys
import json

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

# --- Ollama config (local, free) ---
# Ollama serves an OpenAI-compatible API at this default address once it's
# running (the installer sets it up as a background service). No API key is
# needed — the client library requires *some* string, so "ollama" is passed
# as a placeholder and is not checked by the server.
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "llama3.1:8b")
EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# --- OpenAI config (hosted, paid) ---
# API_KEY = os.environ.get("OPENAI_API_KEY")
# if not API_KEY:
#     sys.exit("ERROR: OPENAI_API_KEY is not set. See .env.example.")
# client = OpenAI(api_key=API_KEY)
# CHAT_MODEL = "gpt-4o-mini"
# EMBED_MODEL = "text-embedding-3-small"


def embed(text: str) -> list[float]:
    try:
        resp = client.embeddings.create(model=EMBED_MODEL, input=text)
        return resp.data[0].embedding
    except Exception as e:
        _handle_connection_error(e)


def chat_completion(messages: list[dict]) -> str:
    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.8,
        )
        return resp.choices[0].message.content
    except Exception as e:
        _handle_connection_error(e)


def _handle_connection_error(e: Exception):
    msg = str(e).lower()
    if "connection" in msg or "refused" in msg:
        sys.exit(
            "ERROR: couldn't reach Ollama at "
            f"{OLLAMA_BASE_URL}.\n"
            "Is Ollama running? Try: ollama serve\n"
            f"Original error: {e}"
        )
    raise


EXTRACTION_PROMPT = """You extract durable, memory-worthy facts from a single \
user message in an ongoing conversation. A fact is memory-worthy if it would \
still matter in a future conversation: relationships, job/work situation, \
plans, stated opinions/preferences, significant events, recurring context.

Do NOT extract: small talk, questions the user asked, one-off transient \
statements ("I'm tired right now"), or anything you're inferring rather than \
something stated.

For each fact, produce a short stable `key` (snake_case) that a later, \
unrelated message about the same topic would also map to — this is what \
lets us detect contradictions later. Use a consistent, minimal vocabulary.

IMPORTANT — singular ("slot") facts vs. plural ("set") facts:
- A SLOT fact has exactly one true value at a time — a new value replaces the
  old one (e.g. "relationship_status", "job_title", "current_city"). Key these
  with a plain name: "job_title".
- A SET fact is one of potentially many true at once — the user can have
  several pets, hobbies, siblings, past jobs (e.g. "pet_name", "hobby",
  "sibling"). Key these with the pattern "type:identifier", using a short
  identifier drawn from the value itself, e.g. a pet named Max becomes
  key "pet_name:max", a hobby of guitar becomes "hobby:guitar". This keeps
  each one as its own fact instead of overwriting the last.
If you're unsure whether something is a slot or a set, prefer treating it as
a set (the "type:identifier" form) — losing an old value silently is worse
than keeping an extra one.

Return ONLY a JSON array (no markdown fences, no prose), each item:
{"key": str, "value": str, "category": "relationship"|"work"|"plan"|"opinion"|"preference"|"event"|"other"}

If there is nothing memory-worthy, return [].

User message: __USER_MESSAGE__
"""

REPAIR_PROMPT = """Your previous response was supposed to be a JSON array but \
could not be parsed. Here is what you returned:

__BAD_OUTPUT__

Return ONLY a valid JSON array in the exact format requested — no markdown \
fences, no prose, no trailing commas. If nothing is extractable, return [].
"""


def extract_facts(user_message: str) -> list[dict]:
    prompt = EXTRACTION_PROMPT.replace("__USER_MESSAGE__", user_message)
    raw = chat_completion([{"role": "user", "content": prompt}])

    facts = _try_parse_fact_list(raw)
    if facts is not None:
        return facts

    # One repair attempt.
    repair = REPAIR_PROMPT.replace("__BAD_OUTPUT__", raw)
    raw_retry = chat_completion([{"role": "user", "content": repair}])
    facts = _try_parse_fact_list(raw_retry)
    if facts is not None:
        return facts

    return []


def _try_parse_fact_list(raw: str) -> list[dict] | None:
    try:
        facts = json.loads(raw)
        assert isinstance(facts, list)
        return facts
    except (json.JSONDecodeError, AssertionError):
        return None