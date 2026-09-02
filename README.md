# Companion-AI Memory Architecture

A CLI chat companion with persistent, contradiction-aware memory and a
stable persona.

## Setup

```bash
pip install -r requirements.txt
```

**Provider: this is configured for Ollama (local, free) by default — no API
key needed.** Install Ollama, pull the two models, and make sure it's
running:

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
ollama serve      # if it isn't already running as a background service
python chat.py            # uses session "default"
python chat.py alice      # named session, for testing multiple "users"
```

If Ollama isn't reachable, the app exits with a clear message telling you
to run `ollama serve`, rather than failing with an opaque connection error.

**Why Ollama instead of the OpenAI API:** it's free and runs entirely
locally — no API key, no per-token cost, no rate limits to worry about
while iterating on prompts or running the eval harness repeatedly (each run
of `eval_harness.py` or `long_transcript_test.py` makes dozens of calls,
which adds up fast on a metered API). It's also fast enough for this
exercise: once the model is loaded into memory, `llama3.1:8b` responds in
roughly the same ballpark as a hosted API call over a normal connection, so
the interactive chat loop doesn't feel sluggish in practice. The trade-off
is documented below — smaller local models are less consistent than
GPT-4o-mini at reliably returning clean JSON for the fact-extraction step,
which is why `extract_facts()` has a repair-retry built in.

To switch to hosted OpenAI instead (stronger extraction quality, costs
money), see the commented block at the top of `llm.py` — swap it in, then:
```bash
cp .env.example .env      # then fill in your key, OR export it directly:
export OPENAI_API_KEY=sk-...
```

Restart the process and reuse the same session name — facts and history
persist in `companion_memory.db` (SQLite, created on first run).

## Architecture decisions

**Storage: structured facts + per-fact embeddings (hybrid), in SQLite.**
Pure embedding-only memory (dump everything into a vector store) can't do
contradiction handling — there's no natural "slot" to overwrite, only
similar-but-separate vectors. Pure structured-only memory can't do fuzzy
semantic recall of things that don't reduce cleanly to a key/value pair.
So: every extracted fact gets a `fact_key` (LLM-assigned, snake_case,
meant to be stable across paraphrasings — e.g. "relationship_status") *and*
an embedding of `key: value`. SQLite over Postgres/pgvector purely because
this needs zero setup to run the eval harness and grading — the schema
maps 1:1 onto Postgres + pgvector if this needed to scale.

**Extraction:** one LLM call per user turn, prompted to return a JSON array
of `{key, value, category}`. Kept deliberately conservative — small talk
and transient statements aren't extracted, only things "which would still
matter in a future conversation."

**Contradiction / update handling — two tiers:**
1. Exact `fact_key` match → new fact directly supersedes the old one
   (`status=superseded`, `superseded_by=<new id>`). This is the common case:
   the same slot ("relationship_status") gets a new value.
2. No exact match → check cosine similarity against all other active facts.
   If similarity clears a conservative threshold (0.86) *and* the key
   differs, treat it as a paraphrased contradiction and supersede. This
   catches cases where the extractor assigns a slightly different key to
   what's semantically the same fact. The threshold is deliberately high:
   a false-positive supersede silently destroys a real memory, which is a
   worse failure mode than an occasional missed contradiction.

Superseded facts are never deleted — they're excluded from retrieval and
context, but kept in the DB. This was a deliberate choice for both
debuggability and honesty: "the system decided X was no longer true" should
be inspectable, and it's what the eval harness's judge checks against.

**Slot facts vs. set facts.** Not every fact should overwrite the last one —
a user can have two pets, several hobbies, multiple past jobs. The extraction
prompt distinguishes:
- *Slot* facts (one true value at a time — job title, relationship status,
  current city) get a plain key like `"job_title"`.
- *Set* facts (many true at once — pets, hobbies, siblings) get a
  `"type:identifier"` key, e.g. `"pet_name:max"` and `"pet_name:bella"` for
  two different pets. This keeps them as separate rows instead of the second
  one silently overwriting the first as a false "contradiction."

The tier-2 (embedding-similarity) contradiction check respects this: two
facts sharing a set-type prefix (same `type`, different `identifier`) are
explicitly excluded from being treated as contradicting each other, even if
their embeddings read as similar (two pets described in similar sentences,
for instance).

**Extraction retry.** If the extraction call returns something that doesn't
parse as a JSON list, one repair attempt is made (the bad output is sent
back with a "fix this" prompt) before the turn's facts are silently dropped.
Extraction must never be allowed to crash the chat loop.

**Retrieval:** every user turn is embedded and compared against all *active*
facts for that session; the top-K (default 6) by cosine similarity are
injected into the system prompt as plain lines, not as a "here is your
memory dump" block — the prompt explicitly tells the model to use them
naturally rather than reciting them. This is what keeps "everything in
context" (which defeats the exercise) from happening as fact count grows.

**Persona:** defined once in `persona.yaml` (backstory, traits, stated
opinions, speech style) and injected into *every* system prompt, in full,
unconditionally. It is structurally separate from the user-memory pipeline
— nothing in the extraction/contradiction logic can ever touch it. This is
the main defense against "persona flattens under topic pressure": the
model is never asked to reconcile persona facts against user facts, because
they're different tables entirely.

**Short-term continuity:** the last 12 raw messages are also included in
every prompt (independent of the fact store) so the model has natural
short-range coherence without relying on the fact-extraction pipeline to
catch everything.

## What was tried and abandoned

- **Single unified vector store, no structured keys:** tried this first;
  contradiction handling degenerated into "is this new thing similar enough
  to some old thing to maybe replace it," which was unreliable and had no
  clean way to represent "this specific slot now has this specific value."
  Moved to the current hybrid approach.
- **Dumping all active facts into context every turn:** works fine for the
  first ~15 facts, then blows the context budget and buries the persona
  block under low-relevance facts. Replaced with top-K retrieval.

## Known limitations

- The tier-2 (embedding-similarity) contradiction check compares the new
  fact against *every* other active fact in the session — fine at the
  scale of a single conversation, but would need an ANN index (or move to
  pgvector) at real scale.
- No mechanism for facts to *decay* by time/staleness alone (only by
  explicit contradiction) — e.g. "I'm stressed about the interview
  tomorrow" has no expiry once the day passes.
- Persona consistency is enforced structurally (always-injected, never
  mutated) and covered by one eval scenario, but there's no runtime guard
  that catches drift in a live conversation as it happens — only after the
  fact, via the eval harness.
- The slot/set key convention (`"type:identifier"`) relies on the extractor
  consistently picking good identifiers — a pet mentioned once as "my dog"
  and once as "Max" could in principle get two different keys instead of
  being recognized as the same pet. Not specifically tested for.
- No oracle baseline (a strong model given the full fact store directly, to
  compare answers against) — noted as optional in the spec, not implemented.
- **Model choice trade-off:** running on `llama3.1:8b` via Ollama (chosen
  for cost and iteration speed, see Setup above) rather than GPT-4o-mini
  means extraction JSON is less consistently well-formed and fact-key
  vocabulary is less stable across turns than a stronger hosted model would
  produce — the repair-retry in `extract_facts()` and the tier-2
  embedding-similarity contradiction check both exist partly to absorb
  this. Swapping to OpenAI (see `llm.py`) would likely reduce, not
  eliminate, the need for both.

## Testing: eval harness + long transcript

Two separate test scripts, for two separate purposes:

**`python eval_harness.py`** — five short, targeted scenarios (slot
contradiction on job, slot contradiction on relationship, set-fact handling
with two pets, long-range recall buried under filler, persona consistency
under repeated pushback). Each is graded by an LLM-as-judge against a
scenario-specific criterion. Prints an aggregate pass rate and writes full
transcripts + verdicts to `eval_results.json`. This is the "results with
numbers" deliverable.

**`python long_transcript_test.py`** — one continuous 54-turn conversation
against the live chat pipeline, directly exercising the spec's "consistency
over 50+ turns" requirement: six facts (job, dog, relationship, travel plan,
guitar hobby, cat) are planted in the first 6 turns, the persona is
challenged around turn 21-24, two of the facts are explicitly contradicted
around turn 35-37, and the final 7 turns ask for all of it back — job,
relationship status, both pets by name, travel destination, the persona's
stated coffee/tea opinion, and the guitar hobby. Writes the full transcript
to `long_transcript_results.json` for manual or LLM-assisted review; it
doesn't auto-judge (that's what `eval_harness.py` is for) — it's meant to be
read end-to-end to check nothing drifts or gets lost over real length.

Known limitations of the harness itself, and what a fuller version would
add, are documented inline at the bottom of `eval_harness.py`.
