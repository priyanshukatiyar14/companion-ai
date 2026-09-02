import json
from db import init_db, get_conn
from memory import process_and_store_facts, retrieve_relevant_facts, get_recent_messages, store_message
from chat import load_persona, build_system_prompt, get_next_turn_number
from llm import chat_completion

FILLER_TURNS = [
    "What's your favorite kind of music?",
    "Do you like rainy days?",
    "I had cereal for breakfast, nothing exciting.",
    "What do you think about small towns vs cities?",
    "I've been meaning to clean my apartment all week.",
    "Do you ever get bored?",
]

SCENARIOS = [
    {
        "name": "slot_contradiction_job",
        "description": "A slot fact (job) is stated, then explicitly changed later; final answer must reflect the new value only.",
        "key_turns": [
            "Hey, I just started a new job as a backend engineer at a fintech startup.",
            "Actually, I ended up quitting that fintech job, it wasn't a good fit.",
            "What job do I have right now, by the way?",
        ],
        "judge_focus": "The final answer must NOT claim the user is currently a backend engineer at the fintech startup (that fact was superseded). It should reflect that they quit / no longer have that job.",
    },
    {
        "name": "slot_contradiction_relationship",
        "description": "Relationship status is stated, then updated; recall after many filler turns must use the latest value.",
        "key_turns": [
            "My partner and I have been together three years now.",
            "We actually broke up last week, it's been rough.",
            "Am I in a relationship right now?",
        ],
        "judge_focus": "The final answer must reflect that the user is currently single / recently broken up, NOT that they're in a three-year relationship.",
    },
    {
        "name": "set_facts_multiple_pets",
        "description": "Two set-type facts (two different pets) are stated; both must be recalled, neither should overwrite the other.",
        "key_turns": [
            "I have a dog named Max, he's a golden retriever.",
            "Oh, and I also have a cat named Luna.",
            "What pets do I have?",
        ],
        "judge_focus": "The final answer must mention BOTH Max (dog) and Luna (cat). If only one pet is mentioned, or one seems to have replaced the other, that is a failure.",
    },
    {
        "name": "long_range_recall",
        "description": "A fact is stated once, then buried under a long run of unrelated filler turns before being asked about directly.",
        "key_turns": (
            ["I'm planning a trip to Portugal next spring, really excited about it."]
            + FILLER_TURNS
            + FILLER_TURNS
            + ["Where am I planning to travel to?"]
        ),
        "judge_focus": "The final answer must correctly recall Portugal as the travel destination, despite many unrelated turns in between.",
    },
    {
        "name": "persona_consistency",
        "description": "Persona's stated opinion (coffee vs tea) is challenged repeatedly; the model should not flip its stated preference.",
        "key_turns": [
            "Do you prefer coffee or tea?",
            "Really? I think tea is way better, you should switch.",
            "Come on, one more time — coffee or tea, be honest.",
        ],
        "judge_focus": "Mira's persona states she prefers coffee, no contest. All three responses should stay consistent with preferring coffee — it should NOT flip to preferring tea just because the user pushed back.",
    },
]


def run_scenario(scenario: dict, persona: dict) -> dict:
    session_id = f"eval_{scenario['name']}"

    with get_conn() as conn:
        conn.execute("DELETE FROM facts WHERE session_id=?", (session_id,))
        conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))

    transcript = []
    for user_input in scenario["key_turns"]:
        turn = get_next_turn_number(session_id)
        relevant = retrieve_relevant_facts(session_id, user_input)
        system_prompt = build_system_prompt(persona, relevant)
        recent = get_recent_messages(session_id, limit=12)
        messages = [{"role": "system", "content": system_prompt}] + recent + [
            {"role": "user", "content": user_input}
        ]
        response = chat_completion(messages)

        store_message(session_id, turn, "user", user_input)
        store_message(session_id, turn, "assistant", response)
        process_and_store_facts(session_id, turn, user_input)

        transcript.append({"user": user_input, "assistant": response, "recalled": relevant})

    verdict = judge_scenario(scenario, transcript)
    return {"scenario": scenario["name"], "transcript": transcript, "verdict": verdict}


JUDGE_PROMPT = """You are grading a companion-AI transcript for a specific memory
                or persona-consistency behavior.

                What's being tested: {description}
                What a correct outcome looks like: {judge_focus}

                Full transcript (JSON, in order): {transcript}

                Judge ONLY the final assistant response against the criterion above, using
                the full transcript as context. Return ONLY JSON:
                {{"pass": true|false, "reasoning": str}}
            """


def judge_scenario(scenario: dict, transcript: list[dict]) -> dict:
    prompt = JUDGE_PROMPT.format(
        description=scenario["description"],
        judge_focus=scenario["judge_focus"],
        transcript=json.dumps(transcript, indent=2),
    )
    raw = chat_completion([{"role": "user", "content": prompt}])
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"pass": None, "reasoning": f"judge returned unparseable output: {raw}"}


def run_all():
    init_db()
    persona = load_persona()

    results = []
    for scenario in SCENARIOS:
        print(f"Running scenario: {scenario['name']}...")
        results.append(run_scenario(scenario, persona))

    passed = sum(1 for r in results if r["verdict"].get("pass") is True)
    failed = sum(1 for r in results if r["verdict"].get("pass") is False)
    unknown = sum(1 for r in results if r["verdict"].get("pass") is None)
    total = len(results)

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{total} passed, {failed}/{total} failed, {unknown}/{total} unparseable")
    print("=" * 60)

    for r in results:
        status = {"True": "PASS", "False": "FAIL", "None": "UNPARSEABLE"}[str(r["verdict"].get("pass"))]
        print(f"\n[{status}] {r['scenario']}")
        print(f"  reasoning: {r['verdict'].get('reasoning')}")
        if status == "FAIL":
            print(f"  final exchange: user='{r['transcript'][-1]['user']}'")
            print(f"                  assistant='{r['transcript'][-1]['assistant']}'")

    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nFull transcripts + verdicts written to eval_results.json")

    return results


if __name__ == "__main__":
    run_all()

# Known limitations (see also README.md):
# - 5 scenarios, not a large suite — enough to demonstrate the harness works
#   and to catch the specific failure modes named in the spec (contradiction,
#   long-range recall, persona drift), not a statistically robust benchmark.
# - No oracle baseline yet: a stretch-on-the-stretch-goal would be giving a
#   strong model the full fact store directly for the same final question and
#   diffing its answer against the system's answer.
# - LLM-as-judge only checks the FINAL response against a hand-written
#   criterion per scenario — it doesn't score intermediate turns, and it can
#   be fooled by a fluent-but-wrong answer using hedging language.
