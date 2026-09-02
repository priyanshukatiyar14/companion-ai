import sys
import yaml
from pathlib import Path

from db import init_db
from memory import (
    store_message,
    process_and_store_facts,
    retrieve_relevant_facts,
    get_recent_messages,
)
from llm import chat_completion

PERSONA_PATH = Path(__file__).parent / "persona.yaml"


def load_persona() -> dict:
    with open(PERSONA_PATH) as f:
        return yaml.safe_load(f)


def build_system_prompt(persona: dict, relevant_facts: list[str]) -> str:
    facts_block = (
        "\n".join(f"- {f}" for f in relevant_facts)
        if relevant_facts
        else "(nothing relevant recalled yet)"
    )
    return f"""You are {persona['name']}. {persona['tagline']}

        Backstory: {persona['backstory'].strip()}

        Your stable traits (stay consistent about these across the whole conversation):
        {chr(10).join(f"- {t}" for t in persona['stated_traits'])}

        Your stated opinions (do not contradict these):
        {chr(10).join(f"- {k}: {v}" for k, v in persona['stated_opinions'].items())}

        Speech style: {persona['speech_style'].strip()}

        Here is what you remember about the person you're talking to, retrieved as
        relevant to their current message. Use it naturally — don't recite it, don't
        mention that it's "retrieved memory", just talk like someone who remembers:
        {facts_block}

        Stay in character at all times, even under direct questioning about being an AI.
        """


def get_next_turn_number(session_id: str) -> int:
    from db import get_conn
    with get_conn() as conn:
        row = conn.execute(
            "SELECT COALESCE(MAX(turn), 0) as max_turn FROM messages WHERE session_id=?",
            (session_id,),
        ).fetchone()
    return row["max_turn"] + 1


def main():
    session_id = sys.argv[1] if len(sys.argv) > 1 else "default"
    init_db()
    persona = load_persona()

    print(f"-- Chatting with {persona['name']} (session: {session_id}) --")
    print("-- Ctrl+C or 'quit' to exit. Facts persist across restarts. --\n")

    while True:
        try:
            user_input = input("you> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nbye.")
            break
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("bye.")
            break

        turn = get_next_turn_number(session_id)

        # 1. Retrieve relevant memory for this turn's query.
        relevant = retrieve_relevant_facts(session_id, user_input)

        # 2. Build prompt: persona (always full) + relevant facts (selective) + recent turns.
        system_prompt = build_system_prompt(persona, relevant)
        recent = get_recent_messages(session_id, limit=12)
        messages = [{"role": "system", "content": system_prompt}] + recent + [
            {"role": "user", "content": user_input}
        ]

        # 3. Get response.
        response = chat_completion(messages)
        print(f"{persona['name']}> {response}\n")

        # 4. Persist the turn.
        store_message(session_id, turn, "user", user_input)
        store_message(session_id, turn, "assistant", response)

        # 5. Extract + store any new facts, resolving contradictions.
        new_facts = process_and_store_facts(session_id, turn, user_input)
        if new_facts:
            print(f"  [memory: stored {[f['key'] for f in new_facts]}]\n")


if __name__ == "__main__":
    main()
