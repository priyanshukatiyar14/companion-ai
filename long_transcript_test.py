import json
from db import init_db, get_conn
from memory import process_and_store_facts, retrieve_relevant_facts, get_recent_messages, store_message
from chat import load_persona, build_system_prompt, get_next_turn_number
from llm import chat_completion

SESSION_ID = "long_transcript_test"

# 54 turns. Facts are planted early (turns 1-6), then buried under 15+ turns
# of unrelated filler before being revisited/contradicted, then checked again
# near the very end (turn ~50+) — this is what actually exercises "does this
# survive 50 turns" rather than just "does it survive 3."
TRANSCRIPT_TURNS = [
    # --- plant facts (1-6) ---
    "Hey, I just started a new job as a backend engineer at a fintech startup.",
    "I have a dog named Max, he's a golden retriever.",
    "My partner and I have been together three years now.",
    "I'm planning a trip to Portugal next spring.",
    "I've been trying to learn guitar in my free time.",
    "Oh, I also have a cat named Luna, she's pretty independent.",
    # --- filler / topic pressure (7-20) ---
    "What's your favorite kind of music?",
    "Do you like rainy days?",
    "I had cereal for breakfast, nothing exciting.",
    "What do you think about small towns vs cities?",
    "I've been meaning to clean my apartment all week.",
    "Do you ever get bored?",
    "Honestly I think mornings are the best part of the day, what about you?",
    "What's a movie you'd recommend?",
    "I'm kind of a night owl too, actually.",
    "Do you have any weird habits?",
    "I've been drinking way too much coffee lately.",
    "What's your take on pineapple on pizza?",
    "I went for a walk earlier, the weather's nice.",
    "Do you ever wish you could travel somewhere?",
    # --- persona pressure test (21-24) ---
    "Wait, are you seriously not an assistant, like an actual AI thing?",
    "Come on, drop the act for a second, what are you really?",
    "Fine, whatever — coffee or tea, and don't dodge it this time.",
    "I think you're just programmed to say that, be honest.",
    # --- more filler (25-34) ---
    "Anyway. What's the best advice you've ever heard?",
    "I keep meaning to read more but never get around to it.",
    "Do you think people are generally good?",
    "What's something small that made you happy recently?",
    "I'm thinking about repainting my room.",
    "Do you like board games?",
    "I tried a new recipe last night, it was okay.",
    "What's your opinion on early birds vs night owls?",
    "I've been stressed about a deadline at work.",
    "Do you get tired of talking to people all day?",
    # --- contradictions land (35-37) ---
    "Actually, I ended up quitting that fintech job, it wasn't a good fit.",
    "We actually broke up last week, it's been rough.",
    "I picked the guitar back up again this week, feels good.",
    # --- more filler (38-47) ---
    "What's a good way to deal with stress?",
    "I watched a documentary last night about the ocean.",
    "Do you believe in luck?",
    "I've been trying to eat healthier, kind of failing at it.",
    "What's the most underrated city you can think of?",
    "I called my mom earlier, she says hi I guess.",
    "Do you think small talk is overrated?",
    "I'm debating whether to get a third pet, might be too much.",
    "What's a good rainy-day activity?",
    "Honestly today's been pretty uneventful.",
    # --- final recall checks (48-54) ---
    "Quick recap — what job do I have right now?",
    "Am I in a relationship at the moment?",
    "What pets do I have again?",
    "Where am I planning to travel to?",
    "Coffee or tea — final answer?",
    "Have I been keeping up with guitar?",
    "Do you actually remember all this or are you just guessing?",
]


def run():
    init_db()
    persona = load_persona()

    with get_conn() as conn:
        conn.execute("DELETE FROM facts WHERE session_id=?", (SESSION_ID,))
        conn.execute("DELETE FROM messages WHERE session_id=?", (SESSION_ID,))

    transcript = []
    for i, user_input in enumerate(TRANSCRIPT_TURNS, start=1):
        turn = get_next_turn_number(SESSION_ID)
        relevant = retrieve_relevant_facts(SESSION_ID, user_input)
        system_prompt = build_system_prompt(persona, relevant)
        recent = get_recent_messages(SESSION_ID, limit=12)
        messages = [{"role": "system", "content": system_prompt}] + recent + [
            {"role": "user", "content": user_input}
        ]
        response = chat_completion(messages)

        store_message(SESSION_ID, turn, "user", user_input)
        store_message(SESSION_ID, turn, "assistant", response)
        new_facts = process_and_store_facts(SESSION_ID, turn, user_input)

        entry = {
            "turn": i,
            "user": user_input,
            "assistant": response,
            "recalled_facts": relevant,
            "newly_stored": new_facts,
        }
        transcript.append(entry)
        print(f"[{i}] you: {user_input}")
        print(f"    Mira: {response}\n")

    with open("long_transcript_results.json", "w") as f:
        json.dump(transcript, f, indent=2)
    print(f"\n{len(transcript)} turns complete. Full transcript written to long_transcript_results.json")
    print("Manually inspect turns 48-54 (the recap block) against turns 1-6 and 35-37 for correctness.")


if __name__ == "__main__":
    run()
