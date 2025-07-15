# chat_recruitment.py

import asyncio
from llama_index.core.workflow import Context
from agents import orchestrator   # your FunctionAgent orchestrator

async def chat_loop():
    ctx = Context(orchestrator)
    await ctx.store.set("state", orchestrator.initial_state)

    print("=== AI Recruitment Chatbot ===")
    print("Type 'exit' to quit.\n")

    while True:
        user_msg = input("You: ").strip()
        if not user_msg:
            continue
        if user_msg.lower() in ("exit", "quit", "bye"):
            print("Bot: Goodbye!")
            break

        # Pass the user message to the orchestrator
        try:
            bot_response = await orchestrator.run(user_msg=user_msg, ctx=ctx)
            print("Bot:", bot_response)
        except Exception as e:
            print("Bot: Sorry, something went wrong:", e)

if __name__ == "__main__":
    asyncio.run(chat_loop())
