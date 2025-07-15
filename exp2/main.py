import asyncio
from llama_index.core.workflow import Context
from agents import orchestrator

async def main():
    ctx = Context(orchestrator)
    await ctx.store.set("state", orchestrator.initial_state)

    # 1. Parse
    print(await orchestrator.run(user_msg="parse c1", ctx=ctx))
    # 2. Match
    print(await orchestrator.run(user_msg="match c1", ctx=ctx))
    # 3. Apply
    print(await orchestrator.run(user_msg="apply c1 j1", ctx=ctx))
    # 4. Slots
    print(await orchestrator.run(user_msg="slots c1_j1", ctx=ctx))
    # 5. Calendar
    print(await orchestrator.run(user_msg="calendar c1_j1 2025-08-01T10:00", ctx=ctx))

asyncio.run(main())
