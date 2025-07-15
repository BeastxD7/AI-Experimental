import asyncio
from AgentWorkflow import ctx, run_orchestrator
from llama_index.utils.workflow import draw_all_possible_flows
# Visualize the workflow
draw_all_possible_flows(ctx)

async def main():
    await run_orchestrator(
        ctx=ctx,
        user_msg=(
            "Write me a report on the company IonIdea "
            # "Briefly describe the history of the internet, including the development of the internet, the development of the web, "
            # "and the development of the internet in the 21st century."
        ),
    )


    
asyncio.run(main())