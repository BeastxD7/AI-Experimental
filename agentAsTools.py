import re
from llama_index.core.workflow import Context
from agents import research_agent, write_agent, review_agent


async def call_research_agent(ctx: Context, prompt: str) -> str:
    """Useful for recording research notes based on a specific prompt."""
    result = await research_agent.run(
        user_msg=f"Write some notes about the following: {prompt}"
    )

    state = await ctx.store.get("state")
    state["research_notes"].append(str(result))
    await ctx.store.set("state", state)

    return str(result)


async def call_write_agent(ctx: Context) -> str:
    """Useful for writing a report based on the research notes or revising the report based on feedback."""
    state = await ctx.store.get("state")
    notes = state.get("research_notes", None)
    if not notes:
        return "No research notes to write from."

    user_msg = f"Write a markdown report from the following notes. Be sure to output the report in the following format: ...:\n\n"

    # Add the feedback to the user message if it exists
    feedback = state.get("review", None)
    if feedback:
        user_msg += f"{feedback}\n\n"

    # Add the research notes to the user message
    notes = "\n\n".join(notes)
    user_msg += f"{notes}\n\n"

    # Run the write agent
    result = await write_agent.run(user_msg=user_msg)
    
    # Extract content between ... tags, or use full result if tags not found
    match = re.search(r"\.\.\.(.*)\.\.\.", str(result), re.DOTALL)
    if match:
        report = match.group(1).strip()
    else:
        # Fallback to full result if no tags found
        report = str(result)
    
    state["report_content"] = str(report)
    await ctx.store.set("state", state)

    return str(report)


async def call_review_agent(ctx: Context) -> str:
    """Useful for reviewing the report and providing feedback."""
    state = await ctx.store.get("state")
    report = state.get("report_content", None)
    if not report:
        return "No report content to review."

    result = await review_agent.run(
        user_msg=f"Review the following report: {report}"
    )
    state["review"] = result
    await ctx.store.set("state", state)

    return result