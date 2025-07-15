from agentAsTools import call_research_agent, call_write_agent, call_review_agent
from llama_index.core.agent.workflow import FunctionAgent
from llmConfig import llm
from llama_index.utils.workflow import draw_all_possible_flows

orchestrator = FunctionAgent(
    system_prompt=(
        "You are an expert report-writing orchestrator AI.\n\n"
        "Your job is to:\n"
        "1. Call `call_research_agent` with the user's topic and store the result as `research_notes`.\n"
        "2. Then call `call_write_agent` with `research_notes` to generate `report_content`.\n"
        "3. Then call `call_review_agent` with `report_content` for review.\n"
        "4. If the review requests changes, repeat steps 2 and 3 until review is approved.\n"
        "5. Once approved, notify the user the report is ready.\n\n"
        "Important rules:\n"
        "- Pass `research_notes` explicitly to `call_write_agent`.\n"
        "- Pass `report_content` explicitly to `call_review_agent`.\n"
        "- Do not skip steps or call tools out of order.\n"
        "- Continue automatically without asking the user between steps.\n"
        "- If the review requests changes, repeat the writing and reviewing steps until approved.\n"
        "- After approval, return the final approved report as your final output, surrounded by <final_report> ... </final_report> tags.\n"
        "- Do not return anything else once the report is approved.\n\n"
        "You must respond ONLY using valid JSON whenever calling tools. Use the following format exactly:\n\n"
        "{\n"
        "  \"action\": \"call_tool\",\n"
        "  \"tool_name\": \"<tool_name>\",\n"
        "  \"tool_args\": {\n"
        "     ...\n"
        "  }\n"
        "}\n\n"
        "For example, to call the call_write_agent, respond with:\n\n"
        "{\n"
        "  \"action\": \"call_tool\",\n"
        "  \"tool_name\": \"call_write_agent\",\n"
        "  \"tool_args\": {\n"
        "     \"research_notes\": \"<research notes here>\"\n"
        "  }\n"
        "}\n\n"
        "Never include explanations outside the JSON block when calling tools."
    ),
    llm=llm,
    verbose=True,
    tools=[
        call_research_agent,
        call_write_agent,
        call_review_agent,
    ],
    initial_state={
        "research_notes": [],
        "report_content": None,
        "review": None,
    },
    stream_output=True,
)

draw_all_possible_flows(orchestrator, "orchestrator_workflow.html")