from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.workflow import Context
import asyncio
from llama_index.llms.ollama import Ollama

llm = Ollama(
    model="llama3.1:8b",
    base_url="http://localhost:11434",
    request_timeout=600.0,  # optional: increase timeout
)



# Resume Parser Agent
resume_parser_agent = FunctionAgent(
    system_prompt="You are a resume parser. Print: 'Parsing resume.'",
    tools=[],
    initial_state={},
    llm=llm
)

# Job Description Parser Agent
job_desc_parser_agent = FunctionAgent(
    system_prompt="You are a job description parser. Print: 'Parsing job description.'",
    tools=[],
    initial_state={},
    llm=llm
)

# Matcher Agent
matcher_agent = FunctionAgent(
    system_prompt="You are a matcher. Print: 'Matching candidate to job.'",
    tools=[],
    initial_state={},
    llm=llm
)

# Application Handler Agent
application_handler_agent = FunctionAgent(
    system_prompt="You are an application handler. Print: 'Handling application.'",
    tools=[],
    initial_state={},
    llm=llm
)


# Agents as a Tool

async def call_resume_parser(ctx: Context, resume_text: str) -> str:
    print("Orchestrator: Calling Resume Parser Agent")
    await resume_parser_agent.run(user_msg=resume_text)
    return "Resume parsed."

async def call_job_desc_parser(ctx: Context, job_desc_text: str) -> str:
    print("Orchestrator: Calling Job Description Parser Agent")
    await job_desc_parser_agent.run(user_msg=job_desc_text)
    return "Job description parsed."

async def call_matcher(ctx: Context) -> str:
    print("Orchestrator: Calling Matcher Agent")
    await matcher_agent.run(user_msg="Match candidate to job")
    return "Matching done."

async def call_application_handler(ctx: Context, candidate_info: dict, job_info: dict) -> str:
    print("Orchestrator: Calling Application Handler Agent")
    await application_handler_agent.run(user_msg="Handle application")
    return "Application handled."


# Orchestrator setup

orchestrator = FunctionAgent(
    system_prompt="You are the orchestrator for a hiring workflow. Call the appropriate sub-agent for each step.",
    tools=[call_resume_parser, call_job_desc_parser, call_matcher, call_application_handler],
    initial_state={},
    llm=llm  # Use your locally running Ollama Llama 3.1 8B instance here
)

# response =  orchestrator.run(user_msg="Start hiring workflow")
# print(response)

async def main():
    ctx = Context(workflow=orchestrator)  # Pass the orchestrator as the workflow
    await orchestrator.run(
        ctx=ctx,
        user_msg="Start hiring workflow",
        verbose=True
    )

asyncio.run(main())



    