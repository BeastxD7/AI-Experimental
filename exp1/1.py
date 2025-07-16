from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.ollama import Ollama
import asyncio

# 1. Initialize shared LLM
llm = Ollama(
    model="llama3.1:8b",
    base_url="http://localhost:11434",
    request_timeout=600.0,
)

# Define tools
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

def subtract(a: int, b: int) -> int:
    """Subtract two numbers."""
    return a - b

add_tool = FunctionTool.from_defaults(fn=add)
sub_tool = FunctionTool.from_defaults(fn=subtract)

# Create agent configs
add_agent = FunctionAgent(
    name="AdderAgent",
    description="Does addition",
    system_prompt="You are specialized in addition.",
    tools=[add_tool],
    llm=llm
)

sub_agent = FunctionAgent(
    name="SubtractorAgent",
    description="Does subtraction",
    system_prompt="You are specialized in subtraction.",
    tools=[sub_tool],
    llm=llm
)

# Orchestrate them in a workflow
workflow = AgentWorkflow(
    agents=[add_agent, sub_agent],
    root_agent="AdderAgent"
)

# Run the multi-agent calculator (async)
async def main():
    response = await workflow.run(user_msg="What is 10 + 5? Then subtract 3.")
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
