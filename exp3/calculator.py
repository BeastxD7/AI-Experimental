import asyncio
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import FunctionAgent, AgentWorkflow
from llama_index.llms.ollama import Ollama

# Connect to local Ollama 
llm = Ollama(model="llama3.1:8b", request_timeout=120.0, context_window=8000)



def add(n1: float, n2: float) -> float:
    """Add two numbers."""
    print(f"Adding {n1} and {n2} using the Add tool...")
    return n1 + n2

def subtract(n1: float, n2: float) -> float:
    """Subtract two numbers."""
    print(f"Subtracting {n1} from {n2} using the Subtract tool...")
    return n1 - n2

def multiply(n1: float, n2: float) -> float:
    """Multiply two numbers."""
    print(f"Multiplying {n1} and {n2} using the Multiply tool...")
    return n1 * n2

def divide(n1: float, n2: float) -> float:
    """Divide two numbers."""
    print(f"Dividing {n1} by {n2} using the Divide tool...")
    if n2 == 0:
        return "Division by zero error."
    return n1 / n2

calculator_tools = [
    FunctionTool.from_defaults(add),
    FunctionTool.from_defaults(subtract),
    FunctionTool.from_defaults(multiply),
    FunctionTool.from_defaults(divide)
]

# 3. AGENTS

calculator_agent = FunctionAgent(
    name="CalculatorAgent",
    description="Performs basic arithmetic operations.",
    system_prompt=(
        "You are a calculator agent. "
        "NEVER answer from your own knowledge or training. "
        "ALWAYS use the calculator tools (Add, Subtract, Multiply, Divide) provided to answer ANY math question. "
        "If the tool cannot answer, respond ONLY with: 'I am here to assist you with simple mathematics questions.' "
        "You MUST NOT add information, explanations, or make up any data. "
        "The output from the tool is the only valid answer."
    ),
    llm=llm,
    tools=calculator_tools,
    verbose=True
)



# 4. WORKFLOW ORCHESTRATOR
agent_workflow = AgentWorkflow(
    agents=[ calculator_agent],
    root_agent=calculator_agent.name,  # Start with weather agent, can be changed as needed
    initial_state={}
)

async def main():
    print("Welcome to the multi-agent system (Weather + Calculator) using local Ollama model.")
    user_prompt = input("Enter your query: ")
    response = await agent_workflow.run(user_msg=user_prompt)
    print("\nResponse:", response)

if __name__ == "__main__":
    asyncio.run(main())
