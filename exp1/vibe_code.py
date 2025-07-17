from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.ollama import Ollama

# 1. Define tool functions
def add(a: int, b: int) -> int:
    return a + b

def subtract(a: int, b: int) -> int:
    return a - b

def multiply(a: int, b: int) -> int:
    return a * b

def divide(a: int, b: int) -> float:
    if b == 0:
        return float('inf')
    return a / b

def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny and 25°C."

# 2. Wrap tools
add_tool = FunctionTool.from_defaults(fn=add)
sub_tool = FunctionTool.from_defaults(fn=subtract)
mul_tool = FunctionTool.from_defaults(fn=multiply)
div_tool = FunctionTool.from_defaults(fn=divide)
weather_tool = FunctionTool.from_defaults(fn=get_weather)

# 3. Create agents
llm = Ollama(model="llama3.1:8b", base_url="http://localhost:11434")

math_agent = FunctionAgent(
    name="MathAgent",
    description="Handles addition, subtraction, multiplication, and division.",
    system_prompt="You are a precise calculator. Use your tools for math and return clean numerical results only.",
    tools=[add_tool, sub_tool, mul_tool, div_tool],
    llm=llm,
)

weather_agent = FunctionAgent(
    name="WeatherAgent",
    description="Handles weather queries for a city.",
    system_prompt="You provide current weather info for requested cities using the weather tool.",
    tools=[weather_tool],
    llm=llm,
)

# 4. Orchestrator agent (no tools, just routing logic in prompt)
orchestrator = FunctionAgent(
    name="OrchestratorAgent",
    description="Routes requests to the right agent.",
    system_prompt=(
        "You are an orchestrator. "
        "If the user wants to do math (add, subtract, multiply, divide), call MathAgent. "
        "If the user wants weather info, call WeatherAgent."
    ),
    tools=[],  # No tools here!
    llm=llm,
)

# 5. Create workflow
workflow = AgentWorkflow(
    agents=[math_agent, weather_agent, orchestrator],
    root_agent="OrchestratorAgent",
    verbose=True,
)

# 6. Run the workflow
import asyncio

async def main():
    print("Ask: 'Add 2 and 3', 'What is 10 divided by 2?', or 'Weather in Paris'")
    user_query = input("User: ")
    response = await workflow.run(user_msg=user_query)
    print("Agent:", response)

if __name__ == "__main__":
    asyncio.run(main())