from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.ollama import Ollama
import asyncio

# 1. Shared LLM
llm = Ollama(
    model="llama3.1:8b",
    base_url="http://localhost:11434",
    request_timeout=600.0,
)

# 2. Tools
def add(a: float, b: float) -> float:
    result = a + b
    print(f"🔧 add({a}, {b}) = {result}")
    return result

def subtract(a: float, b: float) -> float:
    result = a - b
    print(f"🔧 subtract({a}, {b}) = {result}")
    return result

def multiply(a: float, b: float) -> float:
    result = a * b
    print(f"🔧 multiply({a}, {b}) = {result}")
    return result

def divide(a: float, b: float) -> float:
    result = a / b
    print(f"🔧 divide({a}, {b}) = {result}")
    return result

def get_weather(city: str) -> str:
    result = f"The current weather in {city} is sunny, 27°C ☀️."
    print(f"🔧 get_weather({city}) = {result}")
    return result

# Wrap tools
add_tool = FunctionTool.from_defaults(fn=add)
sub_tool = FunctionTool.from_defaults(fn=subtract)
mul_tool = FunctionTool.from_defaults(fn=multiply)
div_tool = FunctionTool.from_defaults(fn=divide)
weather_tool = FunctionTool.from_defaults(fn=get_weather)

# 3. Agents

# CalculatorAgent replacing add/sub agents
calc_agent = FunctionAgent(
    name="CalculatorAgent",
    description="Handles addition, subtraction, multiplication, and division operations.",
    system_prompt="You are a precise calculator. Use your tools for math and return clean numerical results only.",
    tools=[add_tool, sub_tool, mul_tool, div_tool],
    llm=llm
)

weather_agent = FunctionAgent(
    name="WeatherAgent",
    description="Handles weather queries and provides current weather information for a city.",
    system_prompt="You provide current weather info for requested cities using the weather tool.",
    tools=[weather_tool],
    llm=llm
)

# 4. Workflow
workflow = AgentWorkflow(
    agents=[calc_agent, weather_agent],
    root_agent="WeatherAgent"  # or "CalculatorAgent" depending on your primary routing preference
)

# 5. Run
async def main():
    response = await workflow.run(
        user_msg="What's the weather in Tokyo today? Then, what is 10 - 5? Then add 3. Also multiply 6 by 7 and divide by 2."
    )
    print("\n🚀 [Final Response]:\n", response)

if __name__ == "__main__":
    asyncio.run(main())
