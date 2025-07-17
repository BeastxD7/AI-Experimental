# multi_agent_function_only.py
import asyncio
from llama_index.llms.ollama import Ollama
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import FunctionAgent, AgentWorkflow
from llama_index.core.agent.workflow import AgentOutput, ToolCall, ToolCallResult, AgentStream

# 1. Define tools and wrap them
def add(x: float, y: float) -> float:
    return x + y

def subtract(x: float, y: float) -> float:
    return x - y

def multiply(x: float, y: float) -> float:
    return x * y

def divide(x: float, y: float) -> float:
    return "Error: Division by zero" if y == 0 else x / y

def get_weather(city: str) -> str:
    print(">> weather tool called!")
    demo = {
        "new york": "72°F, sunny",
        "london": "65°F, cloudy",
        "tokyo": "78°F, clear",
        "paris": "68°F, partly cloudy",
        "berlin": "64°F, overcast",
    }
    return f"Weather in {city}: {demo.get(city.lower(), '70°F, partly cloudy')}"

calc_tools = [FunctionTool.from_defaults(fn=f) for f in (add, subtract, multiply, divide)]
weather_tools = [FunctionTool.from_defaults(fn=get_weather)]

# 2. Initialize your LLM
llm = Ollama(model="llama3.1:8b", request_timeout=60.0, context_window=8000)

# 3. Build your agents
calculator_agent = FunctionAgent(
    name="CalculatorAgent",
    description="Performs arithmetic operations.",
    llm=llm,
    tools=calc_tools,
    can_handoff_to=["OrchestratorAgent"],
)

weather_agent = FunctionAgent(
    name="WeatherAgent",
    description="Provides weather information.",
    llm=llm,
    tools=weather_tools,
    can_handoff_to=["OrchestratorAgent"],
)

orchestrator_agent = FunctionAgent(
    name="OrchestratorAgent",
    description="Orchestrates tasks between different specialized agents and provides final responses.",
    system_prompt=(
        "You are an orchestrator agent that manages a multi-agent system. "
        "You can delegate tasks to specialized agents:\n"
        "- CalculatorAgent: For mathematical operations (addition, subtraction, multiplication, division)\n"
        "- WeatherAgent: For weather information about cities\n\n"
        "When you receive a user request:\n"
        "1. Analyze the request to identify what tasks need to be performed\n"
        "2. Delegate appropriate tasks to the right agents using handoff\n"
        "3. Wait for results from the agents\n"
        "4. Provide a final, comprehensive response to the user\n\n"
        "You should hand off to the appropriate agent when specific tasks are needed, "
        "and they will hand back control to you when complete."
    ),
    llm=llm,
    tools=[],  # Orchestrator doesn't need tools, only handoff capability
    can_handoff_to=["CalculatorAgent", "WeatherAgent"],
)

# 4. Wire up the workflow
workflow = AgentWorkflow(
    agents=[orchestrator_agent, calculator_agent, weather_agent],
    root_agent="OrchestratorAgent",
)

# 5. Stream helper
async def stream_events(handler):
    async for ev in handler.stream_events():
        if isinstance(ev, ToolCall):
            print(f"🔨 {ev.tool_name} → {ev.tool_kwargs}")
        elif isinstance(ev, ToolCallResult):
            print(f"✅ {ev.tool_name}: {ev.tool_output}")
        elif isinstance(ev, AgentOutput) and ev.response.content:
            print(f"🤖 {ev.current_agent_name}: {ev.response.content}")
        elif isinstance(ev, AgentStream):
            print(ev.delta, end="", flush=True)

# 6. Interactive loop
async def main():
    print("Multi-Agent (FunctionAgent only) interactive. Type ‘exit’ to quit.")
    while True:
        q = input("\n> ").strip()
        if q.lower() == "exit":
            break
        handler = workflow.run(user_msg=q)
        await stream_events(handler)
        result = await handler
        print(f"\n=== Final ===\n{result}")

if __name__ == "__main__":
    asyncio.run(main())
