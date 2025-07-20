import asyncio
import os
from llama_index.core.agent.workflow import (
    AgentWorkflow, FunctionAgent, AgentInput, ToolCall, ToolCallResult, AgentOutput
)
from llama_index.llms.groq import Groq
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

# 1. Setup LLM
llm = Groq(model="llama-3.1-8b-instant", api_key=API_KEY, temperature=0.1)

# 2. Define FunctionTools

def add(a: float, b: float) -> float:        return float(a + b)
def subtract(a: float, b: float) -> float:   return float(a - b)
def multiply(a: float, b: float) -> float:   return float(a * b)
def divide(a: float, b: float) -> float:
    if b == 0: raise ValueError("Cannot divide by zero")
    return float(a / b)
def get_weather(city: str) -> str:
    if not city or not isinstance(city, str):
        raise ValueError("City must be a non-empty string")
    return f"The current weather in {city} is sunny with a temperature of 25°C."
def synthesize(results: str, original: str) -> str:
    return f"Results for your question '{original}': {results}"

add_tool = FunctionTool.from_defaults(fn=add)
sub_tool = FunctionTool.from_defaults(fn=subtract)
mul_tool = FunctionTool.from_defaults(fn=multiply)
div_tool = FunctionTool.from_defaults(fn=divide)
weather_tool = FunctionTool.from_defaults(fn=get_weather)
synth_tool = FunctionTool.from_defaults(fn=synthesize, name="synthesize")

# 3. Define agents

CalculatorAgent = FunctionAgent(
    name="CalculatorAgent",
    description="Handles arithmetic using add, subtract, multiply, and divide.",
    system_prompt=(
        "You are CalculatorAgent. For any math question, pick the correct tool "
        "and generate a plain English sentence response like 'The result is ...'. "
        "If the question is not math, reply: 'I cannot answer this, please ask the Orchestrator.'"
    ),
    tools=[add_tool, sub_tool, mul_tool, div_tool],
    llm=llm,
    verbose=True,
)

WeatherAgent = FunctionAgent(
    name="WeatherAgent",
    description="Handles weather queries using get_weather.",
    system_prompt=(
        "You are WeatherAgent. For weather queries, use get_weather(city). "
        "If the question is not about weather, reply: 'I cannot answer this, please ask the Orchestrator.'"
    ),
    tools=[weather_tool],
    llm=llm,
    verbose=True,
)

OrchestratorAgent = FunctionAgent(
    name="OrchestratorAgent",
    description=(
        "Receives all user queries. Routes math queries to CalculatorAgent and weather queries to WeatherAgent. "
        "If the user asks for both, call both, then call the synthesize tool to summarize the results."
    ),
    system_prompt=(
        "You are OrchestratorAgent. Parse the user message. "
        "If the message includes math (calculations with numbers), call CalculatorAgent. "
        "If it includes weather, call WeatherAgent. "
        "If both, call each in turn and collect their responses. "
        "Finally, call the synthesize tool with 'results' as '[math_answer] [weather_answer]' and original as the original user message."
        "If unsure or unsupported, respond: 'I can only answer math or weather questions.'"
    ),
    tools=[synth_tool],
    llm=llm,
    verbose=True,
)

# 4. Set up the workflow
workflow = AgentWorkflow(
    agents=[OrchestratorAgent, CalculatorAgent, WeatherAgent],
    root_agent="OrchestratorAgent",
    initial_state={
        "math_result": None,
        "weather_result": None,
        "original_query": ""
    }
)

# 5. Main event loop
async def main():
    print("\n=== LlamaIndex Orchestrator Pattern Demo (Challenge 6) ===")
    print("Type 'exit' to quit.\n")
    ctx = Context(workflow)
    while True:
        user_query = input("User: ").strip()
        if not user_query or user_query.lower() in ("exit", "quit"): break
        await ctx.store.set("original_query", user_query)
        handler = workflow.run(user_msg=user_query, ctx=ctx)
        results, final_answer = [], None
        async for ev in handler.stream_events():
            if isinstance(ev, ToolCallResult):
                out = getattr(ev.tool_output, "raw_output", ev.tool_output)
                if ev.tool_name in ["add", "subtract", "multiply", "divide"]:
                    results.append(f"Calculation: {out}")
                    await ctx.store.set("math_result", out)
                elif ev.tool_name == "get_weather":
                    results.append(f"Weather: {out}")
                    await ctx.store.set("weather_result", out)
                elif ev.tool_name == "synthesize":
                    final_answer = out
            elif isinstance(ev, AgentOutput):
                if ev.response.content and ev.response.content.strip():
                    print(f"{ev.current_agent_name}: {ev.response.content.strip()}")
        if final_answer:
            print("\nOrchestrator (final):", final_answer, "\n")
        elif results:
            print("\nPartial results:", " ".join(results), "\n")
        else:
            print("Sorry, I couldn't produce an answer.\n")

if __name__ == "__main__":
    asyncio.run(main())
