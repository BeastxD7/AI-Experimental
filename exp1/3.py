import logging, sys
import llama_index.core
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
from llama_index.llms.ollama import Ollama
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler

# Silence noisy HTTP logs
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Setup debug handler for structured tracing
debug_handler = LlamaDebugHandler()
debug_handler.print_trace = True  # <-- Fix for your version

callback_manager = CallbackManager([debug_handler])

# Math functions with print tracing
def multiply(a, b):
    a = float(a)
    b = float(b)
    result = a * b
    print(f"🔧 multiply({a}, {b}) = {result}")
    return result

def divide(a, b):
    a = float(a)
    b = float(b)
    result = a / b
    print(f"🔧 divide({a}, {b}) = {result}")
    return result

def add(a, b):
    a = float(a)
    b = float(b)
    result = a + b
    print(f"🔧 add({a}, {b}) = {result}")
    return result

def subtract(a, b):
    a = float(a)
    b = float(b)
    result = a - b
    print(f"🔧 subtract({a}, {b}) = {result}")
    return result

def greet(name):
    result = f"Hey {name}! Hope you’re having a rad day! 🌟"
    print(f"🔧 greet({name}) = {result}")
    return result

# Wrap tools
add_tool = FunctionTool.from_defaults(fn=add)
sub_tool = FunctionTool.from_defaults(fn=subtract)
mul_tool = FunctionTool.from_defaults(fn=multiply)
div_tool = FunctionTool.from_defaults(fn=divide)
greet_tool = FunctionTool.from_defaults(fn=greet)

# LLM
llm = Ollama(model="llama3.1:8b", base_url="http://localhost:11434", request_timeout=600.0)

# Agents
calc_agent = FunctionAgent(
    name="CalculatorAgent",
    description="Handles math operations precisely.",
    system_prompt="Use tools for math. Do not explain steps, return results only.",
    tools=[add_tool, sub_tool, mul_tool, div_tool],
    llm=llm,
    callback_manager=callback_manager
)

helper_agent = FunctionAgent(
    name="HelperAgent",
    description="Handles greetings.",
    system_prompt="Use tools to greet, and return greeting directly.",
    tools=[greet_tool],
    llm=llm,
    callback_manager=callback_manager
)

# Workflow
workflow = AgentWorkflow(
    agents=[calc_agent, helper_agent],
    root_agent="CalculatorAgent",
)

# Query
user_query = "Multiply 7 by 6, then divide the result by 3. After that, say hi to Alice."

import asyncio
async def main():
    response = await workflow.run(user_msg=user_query)
    print("\n🚀 Final Response:", response)

if __name__ == "__main__":
    asyncio.run(main())
