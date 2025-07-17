import json
from typing import List, Optional

# LlamaIndex core components
from llama_index.core.tools import FunctionTool, ToolMetadata
from llama_index.core.agent import AgentRunner, ReActAgent

# Import Ollama
from llama_index.llms.ollama import Ollama

# --- 1. Define the Tools for the Agents ---
# These are the basic functions that our agents will be able to call.

# --- Calculator Tools ---
def add(a: float, b: float) -> float:
    """Adds two numbers."""
    print(f"TOOL CALLED: add(a={a}, b={b})")
    return a + b

def subtract(a: float, b: float) -> float:
    """Subtracts the second number from the first."""
    print(f"TOOL CALLED: subtract(a={a}, b={b})")
    return a - b

def multiply(a: float, b: float) -> float:
    """Multiplies two numbers."""
    print(f"TOOL CALLED: multiply(a={a}, b={b})")
    return a * b

def divide(a: float, b: float) -> float:
    """Divides the first number by the second. Returns an error message if division by zero."""
    print(f"TOOL CALLED: divide(a={a}, b={b})")
    if b == 0:
        return "Error: Cannot divide by zero."
    return a / b

# --- Weather Tool ---
def get_weather(city: str) -> str:
    """Gets the current weather for a specified city."""
    print(f"TOOL CALLED: get_weather(city='{city}')")
    # Hardcoded responses as requested
    if city.lower() == "new york":
        return "The temperature in New York is 30 degrees Celsius."
    elif city.lower() == "tokyo":
        return "It's currently 25 degrees Celsius and sunny in Tokyo."
    else:
        return f"Sorry, I don't have weather information for {city}."

# --- 2. Convert Functions into LlamaIndex Tools ---
# We wrap our Python functions in FunctionTool to make them usable by the agents.
# This approach is current with the latest llama-index practices.

# Calculator tools
add_tool = FunctionTool.from_defaults(fn=add)
subtract_tool = FunctionTool.from_defaults(fn=subtract)
multiply_tool = FunctionTool.from_defaults(fn=multiply)
divide_tool = FunctionTool.from_defaults(fn=divide)

# Weather tool
weather_tool = FunctionTool.from_defaults(fn=get_weather)

# --- 3. Define the LLM (Using Ollama as requested) ---
# This configuration connects to your local Ollama instance.
# Ensure Ollama is running and the specified model is available.
print("Initializing Ollama LLM...")
llm = Ollama(model="llama3.1:8b", base_url="http://localhost:11434", request_timeout=600.0)
print("Ollama LLM initialized.")


# --- 4. Create the Specialized Agents ---
# Each agent gets a specific set of tools. This makes them experts in their domain.
# The use of AgentRunner with a ReActAgent is the modern approach in llama-index.

# Calculator Agent: Specializes in mathematical operations.
print("Creating Calculator Agent...")
calculator_agent = AgentRunner(
    ReActAgent.from_tools(
        tools=[add_tool, subtract_tool, multiply_tool, divide_tool],
        llm=llm,
        verbose=True,
        # The context/system prompt tells the agent its role and how to behave.
        context="""
        You are a specialized calculator agent.
        Your sole purpose is to perform mathematical calculations based on the tools provided.
        You must use the provided tools to answer the query. Do not rely on your internal knowledge for math.
        Given a mathematical query, you will break it down into steps, use your tools to solve each step, and then present the final answer.
        """
    )
)
print("Calculator Agent created.")

# Weather Agent: Specializes in retrieving weather information.
print("Creating Weather Agent...")
weather_agent = AgentRunner(
    ReActAgent.from_tools(
        tools=[weather_tool],
        llm=llm,
        verbose=True,
        context="""
        You are a specialized weather agent.
        Your only job is to provide weather information for a given city using the available tool.
        Do not answer any questions that are not related to weather.
        """
    )
)
print("Weather Agent created.")

# --- 5. Create the Orchestrator (Top-Level Agent) ---
# This agent doesn't perform the tasks itself. Instead, its "tools" are the other agents.
# It plans and delegates tasks to the appropriate specialized agent.

# We create new tools where the function is a call to another agent's `chat` method.
# We also provide a detailed description (metadata) so the orchestrator knows which agent to use for which task.
calculator_tool_agent = FunctionTool(
    fn=lambda query: calculator_agent.chat(query),
    metadata=ToolMetadata(
        name="calculator_agent",
        description="""
        Use this agent for any mathematical calculations, such as addition, subtraction, multiplication, or division.
        Pass the entire mathematical part of the query to this agent.
        For example: 'what is 5 times 4?' or 'calculate 100 / 25'.
        """,
    ),
)

weather_tool_agent = FunctionTool(
    fn=lambda query: weather_agent.chat(query),
    metadata=ToolMetadata(
        name="weather_agent",
        description="""
        Use this agent to get the current weather for a specific city.
        Pass the entire weather-related part of the query to this agent.
        For example: 'what is the weather in London?'.
        """,
    ),
)

# The Orchestrator gets the agent-tools.
print("Creating Orchestrator Agent...")
orchestrator_agent = AgentRunner(
    ReActAgent.from_tools(
        tools=[calculator_tool_agent, weather_tool_agent],
        llm=llm,
        verbose=True,
        # This context is crucial for the planning and delegation logic.
        context="""
        You are an orchestrator agent. Your job is to understand a user's complex query, break it down into sub-tasks, and delegate each sub-task to the correct specialized agent.
        You have two agents available: a 'calculator_agent' for math and a 'weather_agent' for weather.

        **Your Workflow:**
        1.  **Analyze the user's prompt** to identify all the distinct tasks required.
        2.  **Create a step-by-step plan.** If tasks are dependent on each other (e.g., using the result of a calculation in a later step), you must execute them in the correct order.
        3.  **Delegate:** For each step, call the appropriate agent ('calculator_agent' or 'weather_agent') with the precise sub-query it needs to solve.
        4.  **Synthesize:** After all sub-tasks are complete, gather the results and present a final, consolidated, and easy-to-understand answer to the user.
        5.  **Handle Dependencies:** If a query is "add 2 and 3, then subtract 5", you must first call the calculator for "add 2 and 3", get the result (5), and then call the calculator again for "subtract 5 from the previous result".
        """
    )
)
print("Orchestrator Agent created.")


# --- 6. Run the Workflow ---
print("\n--- Running the Orchestrator Agent ---")
user_prompt = "Add 2 and 3, then subtract 5 from the result. Also, what's the weather in new york?"
print(f"User Prompt: {user_prompt}\n")

# The orchestrator agent receives the complex prompt
response = orchestrator_agent.chat(user_prompt)

print("\n\n--- Final Response from Orchestrator ---")
print(response)

