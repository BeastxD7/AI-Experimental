from llama_index.core.agent.workflow import FunctionAgent, AgentWorkflow
from llama_index.llms.ollama import Ollama
from llama_index.core.tools import FunctionTool
import asyncio


# Calculator Tools for calculator operations

def add_tool(n1: float, n2: float) -> float:
    """Useful for adding two numbers."""
    print(f"Add Tool: Adding {n1} and {n2}")
    return n1 + n2

def subtract_tool(n1: float, n2: float) -> float:
    """Useful for subtracting two numbers."""
    print(f"Subtract Tool: Subtracting {n2} from {n1}") 
    return n1 - n2

def multiply_tool(n1: float, n2: float) -> float:
    """Useful for multiplying two numbers."""
    print(f"Multiply Tool: Multiplying {n1} and {n2}")  
    return n1 * n2

def divide_tool(n1: float, n2: float) -> float:
    """Useful for dividing two numbers."""
    print(f"Divide Tool: Dividing {n1} by {n2}")
    if n2 == 0:
        raise ValueError("Cannot divide by zero")
    return n1 / n2

def sqrt_tool(n: float) -> float:
    """Useful for calculating the square root of a number."""
    print(f"Sqrt Tool: Calculating square root of {n}")
    if n < 0:
        raise ValueError("Cannot calculate square root of negative number")
    return n ** 0.5

def power_tool(base: float, exponent: float) -> float:
    """Useful for calculating the power of a number."""
    print(f"Power Tool: Calculating {base} raised to the power of {exponent}")
    return base ** exponent

# Date and time tools for DateTimeAgent

def get_current_date() -> str:
    """Useful for getting the current date."""
    from datetime import datetime
    current_date = datetime.now().strftime("%Y-%m-%d")
    print(f"GetCurrentDate Tool: Current date is {current_date}")
    return current_date

def get_current_time() -> str:
    """Useful for getting the current time."""
    from datetime import datetime
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"GetCurrentTime Tool: Current time is {current_time}")
    return current_time

def get_current_day() -> str:
    """Useful for getting the current day of the week."""
    from datetime import datetime
    current_day = datetime.now().strftime("%A")
    print(f"GetCurrentDay Tool: Current day is {current_day}")
    return current_day


# Connect to local Ollama
llm = Ollama(model="llama3.1:8b", request_timeout=120.0, context_window=8000)

# Define the CalculatorAgent with FunctionTools

CalculatorAgent = FunctionAgent(
    name="CalculatorAgent",
    description="Performs basic arithmetic operations.",
    system_prompt = (
    "You are a calculator agent named CalculatorAgent. "
    "You ONLY assist with simple single-step arithmetic operations like addition, subtraction, multiplication, division, square root, and power. "
    "You have access to the following tools: add_tool, subtract_tool, multiply_tool, divide_tool, sqrt_tool, power_tool. "
    "If the user asks anything that is not a simple single-step arithmetic calculation, including personal questions or complex math, respond with: 'I can only assist you with simple arithmetic operations.' "
    "When the user asks a valid calculation, respond by calling the appropriate function tool in the format required for tool execution, not by showing the calculation as text. "
    "After executing the function, respond with: 'The result is X.' replacing X with the calculated result."
    "If user asks about date or time, handoff to Agent DateTimeAgent. "
),

    tools=[
        add_tool,
        subtract_tool,
        multiply_tool,
        divide_tool,
        sqrt_tool,
        power_tool
    ],
    llm=llm,
    can_handoff_to=["DateTimeAgent"]  # Allow handoff to DateTimeAgent for date/time queries
)


DateTimeAgent = FunctionAgent(
    name="DateTimeAgent",
    description="Provides current date and time information.",
    system_prompt=(
        "You are a DateTime agent. "
        "You ONLY provide current date, time, and day of the week. "
        "You have access to the following tools: get_current_date, get_current_time, get_current_day. "
        "If the user asks anything that is not related to date or time, reply ONLY with: 'I can only assist you with date and time information.'"
        "If the user asks about date, time, or day, use the appropriate tool to provide the information."
        "If user asks for both date and time, provide both in a single response."
        "If the user asks for mathematical operations, handoff to agent CalculatorAgent."
    ),
    tools=[
        get_current_date,
        get_current_time,
        get_current_day
    ],
    llm=llm,
    can_handoff_to=["CalculatorAgent"]  # Allow handoff to CalculatorAgent for math queries
)

# 4. WORKFLOW ORCHESTRATOR

workflow = AgentWorkflow(
    agents=[CalculatorAgent, DateTimeAgent],
    root_agent="CalculatorAgent",  # Start with Calculator agent, can be changed as needed
    verbose=True
)

# Run the workflow
async def main():
    print("Welcome to the multi-agent system using local Ollama model.")

    print("Registered agents in workflow:", workflow.agents.keys())

    
    user_prompt = input("Enter your query: ")
    response = await workflow.run(user_msg=user_prompt)
    print("\nResponse:", response)

if __name__ == "__main__":
    asyncio.run(main())