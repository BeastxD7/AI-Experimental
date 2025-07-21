import asyncio
import os
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.groq import Groq
from llama_index.llms.ollama import Ollama
from llama_index.core.workflow import Context
from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager
import datetime
from dotenv import load_dotenv

load_dotenv()

# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ---- Enhanced LLM and Debugging Setup ----
# llm = Groq(model="llama3-70b-8192", temperature=0.1)  # Lower temperature for more consistent tool usage
llm = Ollama(model="qwen3:14b", request_timeout=120.0, context_window=8000, is_function_calling_model=True)

# Add debug handler to see what's happening
llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])

# ---- Enhanced Core Tools with Better Descriptions ----
def add_tool(n1: float, n2: float) -> float:
    """Add two numbers together and return the sum."""
    print(f"🔢 ADD TOOL: {n1} + {n2} = {n1 + n2}")
    return n1 + n2

def subtract_tool(n1: float, n2: float) -> float:
    """Subtract the second number from the first number."""
    print(f"🔢 SUBTRACT TOOL: {n1} - {n2} = {n1 - n2}")
    return n1 - n2

def multiply_tool(n1: float, n2: float) -> float:
    """Multiply two numbers together and return the product."""
    print(f"🔢 MULTIPLY TOOL: {n1} × {n2} = {n1 * n2}")
    return n1 * n2

def get_current_day() -> str:
    """Get the current day of the week."""
    day = datetime.datetime.now().strftime("%A")
    print(f"📅 DAY TOOL: Today is {day}")
    return day

def get_current_month() -> int:
    """Get the current month as an integer (1-12)."""
    month = datetime.datetime.now().month
    print(f"📅 MONTH TOOL: Current month is {month}")
    return month

def get_current_date() -> str:
    """Get the current date in YYYY-MM-DD format."""
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    print(f"📅 DATE TOOL: Current date is {date}")
    return date

# ---- Enhanced FunctionTool Wrappers ----
CalcAddTool = FunctionTool.from_defaults(fn=add_tool)
CalcSubtractTool = FunctionTool.from_defaults(fn=subtract_tool)
CalcMultiplyTool = FunctionTool.from_defaults(fn=multiply_tool)
CurrentDayTool = FunctionTool.from_defaults(fn=get_current_day)
CurrentMonthTool = FunctionTool.from_defaults(fn=get_current_month)
CurrentDateTool = FunctionTool.from_defaults(fn=get_current_date)

# ---- Enhanced Sub-Agent: Calculator Agent ----
calculator_agent = FunctionAgent(
    name="CalculatorAgent",
    description="Performs arithmetic operations (add, subtract, multiply).",
    system_prompt=(
        "You are CalculatorAgent. You MUST use the available calculation tools for ANY mathematical operation. "
        "Never calculate manually - always call the appropriate tool. "
        "Available tools: add_tool, subtract_tool, multiply_tool. "
        "After calling a tool, respond with 'The calculation result is X' where X is the tool's output."
    ),
    llm=llm,
    tools=[CalcAddTool, CalcSubtractTool, CalcMultiplyTool],
    callback_manager=callback_manager,
    verbose=True
)

# ---- Enhanced Sub-Agent: DateTime Agent ----
datetime_agent = FunctionAgent(
    name="DateTimeAgent",
    description="Provides current date and time information.",
    system_prompt=(
        "You are DateTimeAgent. You MUST use the available datetime tools for ANY date/time query. "
        "Never provide date/time information from memory - always call the appropriate tool. "
        "Available tools: get_current_day, get_current_month, get_current_date. "
        "After calling a tool, respond with the information from the tool output."
    ),
    llm=llm,
    tools=[CurrentDayTool, CurrentMonthTool, CurrentDateTool],
    callback_manager=callback_manager,
    verbose=True
)

# ---- Enhanced Async Wrappers to Use Sub-Agents as Tools ----
async def calculator_tool(ctx: Context, prompt: str) -> str:
    """
    Use this tool for any mathematical calculation or arithmetic operation.
    Examples: addition, subtraction, multiplication, 'what is 2+2', 'calculate 5*3', etc.
    """
    print(f"🔧 CALCULATOR TOOL called with prompt: '{prompt}'")
    try:
        result = await calculator_agent.run(user_msg=prompt)
        async with ctx.store.edit_state() as ctx_state:
            ctx_state["state"]["math_result"] = str(result)
        print(f"🔧 CALCULATOR TOOL result: {result}")
        return str(result)
    except Exception as e:
        error_msg = f"Calculator tool error: {str(e)}"
        print(f"❌ {error_msg}")
        return error_msg

async def datetime_tool(ctx: Context, prompt: str) -> str:
    """
    Use this tool for any date or time related query.
    Examples: current day, current month, current date, 'what day is today', 'what month is it', etc.
    """
    print(f"🔧 DATETIME TOOL called with prompt: '{prompt}'")
    try:
        result = await datetime_agent.run(user_msg=prompt)
        async with ctx.store.edit_state() as ctx_state:
            if "day" in prompt.lower():
                ctx_state["state"]["current_day"] = str(result)
            elif "month" in prompt.lower():
                ctx_state["state"]["current_month"] = str(result)
            elif "date" in prompt.lower():
                ctx_state["state"]["current_date"] = str(result)
            else:
                ctx_state["state"]["datetime_info"] = str(result)
        print(f"🔧 DATETIME TOOL result: {result}")
        return str(result)
    except Exception as e:
        error_msg = f"DateTime tool error: {str(e)}"
        print(f"❌ {error_msg}")
        return error_msg

# ---- Create Enhanced FunctionTools ----
CalcAgentTool = FunctionTool.from_defaults(
    fn=calculator_tool,
    name="calculator_tool",
    description="Use this tool for ANY mathematical calculation, arithmetic operation, or when numbers need to be added, subtracted, or multiplied. Examples: '2+2', '5*3', 'add 10 and 20', 'what is 15-7'."
)

DateTimeAgentTool = FunctionTool.from_defaults(
    fn=datetime_tool,
    name="datetime_tool", 
    description="Use this tool for ANY date or time related question. Examples: 'what day is today', 'current month', 'what's the date', 'day of the week'."
)

# ---- Enhanced Orchestrator Agent with Stronger System Prompt ----
orchestrator = FunctionAgent(
    name="OrchestratorAgent",
    description="Orchestrates tasks by delegating to specialized tools for math and datetime queries.",
    system_prompt=(
        "You are OrchestratorAgent, a task coordinator that MUST use tools to answer user queries. "
        "CRITICAL INSTRUCTIONS:\n"
        "1. You CANNOT answer questions directly - you MUST use the appropriate tools\n"
        "2. For ANY mathematical question or calculation, use calculator_tool\n"
        "3. For ANY date/time question, use datetime_tool\n"
        "4. If the query has multiple parts, use multiple tools in sequence\n"
        "5. After getting tool results, combine them into a natural response\n\n"
        "Available tools:\n"
        "- calculator_tool: For all math operations (addition, subtraction, multiplication)\n"
        "- datetime_tool: For current day, month, date information\n\n"
        "You must call tools for every relevant part of the user's query. Never skip using tools."
    ),
    llm=llm,
    tools=[CalcAgentTool, DateTimeAgentTool],
    callback_manager=callback_manager,
    verbose=True,
    initial_state={
        "current_day": None,
        "current_month": None,
        "current_date": None,
        "math_result": None,
        "datetime_info": None,
    },
)

# ---- Enhanced Main Event Loop ----
async def main():
    print("🚀 Multi-agent Orchestrator System (Enhanced)")
    print("=" * 50)
    print("Try queries like:")
    print("- 'What day is it today and what is 2 + 2?'")
    print("- 'What's the current month and multiply 5 by 3?'")
    print("- 'Tell me today's date and calculate 10 - 4'")
    print("=" * 50)
    
    while True:
        try:
            user_msg = input("\nEnter your query (or 'quit' to exit): ")
            if user_msg.lower() in ['quit', 'exit', 'q']:
                break
                
            print(f"\n🎯 Processing query: '{user_msg}'")
            print("-" * 30)
            
            # Use arun instead of run for async execution
            response = await orchestrator.run(user_msg=user_msg)
            
            print("\n" + "=" * 30)
            print(f"🤖 Final Response: {response}")
            print("=" * 30)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            continue

if __name__ == "__main__":
    asyncio.run(main())
