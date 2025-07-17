import asyncio
import os
from llama_index.core.agent.workflow import FunctionAgent, AgentWorkflow, AgentStream
from llama_index.llms.groq import Groq # Changed from Ollama to Groq 
from llama_index.llms.ollama import Ollama
from llama_index.core.workflow import Context
from llama_index.core.tools import FunctionTool
from dotenv import load_dotenv

load_dotenv()

# Initialize Groq LLM
llm = Groq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.1,
)

# llm = Ollama(
#     model="llama3.1:8b",
#     base_url="http://localhost:11434",
#     request_timeout=600.0,
# )

# Tool functions (unchanged)
def add(a: float, b: float) -> float:
    """Add two numbers together and return the result."""
    return a + b

def subtract(a: float, b: float) -> float:
    """Subtract the second number from the first number and return the result."""
    return a - b

def multiply(a: float, b: float) -> float:
    """Multiply two numbers together and return the result."""
    return a * b

def divide(a: float, b: float) -> float:
    """Divide the first number by the second number and return the result."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

def getWeather(city: str) -> str:
    """Get the current weather for a given city."""
    return f"The current weather in {city} is sunny with a temperature of 25°C."

# Create function tools
add_tool = FunctionTool.from_defaults(fn=add)
subtract_tool = FunctionTool.from_defaults(fn=subtract)
multiply_tool = FunctionTool.from_defaults(fn=multiply)
divide_tool = FunctionTool.from_defaults(fn=divide)
get_weather_tool = FunctionTool.from_defaults(fn=getWeather)

# Create agents (REMOVED can_handoff_to parameter)
calculator_agent = FunctionAgent(
    name="CalculatorAgent",
    description="A mathematical calculator that can perform arithmetic operations.",
    system_prompt="You are a helpful mathematical assistant. Use the available tools to perform calculations step by step. Show your work clearly. If asked about weather, hand off to the WeatherAgent.",
    tools=[add_tool, subtract_tool, multiply_tool, divide_tool],
    llm=llm,
    verbose=True,
    can_handoff_to=["WeatherAgent"]  # Added handoff capability for calculator agent
)

weather_agent = FunctionAgent(
    name="WeatherAgent", 
    description="A weather information agent that provides current weather data.",
    system_prompt="You are a helpful weather assistant. Provide accurate and concise weather information. If asked about calculations, hand off to the CalculatorAgent.",
    tools=[get_weather_tool],
    llm=llm,
    verbose=True,
    can_handoff_to=["CalculatorAgent"]  # Added handoff capability for weather agent
)


# Create AgentWorkflow (FIXED variable name)
workflow = AgentWorkflow(
    agents=[calculator_agent, weather_agent],
    root_agent="CalculatorAgent",
    initial_state={
        "conversation_history": [],
        "last_calculation": None,
        "last_weather_query": None
    }
)

async def get_user_input(prompt: str) -> str:
    """Async wrapper for user input"""
    return await asyncio.to_thread(input, prompt)

async def process_query(query: str, ctx: Context) -> None:
    """Process user query with streaming output"""
    print("Response: ", end="", flush=True)
    
    try:
        # Use workflow instead of undefined agent
        handler = workflow.run(user_msg=query, ctx=ctx)
        
        async for event in handler.stream_events():
            if isinstance(event, AgentStream):
                print(event.delta, end="", flush=True)
        
        print()  # New line after response
        
    except Exception as e:
        print(f"❌ Error: {e}")

async def main():
    """Main interactive loop"""
    print("="*60)
    print("🧮 Interactive Multi-Agent System")
    print("="*60)
    
    
    # Create context with the workflow (FIXED)
    ctx = Context(workflow)
    
    while True:
        try:
            user_input = await get_user_input("\n📝 Enter your question: ")
            
            if user_input.lower() in ['exit', 'quit']:
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'clear':
                ctx = Context(workflow)
                print("🗑️ Context cleared!")
                continue
            elif not user_input.strip():
                continue
            
            await process_query(user_input, ctx)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    asyncio.run(main())