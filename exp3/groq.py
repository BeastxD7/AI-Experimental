import asyncio
import os
from dotenv import load_dotenv
from llama_index.core.agent.workflow import FunctionAgent, AgentStream
from llama_index.llms.groq import Groq  # Changed from Ollama to Groq
from llama_index.core.workflow import Context
from llama_index.core.tools import FunctionTool

# Load environment variables
load_dotenv()

# Initialize Groq LLM with llama-3.1-8b-instant model
llm = Groq(
    model="llama-3.1-8b-instant",  # Specified model
    api_key=os.getenv("GROQ_API_KEY"),  # Load API key from environment
    temperature=0.1,  # Optional: adjust for more deterministic responses
)

# Your existing arithmetic tools (unchanged)
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

# Your existing tool creation (unchanged)
add_tool = FunctionTool.from_defaults(fn=add, name="add", description="Adds two numbers together. Where a and b are parameters.")
subtract_tool = FunctionTool.from_defaults(fn=subtract, name="subtract", description="Subtracts the second number from the first. Where a is the minuend and b is the subtrahend.")
multiply_tool = FunctionTool.from_defaults(fn=multiply, name="multiply", description="Multiplies two numbers together. Where a and b are parameters.")
divide_tool = FunctionTool.from_defaults(fn=divide, name="divide", description="Divides the first number by the second. Where a is the dividend and b is the divisor.")

# Your existing agent creation (unchanged)
agent = FunctionAgent(
    tools=[add_tool, subtract_tool, multiply_tool, divide_tool],
    llm=llm,
    system_prompt="You are a helpful mathematical assistant. Use the available tools to perform calculations step by step. Show your work clearly.",
    verbose=True
)

async def get_user_input(prompt: str) -> str:
    """Async wrapper for user input using asyncio.to_thread()"""
    return await asyncio.to_thread(input, prompt)

async def process_query(query: str, ctx: Context) -> None:
    """Process user query with streaming output"""
    print("Response: ", end="", flush=True)
    
    try:
        handler = agent.run(user_msg=query, ctx=ctx)
        
        async for event in handler.stream_events():
            if isinstance(event, AgentStream):
                print(event.delta, end="", flush=True)
        
        print()  # New line after response
        
    except Exception as e:
        print(f"❌ Error: {e}")

async def main():
    """Main interactive loop"""
    print("="*60)
    print("🧮 Interactive Calculator Agent (Powered by Groq)")
    print("="*60)
    print("Commands:")
    print("  • Enter mathematical expressions (e.g., 'What is 25 + 17?')")
    print("  • Type 'exit' or 'quit' to stop")
    print("  • Type 'clear' to reset conversation context")
    print("="*60)
    
    # Create context for memory management
    ctx = Context(agent)
    
    while True:
        try:
            # Get user input asynchronously
            user_input = await get_user_input("\n📝 Enter calculation: ")
            
            # Handle special commands
            if user_input.lower() in ['exit', 'quit']:
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'clear':
                ctx = Context(agent)
                print("🗑️ Context cleared!")
                continue
            elif not user_input.strip():
                continue
            
            # Process the query with context and streaming
            await process_query(user_input, ctx)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
