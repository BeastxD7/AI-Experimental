# Fixed Multi-Agent Setup with LlamaIndex, Ollama, and AgentWorkflow
# This corrected version addresses both critical issues

import asyncio
import os
from dotenv import load_dotenv
from typing import List, Optional
from llama_index.llms.ollama import Ollama
from llama_index.llms.groq import Groq  
from llama_index.core.agent.workflow import FunctionAgent, AgentWorkflow
from llama_index.core.agent.workflow import AgentInput, AgentOutput, ToolCall, ToolCallResult, AgentStream
from llama_index.core.workflow import Context
from llama_index.core.tools import FunctionTool  # CRITICAL: Import FunctionTool

load_dotenv()

# ===== TOOLS IMPLEMENTATION =====

def add(x: float, y: float) -> float:
    """Add two numbers together."""
    return x + y

def subtract(x: float, y: float) -> float:
    """Subtract y from x."""
    return x - y

def multiply(x: float, y: float) -> float:
    """Multiply two numbers together."""
    return x * y

def divide(x: float, y: float) -> float:
    """Divide x by y."""
    if y == 0:
        return "Error: Division by zero"
    return x / y

def get_weather(city: str) -> str:
    """Get weather information for a given city (returns hardcoded demo data)."""
    weather_data = {
        "new york": "Temperature: 72°F, Sunny with light clouds",
        "london": "Temperature: 65°F, Cloudy with occasional rain",
        "tokyo": "Temperature: 78°F, Clear skies",
        "paris": "Temperature: 68°F, Partly cloudy",
        "berlin": "Temperature: 64°F, Overcast"
    }
    
    city_lower = city.lower()
    if city_lower in weather_data:
        return f"Weather in {city}: {weather_data[city_lower]}"
    else:
        return f"Weather in {city}: Temperature: 70°F, Partly cloudy (default)"

# ===== CRITICAL FIX: PROPER TOOL REGISTRATION =====
# Convert all functions to FunctionTool objects
calculator_tools = [
    FunctionTool.from_defaults(fn=add),
    FunctionTool.from_defaults(fn=subtract),
    FunctionTool.from_defaults(fn=multiply),
    FunctionTool.from_defaults(fn=divide),
]

weather_tools = [
    FunctionTool.from_defaults(fn=get_weather)
]

# ===== LLM SETUP =====
# Direct model specification - change this to your preferred model
# llm = Ollama(
#     model="llama3.1:8b",
#     request_timeout=30.0,
#     context_window=8000,
# )


llm = Groq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.1,
)


# ===== AGENT DEFINITIONS =====

# Calculator Agent - handles mathematical operations
calculator_agent = FunctionAgent(
    name="CalculatorAgent",
    description="Performs basic arithmetic operations including addition, subtraction, multiplication, and division.",
    system_prompt=(
        "You are a calculator agent that can perform basic arithmetic operations. "
        "You have access to add, subtract, multiply, and divide functions. "
        "When you complete mathematical operations, you should hand off control back to the OrchestratorAgent "
        "to continue with any other tasks or provide the final response."
    ),
    llm=llm,
    tools=calculator_tools,  # Use the properly registered tools
    can_handoff_to=["OrchestratorAgent"],
)

# Weather Agent - handles weather queries
weather_agent = FunctionAgent(
    name="WeatherAgent", 
    description="Provides weather information for cities around the world.",
    system_prompt=(
        "You are a weather agent that can provide weather information for any city. "
        "Use the get_weather tool to retrieve weather data. This is only the source of truth for weather information. "
        "When you complete weather queries, you should hand off control back to the OrchestratorAgent "
        "to continue with any other tasks or provide the final response."
    ),
    llm=llm,
    tools=weather_tools,  # Use the properly registered tools
    can_handoff_to=["OrchestratorAgent"],
)

# Orchestrator Agent - manages the workflow and delegates tasks
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

# ===== WORKFLOW SETUP =====

# Create the AgentWorkflow
workflow = AgentWorkflow(
    agents=[orchestrator_agent, calculator_agent, weather_agent],
    root_agent="OrchestratorAgent",  # Start with the orchestrator
    initial_state={
        "calculation_results": [],
        "weather_results": [],
        "task_queue": []
    },
)

# ===== STREAMING OUTPUT FUNCTION =====

async def stream_workflow_events(handler):
    """Stream workflow events to provide real-time feedback."""
    current_agent = None
    
    async for event in handler.stream_events():
        # Track current agent
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            print(f"\n{'='*50}")
            print(f"🤖 Active Agent: {current_agent}")
            print(f"{'='*50}")
        
        # Handle different event types
        if isinstance(event, AgentOutput):
            if event.response.content:
                print(f"📤 Agent Response: {event.response.content}")
            if event.tool_calls:
                print(f"🛠️  Planning to use tools: {[call.tool_name for call in event.tool_calls]}")
        
        elif isinstance(event, ToolCall):
            print(f"🔨 Calling Tool: {event.tool_name}")
            print(f"   Arguments: {event.tool_kwargs}")
        
        elif isinstance(event, ToolCallResult):
            print(f"🔧 Tool Result ({event.tool_name}): {event.tool_output}")
        
        elif isinstance(event, AgentStream):
            # Stream token by token output
            print(event.delta, end="", flush=True)

# ===== INTERACTIVE MODE =====

async def interactive_mode():
    """Interactive mode for testing the multi-agent system."""
    print("🚀 Interactive Multi-Agent System")
    print("=" * 50)
    print("Available agents:")
    print("- OrchestratorAgent: Manages tasks and delegates to other agents")
    print("- CalculatorAgent: Handles mathematical operations")
    print("- WeatherAgent: Provides weather information")
    print("=" * 50)
    print("Type 'exit' to quit")
    print("Example queries:")
    print("- 'Add 5 and 3, then multiply by 2'")
    print("- 'What's the weather in Paris?'")
    print("- 'Calculate 20 / 4 and tell me weather in Berlin'")
    print("=" * 50)
    
    while True:
        user_input = input("\n💬 Enter your query: ").strip()
        
        if user_input.lower() == 'exit':
            print("👋 Goodbye!")
            break
        
        if not user_input:
            continue
        
        print(f"\n🔍 Processing: {user_input}")
        print("-" * 50)
        
        try:
            # Run the workflow
            handler = workflow.run(user_msg=user_input)
            
            # Stream the events
            await stream_workflow_events(handler)
            
            # CRITICAL FIX: Use await handler instead of handler.get_final_response()
            final_response = await handler
            print(f"\n✅ Final Response: {final_response}")
            
        except Exception as e:
            print(f"❌ Error: {e}")

# ===== MAIN ENTRY POINT =====

if __name__ == "__main__":
    # Always run in interactive mode
    asyncio.run(interactive_mode())
