# Complete Four-Agent Pipeline with LlamaIndex and Ollama
# Weather Agent + Calculator Agent + Orchestrator Agent + Chat Agent

import asyncio
from typing import Dict, Any, List
from llama_index.llms.ollama import Ollama
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage
from llama_index.core.chat_engine.types import AgentChatResponse

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
    """Get weather information for a given city (returns dummy temperature data)."""
    weather_data = {
        "india": "Temperature: 30°C, Sunny and hot",
        "new york": "Temperature: 22°C, Partly cloudy",
        "london": "Temperature: 18°C, Rainy",
        "tokyo": "Temperature: 25°C, Clear skies",
        "paris": "Temperature: 20°C, Cloudy",
        "berlin": "Temperature: 19°C, Overcast"
    }
    
    city_lower = city.lower()
    if city_lower in weather_data:
        return f"Weather in {city}: {weather_data[city_lower]}"
    else:
        return f"Weather in {city}: Temperature: 25°C, Partly cloudy (default)"

# ===== TOOL REGISTRATION =====
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
llm = Ollama(
    model="llama3.1:8b",
    request_timeout=120.0,
    context_window=8000,
)

# ===== INDIVIDUAL AGENT CREATION =====

# Weather Agent
weather_agent = ReActAgent.from_tools(
    weather_tools,
    llm=llm,
    verbose=True,
    memory=ChatMemoryBuffer.from_defaults(token_limit=2000),
    system_prompt=(
        "You are a weather agent that provides weather information for cities. "
        "Use the get_weather tool to retrieve weather data for any city requested. "
        "Always use the tool to get weather information, never make up weather data. "
        "Return the weather information in a clear, concise format."
    )
)

# Calculator Agent
calculator_agent = ReActAgent.from_tools(
    calculator_tools,
    llm=llm,
    verbose=True,
    memory=ChatMemoryBuffer.from_defaults(token_limit=2000),
    system_prompt=(
        "You are a calculator agent that performs mathematical operations. "
        "Use the available tools (add, subtract, multiply, divide) to solve mathematical problems. "
        "Show your work step by step and provide clear results. "
        "Always use the tools for calculations, never calculate manually."
    )
)

# ===== ORCHESTRATOR CLASS =====
class SimpleOrchestrator:
    def __init__(self):
        self.calculator_agent = calculator_agent
        self.weather_agent = weather_agent
        self.context = {
            "weather_results": [],
            "calculation_results": [],
            "task_sequence": []
        }
        
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze user query to determine required actions."""
        query_lower = query.lower()
        
        # Detect math operations
        math_keywords = [
            'add', 'subtract', 'multiply', 'divide', 'calculate', 'compute',
            '+', '-', '*', '/', 'times', 'plus', 'minus', 'divided by',
            'what is', 'equals', 'sum', 'difference', 'product', 'quotient'
        ]
        
        # Detect weather queries
        weather_keywords = [
            'weather', 'temperature', 'temp', 'climate', 'forecast',
            'hot', 'cold', 'sunny', 'rainy', 'cloudy'
        ]
        
        has_math = any(keyword in query_lower for keyword in math_keywords)
        has_weather = any(keyword in query_lower for keyword in weather_keywords)
        
        # Extract numbers for math operations
        import re
        numbers = re.findall(r'\d+\.?\d*', query)
        
        # Extract city names (simple approach)
        city_patterns = [
            r'weather.*?(?:in|of|for)\s+(\w+)',
            r'temperature.*?(?:in|of|for)\s+(\w+)',
            r'(?:in|of|for)\s+(\w+).*?weather'
        ]
        
        cities = []
        for pattern in city_patterns:
            matches = re.findall(pattern, query_lower)
            cities.extend(matches)
        
        # If no specific city mentioned, look for common city names
        common_cities = ['india', 'new york', 'london', 'tokyo', 'paris', 'berlin']
        if not cities:
            for city in common_cities:
                if city in query_lower:
                    cities.append(city)
        
        return {
            'has_math': has_math,
            'has_weather': has_weather,
            'numbers': numbers,
            'cities': cities,
            'original_query': query
        }
    
    async def execute_plan(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the planned tasks based on analysis."""
        results = {
            'weather_results': [],
            'calculation_results': [],
            'execution_log': []
        }
        
        # Execute weather queries
        if analysis['has_weather']:
            print("🌤️  Executing weather queries...")
            cities = analysis['cities'] if analysis['cities'] else ['default location']
            
            for city in cities:
                try:
                    weather_query = f"What is the weather in {city}?"
                    weather_response = await self.weather_agent.achat(weather_query)
                    results['weather_results'].append({
                        'city': city,
                        'response': str(weather_response.response)
                    })
                    results['execution_log'].append(f"✅ Weather query for {city} completed")
                except Exception as e:
                    results['execution_log'].append(f"❌ Weather query for {city} failed: {e}")
        
        # Execute math operations
        if analysis['has_math']:
            print("🔢 Executing mathematical operations...")
            try:
                math_response = await self.calculator_agent.achat(analysis['original_query'])
                results['calculation_results'].append({
                    'query': analysis['original_query'],
                    'response': str(math_response.response)
                })
                results['execution_log'].append("✅ Mathematical calculation completed")
            except Exception as e:
                results['execution_log'].append(f"❌ Mathematical calculation failed: {e}")
        
        return results
    
    async def process_query(self, user_query: str) -> str:
        """Main orchestration method."""
        print(f"🎯 Orchestrator analyzing query: {user_query}")
        
        # Step 1: Analyze the query
        analysis = self.analyze_query(user_query)
        print(f"📊 Analysis results: Math={analysis['has_math']}, Weather={analysis['has_weather']}")
        
        # Step 2: Execute the plan
        results = await self.execute_plan(analysis)
        
        # Step 3: Return results for chat agent
        return results

# ===== CHAT AGENT =====
class ChatAgent:
    def __init__(self):
        self.orchestrator = SimpleOrchestrator()
        self.llm = llm
        
    async def format_response(self, results: Dict[str, Any], original_query: str) -> str:
        """Format the results into a natural language response."""
        response_parts = []
        
        # Add calculation results
        if results['calculation_results']:
            for calc_result in results['calculation_results']:
                response_parts.append(f"Mathematical result: {calc_result['response']}")
        
        # Add weather results
        if results['weather_results']:
            for weather_result in results['weather_results']:
                response_parts.append(f"Weather information: {weather_result['response']}")
        
        # If no specific results, provide a general response
        if not response_parts:
            response_parts.append("I couldn't find specific information to answer your query.")
        
        return "\n\n".join(response_parts)
    
    async def process_user_message(self, user_message: str) -> str:
        """Process user message through the entire pipeline."""
        print(f"💬 Chat Agent received: {user_message}")
        
        # Step 1: Hand off to orchestrator
        orchestrator_results = await self.orchestrator.process_query(user_message)
        
        # Step 2: Format final response
        final_response = await self.format_response(orchestrator_results, user_message)
        
        return final_response

# ===== MAIN INTERACTIVE SYSTEM =====
async def interactive_mode():
    """Interactive mode for testing the four-agent system."""
    print("🚀 Four-Agent Interactive System")
    print("=" * 60)
    print("Available capabilities:")
    print("- Weather: Get weather information for any city")
    print("- Calculator: Perform mathematical operations")
    print("- Orchestrator: Plan and coordinate tasks")
    print("- Chat: Natural language interaction")
    print("=" * 60)
    print("Type 'exit' to quit")
    print("Example queries:")
    print("- 'What is the weather in India and what is 2+2?'")
    print("- 'Calculate 15 * 4 and tell me weather in Tokyo'")
    print("- 'What is 100 divided by 25?'")
    print("- 'Weather in London please'")
    print("=" * 60)
    
    chat_agent = ChatAgent()
    
    while True:
        user_input = input("\n💬 Enter your query: ").strip()
        
        if user_input.lower() == 'exit':
            print("👋 Goodbye!")
            break
        
        if not user_input:
            continue
        
        print(f"\n🔍 Processing: {user_input}")
        print("-" * 60)
        
        try:
            response = await chat_agent.process_user_message(user_input)
            print(f"\n✅ Final Response:")
            print(response)
            print("\n" + "=" * 60)
        except Exception as e:
            print(f"❌ Error: {e}")

# ===== MAIN ENTRY POINT =====
if __name__ == "__main__":
    asyncio.run(interactive_mode())
