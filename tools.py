from tavily import AsyncTavilyClient
import dotenv
import os

dotenv.load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API")

if TAVILY_API_KEY is None:
    raise ValueError("🚨 TAVILY_API environment variable not found in .env. Please set it before running.")


async def search_web(query: str) -> str:
    """Useful for using the web to answer questions."""
    client = AsyncTavilyClient(api_key=TAVILY_API_KEY)
    return str(await client.search(query))
