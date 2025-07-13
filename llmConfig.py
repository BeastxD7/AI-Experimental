from llama_index.llms.ollama import Ollama


llm = Ollama(
    model="llama3.1:8b",
    base_url="http://localhost:11434",
    temperature=0.7,
    request_timeout=120.0
)