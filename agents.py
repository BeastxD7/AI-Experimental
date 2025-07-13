from llama_index.core.agent.workflow import FunctionAgent
from llmConfig import llm
from tools import search_web  

research_agent = FunctionAgent(
    system_prompt=(
        "You are the ResearchAgent that can search the web for information on a given topic and record notes on the topic. "
        "You should output notes on the topic in a structured format."
    ),
    llm=llm,
    tools=[search_web],
)

write_agent = FunctionAgent(
    system_prompt=(
        "You are the WriteAgent that can write a report on a given topic. "
        "Your report should be in a markdown format. The content should be grounded in the research notes. "
        "Return your markdown report surrounded by ... tags. "
        "NOTE: Please Don't use Emojis in the report."
    ),
    llm=llm,
    tools=[],
)

review_agent = FunctionAgent(
    system_prompt=(
        "You are the ReviewAgent that can review the write report and provide feedback. "
        "Your review should either approve the current report or request changes to be implemented."
    ),
    llm=llm,
    tools=[],
)