from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

@tool
def search(query: str) -> str:
    """Search for information about a query."""
    # In a real scenario, this would be an actual search function
    return f"Results for: {query}. AGI has been achieved !"

model = ChatOpenAI(model="gpt-4", api_key=api_key)
tools = [search]

# Create a prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# Create the agent
agent = create_openai_tools_agent(model, tools, prompt)

# Create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Run the agent
result = agent_executor.invoke(
    {"input": "What's the latest news about artificial intelligence?"})
print(result['output'])
