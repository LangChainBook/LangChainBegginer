import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain.tools import Tool
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import create_retriever_tool

load_dotenv() # Load environment variables
api_key = os.getenv('OPENAI_API_KEY')

# Initialize the chat model and embeddings
chat_model = ChatOpenAI(api_key=api_key, model_name="gpt-4")
embedding = OpenAIEmbeddings()

# Define a web scraping tool function
def web_scraper(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        return {"content": soup.get_text()}
    except Exception as e:
        return {"content": str(e)}

# Define the web scraping tool from the function
web_scraping_tool = Tool(
    name="web_scraper",
    func=web_scraper,
    description="A tool to scrape information from a website"
)


# Define the prompt template
template = """
You are a helpful assistant.
Use the web_scraper tool to scrape information from {url}.
{agent_scratchpad}
"""
prompt_template = ChatPromptTemplate.from_template(template)

# Fetch documents from the web
# You can replace the URL with any other website
docs = WebBaseLoader(["https://www.cubanhacker.com"]).load()

# Split documents into manageable chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=20
)
split_docs = splitter.split_documents(docs)

# Create embeddings and vector store initialized with a single dot
vector_store = Chroma.from_documents(
    split_docs,
    embedding=embedding
)

# Create a retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

# Create the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly assistant called Max."),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad", optional=True),
])

# Initialize Tavily Search tool
search_tool = TavilySearchResults()

# Create the retriever tool
retriever_tool = create_retriever_tool(
    retriever,
    "custom_knowledge_base",
    "Use this tool to retrieve information from the custom knowledge base."
)

# Combine tools
tools = [search_tool, retriever_tool, web_scraping_tool]

# Create an agent with the tools
agent = create_openai_functions_agent(
    llm=chat_model,
    prompt=prompt,
    tools=tools
)

# Create the agent executor
executor = AgentExecutor(agent=agent, tools=tools)

# Function to process chat with the agent
def process_chat(agent_executor, user_input, chat_history):
    response = agent_executor.invoke({
        "input": user_input,
        "chat_history": chat_history
    })
    return response['output']

# Main loop to interact with the agent
if __name__ == "__main__":
    chat_history = []
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        response = process_chat(executor, user_input, chat_history)
        print(f"Max: {response}")
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))


