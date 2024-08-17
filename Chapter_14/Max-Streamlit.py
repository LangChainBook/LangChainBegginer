import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain.agents import create_openai_tools_agent
from langchain.agents import  AgentExecutor
from langchain.prompts import  ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import HumanMessage,AIMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import create_retriever_tool
from langchain.globals import set_verbose


load_dotenv() # Load environment variables
api_key = os.getenv('OPENAI_API_KEY')

# Set verbosity for debugging
set_verbose(True)

st.set_page_config(layout="wide",initial_sidebar_state="collapsed")

# Initialize the chat model and embeddings
chat_model = ChatOpenAI(api_key=api_key, model_name="gpt-4")
embedding = OpenAIEmbeddings()


# Define a web scraping tool function
def web_scraper(url):
    with st.spinner(f"Scraping information from the website {url}"):
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.content)
            return {"content": soup.get_text()}
        except Exception as e:
            return {"content": str(e)}

# Define the web scraping tool from the function
web_scraping_tool = Tool(
    name="web_scraper",
    func=web_scraper,
    description="A tool to scrape information from a website"
)

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
    embedding=embedding,
    persist_directory="."
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
agent = create_openai_tools_agent(
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

# Set the page layout for Streamlit app

# Streamlit UI setup
st.markdown(    # Custom CSS for fixed input and scrolling chat box
    """
    <style>
        div[data-testid="stVerticalBlock"] div:has(div.fixed-header)
        {    position: sticky;
            top: 2.875rem;
            background-color: white;
            z-index: 999;
        }
        .fixed-header {
            border-bottom: 1px solid black;
        }
    </style>
    """,
    unsafe_allow_html=True
)

with st.container():
    st.title("Interactive Chat with Max")

    st.write("<div class='fixed-header'/>", unsafe_allow_html=True)
with st.sidebar:
    st.title('Side Bar')


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        st.markdown(
            f"<p><strong>You:</strong> {message.content}</p>",
            unsafe_allow_html=True
        )
    elif isinstance(message, AIMessage):
        st.markdown(
            f"<p><strong>Max:</strong> {message.content}</p>",
            unsafe_allow_html=True
        )
st.markdown('</div>', unsafe_allow_html=True)

# Form for user input
st.markdown('<div class="fixed-input">', unsafe_allow_html=True)
user_input = st.chat_input("You: ", key="input_text")
st.markdown('</div>', unsafe_allow_html=True)

# Fixed input box
if user_input:
    with st.spinner(f"Thinking about {user_input}"): #spinner for chat
        try:
            response = executor.invoke({
                "input": user_input,
                "chat_history": st.session_state.chat_history
            })
            st.session_state.chat_history.append(
                HumanMessage(content=user_input)
            )
            st.session_state.chat_history.append(
                AIMessage(content=response['output'])
            )
            st.rerun()  # Refresh the page to display new messages
        except Exception as e:
            st.error(f"Failed to process the request: {e}")





