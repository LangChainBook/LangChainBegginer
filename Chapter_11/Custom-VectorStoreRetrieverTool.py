import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.tools import create_retriever_tool
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
embedding = OpenAIEmbeddings()

docscontent = """ # Example documents
One of the most powerful and obvious uses for LLM tool-calling abilities is
to build agents. LangChain already has a create_openai_tools_agent() constructor
that makes it easy to build an agent with tool-calling models that adhere to
the OpenAI tool-calling API, but this won’t work for models like Anthropic and
Gemini. Thanks to the new bind_tools() and tool_calls interfaces, we’ve added
a create_tool_calling_agent() that works with any tool-calling model.
"""

splitter = RecursiveCharacterTextSplitter( # define the splitter
        chunk_size=400, chunk_overlap=20)

# Create embeddings and vector store
docs = [Document(page_content=x) for x in splitter.split_text(docscontent)]
vector_store = Chroma.from_documents(docs, embedding=embedding)
retriever = vector_store.as_retriever( search_kwargs={"k": 3})

retriever_tool = create_retriever_tool( #define retriever tool
        retriever,
        "custom_knowledge_base",
        "Use this tool to retrieve information from the custom knowledge base."
    )

# Perform a search in the vector store
# running the function with the query "agent" as an Agent would execute it.
print(retriever_tool.invoke("agent"))


