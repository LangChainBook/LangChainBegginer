import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema.document import Document
from langchain_core.messages import (HumanMessage,AIMessage )
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.runnables import RunnableLambda
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
# Initialize the LLM
llm = ChatOpenAI(model="gpt-4", api_key=api_key)
prompt = ChatPromptTemplate.from_messages([ #define the prompt
    ("system", """You are a helpful AI assistant, your vector database contains
    related information on previous conversations {documents}"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])
# Split documents into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=500)
embeddings = OpenAIEmbeddings()
# Load documents from a directory
loader = DirectoryLoader("./Chapter_7/documents/")
documents = loader.load()
# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=500
)
#chunk the documents using the text_splitter
chunks = text_splitter.split_documents(documents)
#create or add the document chunks to the vector store
vector_store = Chroma.from_documents(documents=chunks,persist_directory=".",
embedding=embeddings,collection_name="conversations")

def chroma_retriever(input):
    query = input["input"]  # Extract the query string from the input
    results = vector_store.search(query=query, search_type="mmr",k=100)
    documents = [result.page_content for result in results]  # Adjusted to access page_content
    return {"documents": documents, "input": input["input"],"history": input["history"]}
retriever = RunnableLambda(chroma_retriever)

# Initialize memory
memory = ChatMessageHistory()

chain = retriever | prompt | llm    # Create the chain

def savechattochroma (human,ai): # Save chat history to Chroma
    docsx=[]
    docsx.append(Document(page_content=f"human:{human}"))

    docsx.append(Document(page_content=f"AI:{ai}"))
    vector_store.add_documents(documents=docsx) # Save chat history to Chroma
    return

session_id = "user_123"

# Create the runnable with message history
runnable = RunnableWithMessageHistory(chain,lambda session_id: memory,input_messages_key="input",
    history_messages_key="history")
def process_chat(runnable, user_input):
    response = runnable.invoke(    {"input": user_input},
    config={"configurable": {"session_id": session_id}}
).content
    return response

print("Agent is ready to chat. Type 'exit' or 'quit' to end the conversation.")
if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        query = user_input
        response = process_chat(runnable, user_input)
        print(f"AI: {response}")
        savechattochroma(HumanMessage(content=user_input),
AIMessage(content=response)
)



