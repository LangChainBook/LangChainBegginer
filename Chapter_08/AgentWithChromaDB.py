import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema.document import Document
from langchain_core.messages import (
    HumanMessage,
    AIMessage )
import asyncio

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4", api_key=api_key)

# Initialize memory
memory = ChatMessageHistory()

prompt = ChatPromptTemplate.from_messages([ #define the prompt
    ("system", "You are a helpful AI assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])
# Split documents into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=500
)
embeddings = OpenAIEmbeddings()


docs = [Document(page_content=x) for x in splitter.split_text(".")]
vector_store = Chroma.from_documents(
documents=docs,persist_directory=".",
embedding=embeddings,
collection_name="conversations")

chain = prompt | llm    # Create the chain

def savechattochroma (human,ai): # Save chat history to Chroma
    docsx=[]
    docsx.append(Document(page_content=f"human:{human}"))

    docsx.append(Document(page_content=f"AI:{ai}"))
    vector_store.add_documents(documents=docsx) # Save chat history to Chroma
    return

# Create the runnable with message history
runnable = RunnableWithMessageHistory(
    chain,
    lambda session_id: memory,
    input_messages_key="input",
    history_messages_key="history"
)


def process_chat(runnable, user_input):
    response = runnable.invoke(
    {"input": user_input},
    config={"configurable": {"session_id": session_id}}
).content
    return response

session_id = "user_123"
print("Agent is ready to chat. Type 'exit' or 'quit' to end the conversation.")
if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        query = user_input
        results = asyncio.run(vector_store.asearch(query, k=25,search_type="mmr"))
        addedresults=""
        addedresults = "\n".join([result.page_content for result in results])
        response = process_chat(runnable, user_input+
        ", consider this context from previous conversations:"+addedresults)
        print(f"AI: {response}")
        savechattochroma(HumanMessage(content=user_input),
AIMessage(content=response)
)



