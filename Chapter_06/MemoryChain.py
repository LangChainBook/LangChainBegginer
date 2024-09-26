from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

model = ChatOpenAI(model="gpt-4", api_key=api_key)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | model

history = ChatMessageHistory()
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: history,
    input_messages_key="input",
    history_messages_key="history"
)

# Define a session_id
session_id = "user_123"  # This could be any unique identifier for the conversation

# Use the chain with a session_id
print(chain_with_history.invoke(
    {"input": "Hi, my name is Bob"},
    config={"configurable": {"session_id": session_id}}
).content)

print(chain_with_history.invoke(
    {"input": "What's my name?"},
    config={"configurable": {"session_id": session_id}}
).content)
