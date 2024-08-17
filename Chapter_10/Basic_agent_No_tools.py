import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

# Initialize the chat model
agent = ChatOpenAI(api_key=api_key, model_name="gpt-4")

# Define a task for the agent
task = "Find the current weather in New York."

# Invoke the agent to perform the task
response = agent.invoke(task)

# Print the response
print(response.content)




