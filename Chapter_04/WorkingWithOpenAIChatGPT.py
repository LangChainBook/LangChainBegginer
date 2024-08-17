from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv() # Load environment variables
# Get OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Create a chat model instance
chat_model = ChatOpenAI(api_key=openai_api_key, model_name="gpt-4")

# Send a message to the model
response = chat_model.invoke("Hello, how are you?")

# Print the response
print(response.content)
