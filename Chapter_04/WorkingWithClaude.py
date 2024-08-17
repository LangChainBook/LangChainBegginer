from langchain.chat_models.anthropic import ChatAnthropic
import os
from dotenv import load_dotenv
load_dotenv() # Load environment variables
# Get Claude API key from environment variables
claude_api_key = os.getenv("CLAUDE_API_KEY")

# Create a chat model instance
chat_model = ChatAnthropic(api_key=claude_api_key, model_name="claude-1")

# Send a message to the model
response = chat_model.invoke("What is the weather like today?")

# Print the response
print(response['content'])



