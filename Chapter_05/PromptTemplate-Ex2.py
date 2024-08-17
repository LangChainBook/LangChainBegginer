from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

# Define the prompt template with a placeholder for the topic
template = "Tell me a joke about {topic}."
prompt_template = ChatPromptTemplate.from_template(template)

# Create a chat model instance
chat_model = ChatOpenAI(model="gpt-4", api_key=api_key)

# Provide values for the input variables
input_variables = {"topic": "cats"}

# Generate the final prompt
formatted_prompt = prompt_template.invoke(input_variables)

# Send the prompt to the model and get the response
response = chat_model.invoke(formatted_prompt)

print(response.content)



