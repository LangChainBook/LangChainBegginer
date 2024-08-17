from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

template = """ # Define a more complex prompt template
Role: {role}
User: {user}
Task: {task}
"""
prompt_template = ChatPromptTemplate.from_template(template)
# Create a chat model instance
chat_model = ChatOpenAI(model="gpt-4", api_key=api_key)

input_variables = { # Provide input variables
    "role": "Teacher",
    "user": "Alice",
    "task": "Explain the theory of relativity."
}
# Generate the final prompt
formatted_prompt = prompt_template.invoke(input_variables)
# Send the prompt to the model and get the response
response = chat_model.invoke(formatted_prompt)
print(response.content)


