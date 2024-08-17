from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
# Initialize the LLM
llm = ChatOpenAI(model="gpt-4", api_key=api_key)

# Define a simple prompt template
template = ChatPromptTemplate.from_template("""
Human: What is the capital of {place}?
""")
# Generate the final prompt
input_variables = {"place": "California"}

# Send the prompt to the model and get the response
formatted_prompt = template.invoke(input=input_variables)
response = llm.invoke(formatted_prompt)
print(response.content)




