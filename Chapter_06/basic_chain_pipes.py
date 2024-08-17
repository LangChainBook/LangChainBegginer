from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

# Define the prompt template
template = "Tell me a joke about {topic}."
prompt_template = ChatPromptTemplate.from_template(template)

# Create a chat model instance
chat_model = ChatOpenAI(model="gpt-4", api_key=api_key)

# Define an output parser to extract the content
output_parser = StrOutputParser()

# Build the chain
chain = prompt_template | chat_model | output_parser

# Provide input variables
input_variables = {"topic": "dogs"}

# Run the chain
result = chain.invoke(input_variables)

print(result) # Print the result


