from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
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

# Define an additional step to uppercase the output
uppercase_step = RunnableLambda(lambda x: x.upper())

# Build and extend the chain
chain = prompt_template | chat_model | output_parser | uppercase_step

# Provide input variables
input_variables = {"topic": "computers"}

# Run the chain
result = chain.invoke(input_variables)

print(result) # Print the result



