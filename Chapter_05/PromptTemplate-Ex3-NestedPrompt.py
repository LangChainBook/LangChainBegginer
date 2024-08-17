from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

# Define a complex prompt template
prompt_template = ChatPromptTemplate.from_template("""
Write a {length} story about: {content}
""")

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4", api_key=api_key)

# Format the template with specific values
prompt = prompt_template.format(
    length="{sentences}-sentence",
    content="The hometown of the legendary scientist, {who}"
)

# Generate the response
print(llm.invoke(prompt.format(
    who="Albert Einstein",
    sentences=3
    )).content)
print("-------------------")
print(llm.invoke(prompt.format(
    who="Issac Newton",
    sentences=5
    )).content)





