from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

model = ChatOpenAI(model="gpt-4", api_key=api_key)
prompt = ChatPromptTemplate.from_template(
    "Tell me a {adjective} joke about {topic}."
)

chain = prompt | model | StrOutputParser()

result = chain.invoke({
    "adjective": "funny", 
    "topic": "programming"
})

print(result)

