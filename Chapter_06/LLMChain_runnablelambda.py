from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')


model = ChatOpenAI(model="gpt-4", api_key=api_key)
prompt = ChatPromptTemplate.from_template(
    "Tell me a {adjective} joke about {topic}."
)

uppercase_step = RunnableLambda(lambda x: x.upper())
chain = prompt | model | StrOutputParser() | uppercase_step

result = chain.invoke({
    "adjective": "funny", 
    "topic": "programming"
})

print(result)







