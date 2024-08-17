from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI()
prompt = ChatPromptTemplate.from_template(
    "Tell me a {adjective} joke about {topic}."
)

chain = prompt | model | StrOutputParser()

result = chain.invoke({
    "adjective": "funny", 
    "topic": "programming"
})

print(result)

