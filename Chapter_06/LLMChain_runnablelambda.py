from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

model = ChatOpenAI()
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







