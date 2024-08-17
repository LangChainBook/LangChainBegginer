from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

model = ChatOpenAI()
prompt = ChatPromptTemplate.from_template(
    "Tell me a {adjective} joke about {topic}."
)

def route(info):
    if "program" in info:
        return upperlower_step
    else:
        return uppercase_step

def lowerupper (test_str) :
    res = ""
    for idx in range(len(test_str)):
        if not idx % 2:
            res = res + test_str[idx].upper()
        else:
            res = res + test_str[idx].lower()
    return res    
 

upperlower_step = RunnableLambda(lambda x: lowerupper(x))
uppercase_step = RunnableLambda(lambda x: x.upper())

chain = prompt | model | StrOutputParser() | RunnableLambda(route)

result = chain.invoke({
    "adjective": "funny", 
    "topic": "programming"
})
print(result)
print("")
result = chain.invoke({
    "adjective": "funny", 
    "topic": "fishing"
})
print(result)







