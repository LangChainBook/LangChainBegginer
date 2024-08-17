from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain

llm = ChatOpenAI(temperature=0)
chain = ConversationChain(
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory())

chain.invoke(input="The book topic is mathematics, can you explain what is mathematics?")

# Answering the subsequent prompt uses memory.
result=chain.invoke(input="What is the book about?")
print(result["response"])


