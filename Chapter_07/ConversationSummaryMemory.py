from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain

llm = ChatOpenAI(temperature=1)
chain = ConversationChain(
    llm=llm,
    verbose=True,
    memory=ConversationSummaryMemory (llm=llm))

chain.invoke(input="I am writting a book about mathematics, can you explain what is mathematics?")
chain.invoke(input="My name is Karel,what is the book I am writing about ?")

# Answering the subsequent prompt uses memory.
result=chain.invoke(input="What did we talked about?")
print(result["response"])


