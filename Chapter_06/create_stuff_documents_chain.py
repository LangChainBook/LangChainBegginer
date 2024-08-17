# pip install -U langchain langchain-community

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

prompt = ChatPromptTemplate.from_messages(
    [("system", "What are everyone's favorite colors:\n\n{context}")]
)

llm = ChatOpenAI(model="gpt-4")
chain = create_stuff_documents_chain(llm, prompt)

docs = [
    Document(page_content="Jesse loves red but not yellow"),
    Document(page_content="Jamal loves green but not as much as he loves orange")
]

result = chain.invoke({"context": docs})
print(result)


