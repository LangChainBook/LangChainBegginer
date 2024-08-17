from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

model = ChatOpenAI()

# Chain 1: Generate a recipe
prompt1 = ChatPromptTemplate.from_template("Write a recipe for {dish}:")
chain1 = prompt1 | model

# Chain 2: Generate a grocery list from the recipe
prompt2 = ChatPromptTemplate.from_template(
    "Generate a grocery list for this recipe:\n\n{recipe}")
chain2 = prompt2 | model

# Combine the chains
overall_chain = chain1 | chain2

# Run the chain
print(overall_chain.invoke({"dish": "spaghetti carbonara"}).content)



