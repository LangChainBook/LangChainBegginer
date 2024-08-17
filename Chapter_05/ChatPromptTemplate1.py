from langchain.prompts import ChatPromptTemplate

# Define a chat prompt template
chat_template = ChatPromptTemplate.from_messages([
    ("human", "What is the capital of {country}?"),
    ("ai", "The capital of {country} is {capital}.")
])

# Format the chat messages with specific values
messages = chat_template.format_messages(
    country="Canada",
    capital="Ottawa"
)

print(messages)
