from langchain_core.prompts import ChatPromptTemplate

# Define a simple prompt template
template = ChatPromptTemplate.from_template("""
Human: What is the capital of {place}?
AI: The capital of {place} is {capital}.
""")

# Format the template with specific values
prompt = template.format(place="California", capital="Sacramento")

print(prompt)
