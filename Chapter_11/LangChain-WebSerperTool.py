from langchain_community.agent_toolkits.load_tools import load_tools

tools = load_tools(["google-serper"])

# Example usage with a Tool
query = "What is the hometown of the reigning men's U.S. Open champion?"

# Print the response
print(tools[0].invoke(query))



