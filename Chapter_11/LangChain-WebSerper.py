from langchain_community.utilities import GoogleSerperAPIWrapper

# Define the Serper search tool
search_tool = GoogleSerperAPIWrapper()

# Example usage
query = "What is the hometown of the reigning men's U.S. Open champion?"
response = search_tool.invoke(query)

# Print the response
print(response)
