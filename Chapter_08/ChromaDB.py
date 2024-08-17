from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import asyncio
import os

# Load environment variables from a .env file
load_dotenv()

# Retrieve the OpenAI API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")

# Load documents from a directory
loader = DirectoryLoader("./Chapter_08/documents/")
documents = loader.load()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=500
)
chunks = text_splitter.split_documents(documents)

# Create embeddings for each chunk
embeddings = OpenAIEmbeddings(api_key=api_key)

# Store embeddings in Chroma vector database
vector_store = Chroma.from_documents(chunks, embeddings)

# Example usage
query = "Explain the dolphins behavior in the wild"
results = asyncio.run(vector_store.asearch(query, k=5,search_type="similarity"))

# Print the results
for i, result in enumerate(results):
    print(f"Result {i+1}:\n{result}\n")


