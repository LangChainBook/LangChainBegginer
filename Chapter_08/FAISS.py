from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
# Load environment variables from a .env file
load_dotenv()

# Retrieve the OpenAI API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# Load documents from a directory
loader = DirectoryLoader("./Chapter_7/documents/")
documents = loader.load()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=500
)
chunks = text_splitter.split_documents(documents)

# Create embeddings for each chunk
embeddings = OpenAIEmbeddings(api_key=api_key)

# Initialize FAISS index
index = FAISS.from_documents(chunks, embeddings)

# Function to search the FAISS index
def search_index(query, k=5):
    # Perform a similarity search on the index
    results = index.similarity_search(query, k=k)
    return results

# Example usage
query = "Explain the dolphins behavior in the wild"
results = search_index(query)

# Print the results
for i, result in enumerate(results):
    print(f"Result {i+1}:\n{result}\n")



