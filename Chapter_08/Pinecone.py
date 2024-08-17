from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

# Load environment variables from a .env file
load_dotenv()

# Get Pinecone and OpenAI API keys from the environment variables
api_key = os.getenv('PINECONE_API_KEY')
oapi_key = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(api_key=oapi_key) # embedings for the text
pinecone = Pinecone(api_key=api_key) # Initialize Pinecone with the API key

index_name = "langchain-book" # Define the name of the Pinecone index

# Attempt to create a new Pinecone index
try:
    # Create the index with a specified dimension and serverless specifications
    pinecone.create_index(index_name,dimension=1536,
        spec=ServerlessSpec(cloud='aws', region='us-east-1'))
except Exception as e:
    # If the index already exists or an error occurs, handle it
    if pinecone.Index(index_name):
        print(f"Index {index_name} already exists")
        index = pinecone.Index(index_name)
    else:
        print(f"Index {index_name} could not be created")
        print(e)
        pass
finally:
    # Print a message indicating that the index has been created
    print(f"Index {index_name} aquired")

# Connect to the existing or newly created index
index = pinecone.Index(index_name)
# Load documents from a directory
loader = DirectoryLoader("./Chapter_7/documents/")
documents = loader.load() # Load the documents
# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=500)
chunks = text_splitter.split_documents(documents)
PineconeVectorStore.from_documents(chunks,
    index_name="langchain-book",embedding=embeddings )

# Example usage
query = "Explain the dolphins behavior in the wild"
doc_search=PineconeVectorStore.from_existing_index(index_name, embeddings)
results = doc_search.similarity_search(query=query)
# Print the results
for i, result in enumerate(results):
    print(f"Result {i+1}:\n{result}\n")




