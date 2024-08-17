from langchain_community.agent_toolkits import FileManagementToolkit
from tempfile import TemporaryDirectory

from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

# Create a temporary directory
with TemporaryDirectory() as tempdir:
    # Initialize the File Management Toolkit
    tools = FileManagementToolkit(
    root_dir=str(tempdir),
    selected_tools=["read_file", "write_file", "list_directory"],
).get_tools()

# Load the tools we are going to provide to Agent
read_tool, write_tool, list_tool = tools

# Write text to a file in the temporary directory
write_tool.invoke({"file_path": "example.txt", "text": "Hello World!"})

# List the files in the temporary directory
files = list_tool.invoke({})

# Read the text from the file we created in the temporary directory
text = read_tool.invoke({"file_path": "example.txt"})
print(files)
print(text)



