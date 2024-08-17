import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

print("Hello, World!")
print(f"API Key: {api_key}")
