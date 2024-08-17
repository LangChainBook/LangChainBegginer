import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.tools import Tool

# Load environment variables
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

# Define a web scraping tool function
def web_scraper(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        return {"content": soup.get_text()}
    except Exception as e:
        return {"content": str(e)}

# Define the web scraping tool
web_scraping_tool = Tool(
    name="web_scraper",
    func=web_scraper,
    description="A tool to scrape information from a website"
)
# Example usage
url = "https://www.example.com"

# Run the tool as an Agent would and print the response
print(web_scraping_tool.invoke(url)['content'])


