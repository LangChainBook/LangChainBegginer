import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.agents import create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import AgentExecutor

load_dotenv() # Load environment variables
api_key = os.getenv("OPENAI_API_KEY")

# Replace with your actual API key
os.environ["OPENWEATHERMAP_API_KEY"] = "8e79ddb227a10d2ca60cddc65e2ee34e"

# Initialize the chat model
chat_model = ChatOpenAI(api_key=api_key, model_name="gpt-4")

tools = load_tools(["openweathermap-api"]) # Load the weather tool

# Define the prompt template
template = """
You are a helpful assistant. Use the weather_checker tool to find
the current weather in {location}.
{agent_scratchpad}
"""
prompt_template = ChatPromptTemplate.from_template(template)

# Create the agent with the chat model and weather tool
agent = create_openai_tools_agent(
    llm=chat_model,
    tools=tools,
    prompt=prompt_template
)

# Create an executor for the agent
executor = AgentExecutor(agent=agent, tools=tools)

task = {"location": "New York"} # Define the task for the agent

response = executor.invoke(task) # Invoke the agent to perform the task

# Print the response
print(response['output'])





