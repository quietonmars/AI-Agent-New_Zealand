from click import prompt
from dotenv import load_dotenv
import os
import pandas as pd
from simple_pandas_engine import SimplePandasEngine

from prompts import new_prompt, instruction_str, context
from note_engine import note_engine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.agent.openai import OpenAIAgent  # Only this import
from llama_index.llms.openai import OpenAI
from pdf import nz_engine

load_dotenv()

# Configure for DeepSeek using regular OpenAI class
llm = OpenAI(
    model="gpt-4o-mini",  # Use recognized name for validation
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    api_base="https://api.deepseek.com/v1",
    temperature=0,
    max_tokens=4096,
    additional_kwargs={"model": "deepseek-chat"}  # Actual model for DeepSeek
)

population_path = os.path.join("data", "WorldPopulation2023.csv")
population_df = pd.read_csv(population_path)

population_query_engine = SimplePandasEngine(
    df=population_df,
    verbose=True,
    instruction_str=instruction_str,
    llm=llm
)

tools = [
    note_engine,
    QueryEngineTool(
        query_engine=population_query_engine,
        metadata=ToolMetadata(
            name="population_data",
            description="This gives information about the world population and world demographics",
        ),
    ),
    QueryEngineTool(
        query_engine=nz_engine,
        metadata=ToolMetadata(
            name="nz_data",
            description="This gives detailed information about New Zealand the country",
        ),
    )
]

# Use OpenAIAgent instead
agent = OpenAIAgent.from_tools(
    tools=tools,
    llm=llm,
    verbose=True,
    system_prompt=context
)

while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    result = agent.chat(prompt)
    print(result)