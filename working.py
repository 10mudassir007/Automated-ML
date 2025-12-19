from langchain.agents import create_agent
from dotenv import load_dotenv
import os

from helpers import structure_input, extract_code

load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

with open("prompt.md", "r", encoding="utf-8") as f:
    prompt = f.read()

agent = create_agent(
    model="groq:openai/gpt-oss-120b",
    system_prompt=prompt,
)

def run(path:str):
    structured_input = structure_input(path)
    response = agent.invoke(
        {"messages": [{"role": "user", "content": structured_input}]},
        {"configurable": {"thread_id": "1"}},
    )
    
    code_only = extract_code(response['messages'][-1].content)
    print(code_only)    
    # Execute the code
    exec(code_only, globals())

run(r"F:\Files\Portfolio\AUTOML\automated-ml\diabetes_dataset.csv")