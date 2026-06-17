from pathlib import Path
from langchain.agents import create_agent
from dotenv import load_dotenv
import os
from src.helpers import structure_input, extract_code
import io
import contextlib

load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

# Resolves the absolute path to prompt.md relative to this working.py file
CURRENT_DIR = Path(__file__).resolve().parent
PROMPT_PATH = CURRENT_DIR / "prompt.md"

with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    prompt = f.read()

agent = create_agent(
    model="groq:openai/gpt-oss-120b",
    system_prompt=prompt,
)

def run(path: str):
    structured_input = structure_input(path)
    response = agent.invoke(
        {"messages": [{"role": "user", "content": structured_input}]},
        {"configurable": {"thread_id": "1"}},
    )
    
    code_only = extract_code(response['messages'][-1].content)
    print(code_only)    
    # Execute the code
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        exec(code_only, globals())

    return f.getvalue()

if __name__ == "__main__":
    run("diabetes_dataset.csv")