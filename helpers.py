import pandas as pd
import io
from contextlib import redirect_stdout
import re

def structure_input(path: str):
  data = pd.read_csv(path)
  buffer = io.StringIO()
  
  with redirect_stdout(buffer):
        data.info()
  if data.shape[1] > 1000:
    task_type = "Regression"
  else:
    task_type = "Classification"

  message = ""
  message += f"Data Path: {path}\n\n"
  message += f"Task type: {task_type}\n\n"
  message += f"Data Samples:\n {data.head().round(2).to_markdown()}\n\n"
  message += f"Data info:\n{buffer.getvalue()}\n\n"
  message += f"Data description:\n{data.describe().round(2).to_markdown()}\n\n"
  message += f"Null Values:\n{data.isnull().sum()}\n\n"
  message += f"Duplicates:\n{int(data.duplicated().sum())}\n\n"

  if task_type == "Classification":
    message += f"Data Distribution: {data[data.columns[-1]].value_counts()}"
  
  return message

def extract_code(markdown_text):
    code_blocks = re.findall(r"```(?:python)?\n(.*?)```", markdown_text, re.DOTALL)
    return "\n\n".join(code_blocks)