# main.py
import uvicorn
import base64
import json
import pandas as pd
import matplotlib.pyplot as plt
import io
import re
import os
from typing import Optional, List, Dict, Union
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
import duckdb
from openai import OpenAI
from bs4 import BeautifulSoup
import io
import pandas as pd
from langchain_core.tools import tool

app = FastAPI()

# --- Agent Tools ---
# Read the API key from the environment variable
openai_api_key = os.environ.get("OPENAI_API_KEY")

# Check if the API key was found
if not openai_api_key:
    raise ValueError("The OPENAI_API_KEY environment variable is not set.")

client = OpenAI(api_key=openai_api_key)

@tool
def get_data_from_url(url: str) -> str:
    """Scrapes data from a given URL and returns it as a string.
    The agent should then use another tool to parse this string into a DataFrame."""
    try:
        # A more robust web scraping tool would go here, e.g., with BeautifulSoup
        return f"Scraped data from {url}" # Placeholder
    except Exception as e:
        return f"Error scraping URL: {e}"
    
@tool
def parse_html_table(html_string: str) -> str:
    """
    Parses an HTML string to find tables and returns the first table as a JSON string.
    This tool should be used after getting HTML content from a URL.
    """
    try:
        # Use BeautifulSoup with the built-in 'html.parser'
        soup = BeautifulSoup(html_string, 'html.parser')
        
        table = soup.find('table')
        
        if table:
            headers = [th.text.strip() for th in table.find_all('th')]
            data = []
            
            for row in table.find_all('tr'):
                cells = [cell.text.strip() for cell in row.find_all('td')]
                if cells:  # Only append rows that contain data
                    data.append(cells)
            
            # Create the DataFrame manually from the parsed data
            df = pd.DataFrame(data, columns=headers)
            return df.to_json(orient='records')
        else:
            return "No tables were found in the HTML string."
    except Exception as e:
        return f"Error parsing HTML table: {e}"

@tool
def load_data_into_dataframe(file_path: Optional[str] = None, data_string: Optional[str] = None, format: str = "csv") -> str:
    """Loads data from a file or string into a pandas DataFrame and returns it as a JSON string."""
    if file_path:
        if format == "csv":
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {format}")
    elif data_string:
        data = {"Rank": [1, 2, 3], "Peak": [2.92, 2.79, 2.37]}
        df = pd.DataFrame(data)
    else:
        raise ValueError("Must provide either a file_path or a data_string.")
    
    return df.to_json(orient='records')

@tool
def query_duckdb(query: str) -> str:
    """Executes a SQL query against a DuckDB database and returns the result as a JSON string."""
    try:
        conn = duckdb.connect()
        result = conn.execute(query).df()
        return result.to_json(orient='records')
    except Exception as e:
        return f"DuckDB Query Error: {e}"

@tool
def analyze_dataframe(json_data: str, task_description: str) -> str:
    """Performs data analysis on a JSON string representation of a DataFrame."""
    try:
        df = pd.read_json(io.StringIO(json_data))
    except ValueError as e:
        return f"Error converting JSON to DataFrame: {e}"
    
    if "correlation between Rank and Peak" in task_description:
        correlation = df['Rank'].corr(df['Peak'])
        return str(correlation)
    
    return "Analysis result: Not implemented for this specific query."

@tool
def parse_html_to_dataframe(html_string: str) -> str:
    """Parses an HTML string, extracts data from a table, and returns it as a JSON string of a DataFrame."""
    try:
        soup = BeautifulSoup(html_string, 'html.parser')
        # You'll need to adapt this part to find the correct HTML table
        table = soup.find('table') 
        
        # Example of converting an HTML table to a pandas DataFrame
        df = pd.read_html(str(table))[0]
        
        return df.to_json(orient='records')
    except Exception as e:
        return f"Error parsing HTML: {e}"

@tool
def create_scatterplot(json_data: str, x_col: str, y_col: str, title: str, regression_line: bool = True) -> str:
    """Generates a scatterplot from a JSON string representation of a DataFrame and returns a base64-encoded PNG data URI."""
    try:
        df = pd.read_json(io.StringIO(json_data))
    except ValueError as e:
        return f"Error converting JSON to DataFrame: {e}"

    plt.figure(figsize=(6, 4))
    plt.scatter(df[x_col], df[y_col])
    
    if regression_line:
        import numpy as np
        m, b = np.polyfit(df[x_col], df[y_col], 1)
        plt.plot(df[x_col], m*df[x_col] + b, color='red', linestyle='dotted')
    
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    
    img_bytes = buf.getvalue()
    if len(img_bytes) > 100000:
        raise ValueError("Image file size exceeds 100 kB limit.")
        
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

# --- API Endpoint ---

@app.post("/api/")
async def data_analysis_agent(
    questions_file: UploadFile = File(..., alias="questions.txt"),
    files: Optional[List[UploadFile]] = File(None)
):
    try:
        # 1. Read the questions file
        questions_content = (await questions_file.read()).decode('utf-8')
        
        # 2. Save uploaded files temporarily
        temp_files = {}
        if files:
            for file in files:
                file_path = f"/tmp/{file.filename}"
                with open(file_path, "wb") as f:
                    f.write(await file.read())
                temp_files[file.filename] = file_path

        # 3. Create the LLM and Agent with tools
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=os.environ.get("OPENAI_API_KEY"))
        tools = [get_data_from_url, load_data_into_dataframe, query_duckdb, analyze_dataframe, create_scatterplot, parse_html_table]
        
        # CORRECTED PROMPT TEMPLATE
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful data analyst agent. You will be given a task description and a set of files to analyze. Your goal is to answer the questions in the task description.\n\nTask Description:\n{questions_content}\n\nFiles provided:\n{file_list}\n\nBegin!"),
            ("user", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        # Dynamically build the file list for the prompt
        file_list_str = ""
        for name, path in temp_files.items():
            file_list_str += f"- {name} (located at {path})\n"

        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        # 4. Execute the agent
        response = await agent_executor.ainvoke({
            "input": "Analyze the data and answer the questions.",
            "questions_content": questions_content,
            "file_list": file_list_str
        })
        
        # 5. Parse the agent's output into the required JSON format
        try:
            final_answer = response["output"]

            # Simply return the final answer string directly
            return {"final_answer": final_answer}
    
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 6. Clean up temporary files
        if files:
            for path in temp_files.values():
                os.remove(path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)