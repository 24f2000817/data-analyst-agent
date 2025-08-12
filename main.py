# main.py
import uvicorn
import base64
import json
import pandas as pd
import matplotlib.pyplot as plt
import io
import os
import requests
import numpy as np
from typing import Optional, List, Any
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.exceptions import OutputParserException
from langchain_core.agents import AgentAction, AgentFinish

app = FastAPI()

# --- Agent Tools ---

groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("The GROQ_API_KEY environment variable is not set.")

# A global variable to hold the final answers, bypassing the agent's final text output.
final_answers_container = None

@tool
def scrape_and_parse_url(url: str) -> str:
    """
    Scrapes a webpage, finds the first HTML table, and returns its content as a JSON string.
    This tool should be used for data analysis tasks that require data from a web page.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        try:
            df_list = pd.read_html(response.content, flavor='lxml')
        except ImportError:
            return "Error: Missing dependency 'lxml'. Please install it using 'pip install lxml'."
        
        if not df_list:
            return "No tables were found on the web page."

        df = df_list[0]
        df.columns = [col.replace(" ", "_").replace("$", "").replace(",", "") for col in df.columns]
        
        for col in ['Worldwide_gross', 'Rank', 'Peak', 'Year']:
            if col in df.columns:
                df[col] = df[col].astype(str)
                df[col] = df[col].str.replace(r'[^\d.]', '', regex=True)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(inplace=True)
        return df.to_json(orient='records')
    except requests.exceptions.RequestException as e:
        return f"Error fetching URL: {e}"
    except Exception as e:
        return f"Error parsing HTML table: {e}"

@tool
def create_scatterplot(json_data: str, x_col: str, y_col: str, title: str, regression_line: bool = True) -> str:
    """
    Generates a scatterplot from a JSON string of a DataFrame and returns a base64-encoded PNG data URI.
    The plot will include a dotted red regression line.
    """
    try:
        df = pd.read_json(io.StringIO(json_data))
        
        if x_col in df.columns:
            df[x_col] = pd.to_numeric(df[x_col].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')
        if y_col in df.columns:
            df[y_col] = pd.to_numeric(df[y_col].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')
        df.dropna(subset=[x_col, y_col], inplace=True)

        plt.style.use('dark_background')
        plt.figure(figsize=(6, 4))
        plt.scatter(df[x_col], df[y_col], s=10, color='white')
        
        if regression_line:
            z = np.polyfit(df[x_col], df[y_col], 1)
            p = np.poly1d(z)
            plt.plot(df[x_col], p(df[x_col]), "r:", label="Regression Line")

        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(title)
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        plt.close()

        img_bytes = buf.getvalue()
        if len(img_bytes) > 100000:
            return "Error: Image file size exceeds 100 kB limit."
            
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"

    except Exception as e:
        return f"Error creating scatterplot: {e}"

class FinalAnswerSchema(BaseModel):
    answers: List[str] = Field(..., description="A list of strings, where each string is an answer to a question.")

@tool("provide_final_answers", args_schema=FinalAnswerSchema)
def provide_final_answers(answers: List[str]) -> List[dict]:
    """
    Use this tool to return the final list of answers to the user.
    The argument should be a list of strings, with each string being the answer to one of the questions.
    This tool should be the last step in your process after you have answered all the questions.
    """
    global final_answers_container
    final_answers_container = [{"answer": a} for a in answers]
    # The tool still returns a string to not break the AgentExecutor chain
    return "The final answers have been provided."


# --- API Endpoint ---

@app.post("/api/")
async def data_analysis_agent(
    questions_file: UploadFile = File(..., alias="questions.txt"),
    files: Optional[List[UploadFile]] = File(None)
):
    try:
        global final_answers_container
        final_answers_container = None # Reset the container for each new request
        
        questions_content = (await questions_file.read()).decode('utf-8')
        url = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"

        llm = ChatGroq(model="llama3-8b-8192", temperature=0, groq_api_key=groq_api_key)
        tools = [scrape_and_parse_url, create_scatterplot, provide_final_answers]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", 
             """
             You are a data analyst agent. Your task is to scrape data from a given URL, analyze it, and answer all questions from a provided text file.
             The URL to scrape is: {url}.
             The questions to answer are: {questions_content}.
             
             After you have completed your analysis and have a list of all the answers, you MUST use the 'provide_final_answers' tool to return the results. Do not provide any other output.
             """),
            ("user", "Start the analysis now."),
            ("placeholder", "{agent_scratchpad}"),
        ])

        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=50
        )
        
        # We don't need to return intermediate steps anymore, because the tool saves the output directly.
        agent_executor.invoke({
            "input": "Analyze the data and answer all the questions from the provided task description.",
            "questions_content": questions_content,
            "url": url,
        })
        
        # After the agent finishes, check the global container for the result
        if final_answers_container:
            return JSONResponse(content=final_answers_container)
        
        raise HTTPException(status_code=500, detail="The agent did not return the final answers using the expected tool.")

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")




if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
