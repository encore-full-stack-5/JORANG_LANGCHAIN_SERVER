import os
import logging
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.responses import JSONResponse
import asyncio
from langserve import add_routes
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
# load_dotenv()

# Get the Google API key from environment variables
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_API_KEY = "AIzaSyDaQ65xxydMjda790HrSSj6VfpmtpNSQKM"

# Initialize the ChatGoogleGenerativeAI instance
async def initialize_model():
    model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)   
    return model
model = asyncio.run(initialize_model())

# Define the prompt template and parser
prompt_template = PromptTemplate.from_template("Make me the plan for trip according to following 4 information in Korean. 1. place to visit : {place} 2.date for trip : {date} 3. Who do you go with? : {who} 4. What is your travel style? : {style}")
parser = StrOutputParser()

# Create the chain with the prompt template, LLM, and parser
chain = prompt_template | model | parser

# Initialize the FastAPI app
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
)


# place=""
# date=""
# who=""
# style=""

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # 리액트 앱의 출처
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# # Add the routes to the FastAPI app
# @app.post("plan")
# async def stream_log(data: dict = Body(...)):
#     try:
#         global place 
#         place = data.get("place")
#         return "언제 여행을 떠나고 싶으세요?"
#     except Exception as e:
#         logger.error(f"Error processing request: {e}")
#         raise HTTPException(status_code=500, detail="Internal Server Error")
    

# @app.post("/date")
# async def stream_log(data: dict = Body(...)):
#     try:
#         global date 
#         date = data.get("date")
#         return "누구랑 떠나요?"
#     except Exception as e:
#         logger.error(f"Error processing request: {e}")
#         raise HTTPException(status_code=500, detail="Internal Server Error")

# @app.post("/who")
# async def stream_log(data: dict = Body(...)):
#     try:
#         global who 
#         who = data.get("who")
#         return "여행 스타일을 알려주세요"
#     except Exception as e:
#         logger.error(f"Error processing request: {e}")
#         raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post("/plan")
async def stream_log(data: dict = Body(...)):
    try:
        place = data.get("place")
        date = data.get("date")
        who = data.get("who")
        style = data.get("style")
        result = chain.invoke({"place" : place, "date" : date, "who": who, "style": style})
        return result
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8004)