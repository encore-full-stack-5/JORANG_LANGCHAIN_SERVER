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
# print(GOOGLE_API_KEY)
# if not GOOGLE_API_KEY:
#     raise ValueError("GOOGLE_API_KEY environment variable is not set")

# Initialize the ChatGoogleGenerativeAI instance


async def initialize_model():
    model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)   
    # if model.async_client is None:
    #     raise ValueError("ChatGoogleGenerativeAI async_client is not initialized")
    return model
model = asyncio.run(initialize_model())

# Define the prompt template and parser
prompt_template = PromptTemplate.from_template("Recommend me 5 landmarks to visit with 3 foods that are most famous in {place} according to following information. Also tell me brief information of the landmarks in maximum 2 sentences per place Separate answer into 2 sections. One is introduction for place, the other is for food. Don't forget to talk in Korean.")
parser = StrOutputParser()

# Create the chain with the prompt template, LLM, and parser
chain = prompt_template | model | parser

# Initialize the FastAPI app
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # 리액트 앱의 출처
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# # Add the routes to the FastAPI app
@app.post("/landmark")
async def stream_log(data: dict = Body(...)):
    try:
        place = data.get("place")
        return chain.invoke({"place" : place})
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8003)