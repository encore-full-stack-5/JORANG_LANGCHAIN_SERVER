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
landmark_prompt_template = PromptTemplate.from_template("Recommend me 5 landmarks to visit with 3 foods that are most famous in {place} according to following information. Also tell me brief information of the landmarks in maximum 2 sentences per place Separate answer into 2 sections. One is introduction for place, the other is for food. Don't forget to talk in Korean.")
plan_prompt_template = PromptTemplate.from_template("Make me the plan for trip according to following 4 information in Korean. 1. place to visit : {place} 2.date for trip : {date} 3. Who do you go with? : {who} 4. What is your travel style? : {style}")
place_prompt_template = PromptTemplate.from_template("Recommend me 5 cities to visit in {continent} according to this travel style: {style} Tell me brief information of the places in maximum 2 sentences per place. You must speak to me in Korean. Don't ")
mystyle_prompt_template = PromptTemplate.from_template("This is my diaries: {diaries}. It has diaries that I traveled. According to my travel style in my diaries, make me travel plan with 5 places to visit. You don't have to recommend only the countries in my diaries. You can consider the places in the world. When you make the plan, consider this. 1. Is these place good to travel at date you recommended me? Get information about this from my diaries 2. Is this place safe for trip? Don't forget to talk in Korean.")

parser = StrOutputParser()

# Create the chain with the prompt template, LLM, and parser
landmark_chain = landmark_prompt_template | model | parser
plan_chain = plan_prompt_template | model | parser
place_chain = place_prompt_template | model | parser
mystyle_chain = mystyle_prompt_template | model | parser


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
        return landmark_chain.invoke({"place" : place})
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/plan")
async def stream_log(data: dict = Body(...)):
    try:
        place = data.get("place")
        date = data.get("date")
        who = data.get("who")
        style = data.get("style")
        result = plan_chain.invoke({"place" : place, "date" : date, "who": who, "style": style})
        return result
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/place")
async def stream_log(data: dict = Body(...)):
    try:
        style = data.get("style")
        continent = data.get("continent")
        return place_chain.invoke({"continent" : continent, "style" : style})
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/my-style")
async def stream_log(data: dict = Body(...)):
    try:
        diaries = data.get("diaries")
        return mystyle_chain.invoke({"diaries" : diaries})
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)