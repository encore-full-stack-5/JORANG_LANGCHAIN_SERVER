import os
import logging
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from fastapi import FastAPI, HTTPException, Body
import asyncio
from fastapi.middleware.cors import CORSMiddleware


from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import WebBaseLoader

# load_dotenv()

# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_API_KEY = "AIzaSyDaQ65xxydMjda790HrSSj6VfpmtpNSQKM"

async def initialize_model():
    model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)   
    if model.async_client is None:
        raise ValueError("ChatGoogleGenerativeAI async_client is not initialized")
    return model
model = asyncio.run(initialize_model())

loader = WebBaseLoader("https://www.originaltravel.co.uk/travel-blog/36-hours-in-dubrovnik")
docs = loader.load()

# prompt_template = PromptTemplate.from_template("Extract country, date and brief review of author from {docs} that the author traveled. Show it in a form like Country: Korea\nDate: 2023-09-22\nReview: It was a great trip and beatiful nature in Korea")

# parser = StrOutputParser()

chain = load_summarize_chain(model, chain_type="stuff")

result = chain.invoke(docs)

print(result["output_text"])

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
@app.post("/summary")
async def stream_log(data: dict = Body(...)):
    try:
        diary = data.get("diary")
        if not diary:
            # raise HTTPException(status_code=400, detail="Missing 'place' in request body")
            result = "여행기를 작성하지 않아 요약할 수 없습니다."
        result = chain.invoke({"diary": diary})
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8006)