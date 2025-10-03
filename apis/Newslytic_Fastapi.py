# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from src.newslytic_pipeline import NewslyticProcessor

app = FastAPI(title="News Classification & Summarization API")

# initialize Newslytic models both classifier and summarizer
processor = NewslyticProcessor()

class NewsRequest(BaseModel):
    headline: str | list[str]
    body: str | list[str]

@app.post("/process")
async def process_news(request: NewsRequest):
    if isinstance(request.headline, str) and isinstance(request.body, str):
        result = processor.process_single(request.headline, request.body)
    else:
        result = processor.process_batch(request.headline, request.body)
    return {"results": result}

@app.get("/")
async def root():
    return {"message": "API is running 🚀"}
