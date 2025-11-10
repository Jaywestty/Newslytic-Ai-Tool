# app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Union, List
import logging
import os
from dotenv import load_dotenv
from src.newslytic_pipeline import NewslyticProcessor

# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Newslytic API",
    description="News Classification & Summarization API",
    version="1.0.0"
)

# CORS Middleware - allows any frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Newslytic processor (loads classifier and Groq client)
try:
    processor = NewslyticProcessor()
    logger.info("✅ Newslytic processor initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize processor: {e}")
    raise


class NewsRequest(BaseModel):
    headline: Union[str, List[str]] = Field(
        ..., 
        description="Single headline string or list of headlines"
    )
    body: Union[str, List[str]] = Field(
        ..., 
        description="Single article body string or list of article bodies"
    )
    min_length: int = Field(
        default=50, 
        ge=10, 
        le=200,
        description="Minimum summary length"
    )
    max_length: int = Field(
        default=150, 
        ge=50, 
        le=500,
        description="Maximum summary length"
    )

    @field_validator('headline', 'body')
    @classmethod
    def validate_not_empty(cls, v):
        if isinstance(v, str):
            if not v.strip():
                raise ValueError("Text cannot be empty")
        elif isinstance(v, list):
            if len(v) == 0:
                raise ValueError("List cannot be empty")
            if any(not item.strip() for item in v):
                raise ValueError("List contains empty strings")
        return v

    @model_validator(mode='after')
    def validate_matching_lengths(self):
        headline = self.headline
        body = self.body
        
        # Check if both are lists and have matching lengths
        if isinstance(headline, list) and isinstance(body, list):
            if len(headline) != len(body):
                raise ValueError(
                    f"Number of headlines ({len(headline)}) must match "
                    f"number of article bodies ({len(body)})"
                )
        # Check if types match (both single or both batch)
        elif type(headline) != type(body):
            raise ValueError(
                "headline and body must both be strings or both be lists"
            )
        return self


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Newslytic API is running 🚀",
        "status": "healthy",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "classifier_loaded": processor.classifier is not None,
        "groq_client_loaded": processor.groq_client is not None,
        "using_api_summarization": True
    }


@app.post("/analyze")
async def analyze_news(request: NewsRequest):
    """
    Analyze news article(s) - classify headline and summarize body.
    Supports both single article and batch processing.
    """
    try:
        result = processor.process(
            headlines=request.headline,
            articles=request.body,
            min_len=request.min_length,
            max_len=request.max_length
        )
        return {
            "results": result,
            "status": "success"
        }
    except Exception as e:
        logger.exception(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/from_url")
async def process_from_url(
    url: str,
    min_length: int = 50,
    max_length: int = 150
):
    """
    Process a single article by providing its URL directly.
    Extracts headline + body, then runs the full pipeline.
    """
    try:
        if not url.strip():
            raise HTTPException(status_code=400, detail="URL cannot be empty")

        headline, article = processor.extract_from_url(url)
        result = processor.process_single(headline, article, min_length, max_length)

        return {
            "url": url,
            "headline": headline,
            "predicted_class": result["predicted_class"],
            "confidence": result.get("confidence", "N/A"),
            "summary": result["summary"],
            "status": "success"
        }

    except Exception as e:
        logger.exception(f"Error processing URL: {e}")
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)