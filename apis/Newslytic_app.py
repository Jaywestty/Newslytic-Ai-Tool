# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Union, List
import logging
from src.newslytic_pipeline import NewslyticProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Newslytic API",
    description="News Classification & Summarization API",
    version="1.0.0"
)

# Initialize Newslytic processor (loads both classifier and summarizer)
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
        "summarizer_loaded": processor.summarizer is not None,
        "device": str(processor.device)
    }


@app.post("/process")
async def process_news(request: NewsRequest):
    """
    Process one or multiple news articles — classify and summarize.
    """
    try:
        results = processor.process(
            headlines=request.headline,
            articles=request.body,
            min_len=request.min_length,
            max_len=request.max_length
        )

        return {
            "results": results,
            "count": 1 if isinstance(results, dict) else len(results),
            "type": "single" if isinstance(results, dict) else "batch"
        }

    except ValueError as ve:
        logger.error(f"Validation error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify")
async def classify_headline(headline: str):
    """Classify a single headline only (no summarization)"""
    try:
        if not headline or not headline.strip():
            raise HTTPException(status_code=400, detail="Headline cannot be empty")
        
        predicted_class, confidence = processor.classify_headline(headline)
        confidence_str = f"{confidence:.2%}" if confidence is not None else "N/A"
        
        return {
            "headline": headline,
            "predicted_class": predicted_class,
            "confidence": confidence_str
        }
    except Exception as e:
        logger.exception(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summarize")
async def summarize_article(
    article: str,
    min_length: int = 50,
    max_length: int = 150
):
    """Summarize an article body only (no classification)"""
    try:
        if not article or not article.strip():
            raise HTTPException(status_code=400, detail="Article cannot be empty")
        
        cleaned = processor.clean_and_merge_article(article)
        if not cleaned or len(cleaned) < 20:
            raise HTTPException(
                status_code=400,
                detail="Article too short or invalid after cleaning"
            )
        
        summary = processor.abstractive_summary(
            cleaned,
            min_len=min_length,
            max_len=max_length
        )
        
        return {
            "article_length": len(article),
            "cleaned_length": len(cleaned),
            "summary": summary,
            "summary_length": len(summary)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Summarization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)