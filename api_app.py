# Import required libraries
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model Loading with Error Handling
def load_models():
    """Load all models at startup with proper error handling."""
    models = {}
    
    try:
        logger.info('Loading Classification Model...')
        models['classifier'] = joblib.load('classifier_model.pkl')
        logger.info('✅ Classification model loaded successfully')
    except Exception as e:
        logger.error(f"❌ Failed to load classification model: {e}")
        raise RuntimeError(f"Classification model loading failed: {e}")

    try:
        logger.info('Loading Summarization Model...')
        MODEL_NAME = "Jayywestty/bart-summarizer-epoch2"
        
        # Try local cache first, then download
        try:
            models['tokenizer'] = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
            models['summarizer'] = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, local_files_only=True)
            logger.info("✅ Loaded summarizer from local cache")
        except:
            models['tokenizer'] = AutoTokenizer.from_pretrained(MODEL_NAME)
            models['summarizer'] = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
            logger.info("⬇️ Downloaded summarizer from Hugging Face")
            
        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        models['summarizer'] = models['summarizer'].to(device)
        logger.info(f"✅ Summarizer loaded on {device}")
        
    except Exception as e:
        logger.error(f"❌ Failed to load summarization model: {e}")
        raise RuntimeError(f"Summarization model loading failed: {e}")
    
    return models

# Load models at startup
try:
    MODELS = load_models()
    classifier = MODELS['classifier']
    tokenizer = MODELS['tokenizer'] 
    summarizer = MODELS['summarizer']
except Exception as e:
    logger.error(f"Critical error during model loading: {e}")
    exit(1)

# Build dynamic label map from classifier
label_map = {i: label for i, label in enumerate(classifier.classes_)}

# Create user-friendly label mapping
friendly_label_map = {
    0: "Non-crime",
    1: "Crime"
}


# Utility Functions
def normalize_raw_text(text: str) -> str:
    """Normalize raw text from user (smart quotes, dashes, extra spaces)."""
    if not text or not text.strip():
        return ""
    
    text = text.replace(""", "\"").replace(""", "\"").replace("'", "'")
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text.strip())
    return text

def clean_and_merge_article(text: str) -> str:
    """Cleans article text by normalizing, fixing spacing, and merging sentences."""
    if not text or not text.strip():
        return ""
        
    text = normalize_raw_text(text)
    text = re.sub(r'[^\w\s\.\,\?\!]', '', text)
    sentences = re.split(r'(?<=[\.\!\?])\s+', text)
    merged = " ".join(s for s in sentences if s)
    return merged

def abstractive_summary(article: str, min_len: int = 50, max_len: int = 150) -> str:
    """Abstractive summary using fine-tuned BART model."""
    try:
        inputs = tokenizer(
            article,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        ).to(summarizer.device)

        with torch.no_grad():
            summary_ids = summarizer.generate(
                **inputs,
                max_length=max_len,
                min_length=min_len,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summary = re.sub(r"\s+", " ", summary.strip())
        return summary
        
    except Exception as e:
        logger.error(f"Abstractive summarization failed: {e}")
        # Fallback: return truncated original text
        return article[:200] + "..." if len(article) > 200 else article

# FastAPI Setup
app = FastAPI(
    title="News Classification + Summarization API",
    description="API that classifies news headlines as crime/non-crime and provides abstractive summaries",
    version="1.0.0"
)

class NewsRequest(BaseModel):
    headline: str = Field(..., min_length=1, max_length=500, description="News headline to classify")
    article: str = Field(..., min_length=10, max_length=10000, description="News article body to summarize")

class NewsResponse(BaseModel):
    headline: str
    predicted_class: str
    confidence: str
    summary: str
    status: str

@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "message": "News Classification + Summarization API is running!",
        "endpoints": {
            "/process": "POST - Process headline and article",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint for deployment monitoring."""
    try:
        # Quick model check
        test_prediction = classifier.predict(["test headline"])[0]
        return {
            "status": "healthy",
            "classification_model": "loaded",
            "summarization_model": "loaded",
            "device": str(summarizer.device)
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.post("/process", response_model=NewsResponse)
def process_news(req: NewsRequest):
    """Process news headline and article."""
    try:
        # Validate inputs
        if not req.headline.strip():
            raise HTTPException(status_code=400, detail="Headline cannot be empty")
        if not req.article.strip():
            raise HTTPException(status_code=400, detail="Article cannot be empty")
        
        # Step 1: Classify headline
        headline_pred = classifier.predict([req.headline])[0]
        
        # Get prediction probabilities for confidence
        try:
            pred_proba = classifier.predict_proba([req.headline])[0]
            confidence = f"{max(pred_proba):.2%}"
        except:
            confidence = "N/A"
        
        # Convert prediction to user-friendly label
        predicted_class = friendly_label_map.get(int(headline_pred), f"Unknown ({headline_pred})")
        
        # Step 2: Clean + Summarize article
        cleaned_article = clean_and_merge_article(req.article)
        
        if not cleaned_article or len(cleaned_article.strip()) < 20:
            raise HTTPException(status_code=400, detail="Article too short or invalid after cleaning")
        
        summary = abstractive_summary(cleaned_article)
        
        return NewsResponse(
            headline=req.headline,
            predicted_class=predicted_class,
            confidence=confidence,
            summary=summary,
            status="success"
        )
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal processing error: {str(e)}")

# Optional: Add middleware for request logging
@app.middleware("http")
async def log_requests(request, call_next):
    """Log all requests for monitoring."""
    logger.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response: {response.status_code}")
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)