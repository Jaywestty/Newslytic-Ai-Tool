#import required libraries
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ================================
# Load Models at Startup
# ================================
print('Loading Classification Model...')
classifier = joblib.load('classifier_model.pkl')

# Build dynamic label map from classifier
label_map = {i: label for i, label in enumerate(classifier.classes_)}

print('Loading Summarization Model...')
MODEL_NAME = "Jayywestty/bart-summarizer-epoch2"

try:
    # First try: local cache only
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    summarizer = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, local_files_only=True)
    print("✅ Loaded summarizer from local cache")
except:
    # Fallback: download from HF
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    summarizer = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    print("⬇️ Downloaded summarizer from Hugging Face")

# ================================
# Extractive Summarizer (sumy)
# ================================
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

def extractive_summary(text: str, num_sentences: int = 3) -> str:
    """Extractive summary using TextRank (sumy)."""
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary_sentences = summarizer(parser.document, num_sentences)
    return " ".join(str(sentence) for sentence in summary_sentences)

# ================================
# Utility Functions
# ================================
def normalize_raw_text(text: str) -> str:
    """Normalize raw text from user (smart quotes, dashes, extra spaces)."""
    text = text.replace("“", "\"").replace("”", "\"").replace("’", "'")
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text.strip())
    return text

def clean_and_merge_article(text: str) -> str:
    """Cleans article text by normalizing, fixing spacing, and merging sentences."""
    text = normalize_raw_text(text)
    text = re.sub(r'[^\w\s\.\,\?\!]', '', text)
    sentences = re.split(r'(?<=[\.\!\?])\s+', text)
    merged = " ".join(s for s in sentences if s)
    return merged

def abstractive_summary(article: str, min_len: int = 80, max_len: int = 200) -> str:
    """Abstractive summary using fine-tuned BART model."""
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

def hybrid_summarize(article: str) -> str:
    """Hybrid summarization: Extractive (sumy) → Abstractive (BART)."""
    # Step 1: Extractive summary
    extractive = extractive_summary(article, num_sentences=3)

    # Step 2: Abstractive summary of the extractive result
    final_summary = abstractive_summary(extractive, min_len=50, max_len=120)

    return final_summary

# ================================
# FastAPI Setup
# ================================
app = FastAPI(title="News Classification + Hybrid Summarization API")

class NewsRequest(BaseModel):
    headline: str
    article: str

@app.post("/process")
def process_news(req: NewsRequest):
    # Step 1: Classify headline
    headline_pred = classifier.predict([req.headline])[0]
    label_map = {"0": "Non-crime", "1": "Crime"}
    predicted_class = label_map.get(str(headline_pred), str(headline_pred))

    # Step 2: Clean + Hybrid Summarize article
    cleaned_article = clean_and_merge_article(req.article)
    summary = hybrid_summarize(cleaned_article)

    return {
        "headline": req.headline,
        "predicted_class": predicted_class,
        "summary": summary
    }

# Root route
@app.get("/")
def root():
    return {"message": "News API is running! Use /process with headline + article."}
