import requests
from dotenv import load_dotenv
import os

#call api
load_dotenv()
API_URL = os.getenv("API_URL")

def process_news(headline: str, body: str, min_len: int = 50, max_len: int = 150):
    payload = {
        "headline": headline,
        "body": body,
        "min_length": min_len,
        "max_length": max_len
    }
    response = requests.post(f"{API_URL}/process", json=payload)
    response.raise_for_status()
    return response.json()

def classify_headline(headline: str):
    response = requests.post(f"{API_URL}/classify", params={"headline": headline})
    response.raise_for_status()
    return response.json()

def summarize_article(article: str, min_len: int = 50, max_len: int = 150):
    params = {"article": article, "min_length": min_len, "max_length": max_len}
    response = requests.post(f"{API_URL}/summarize", params=params)
    response.raise_for_status()
    return response.json()

def health_check():
    response = requests.get(f"{API_URL}/health")
    response.raise_for_status()
    return response.json()