# utils/api_client.py
import requests
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
API_URL = os.getenv("API_URL")

def process_from_url(url: str, min_len=50, max_len=150):
    """Send article URL to backend API for processing"""
    response = requests.post(
        f"{API_URL}/from_url",
        params={
            "url": url,
            "min_length": min_len,
            "max_length": max_len
        }
    )
    response.raise_for_status()
    return response.json()

def health_check():
    """Check backend API health"""
    response = requests.get(f"{API_URL}/health")
    response.raise_for_status()
    return response.json()
