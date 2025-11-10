# utils/api_client.py
import requests
import streamlit as st
from dotenv import load_dotenv
import os


# Try to read from Streamlit secrets (for deployment), fallback to env var (for local)
try:
    API_URL = st.secrets["API_URL"]
except (FileNotFoundError, KeyError):
    API_URL = os.getenv("API_URL", "http://localhost:8000")  # Default to local FastAPI


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
