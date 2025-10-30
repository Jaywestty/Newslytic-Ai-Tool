# streamlit_app.py
import streamlit as st
from utils.api_client import process_from_url, health_check

st.set_page_config(page_title="📰 Newslytic", layout="centered")

st.title("📰 Newslytic")
st.caption("AI-powered news summarizer and classifier — just enter a news URL!")

# Check API health
try:
    health = health_check()
    st.success(f"✅ API Connected | Device: {health['device']}")
except Exception as e:
    st.error(f"⚠️ Failed to connect to API: {e}")

# Input field for URL
url = st.text_input("Enter a news article URL:")

min_len = st.slider("Minimum summary length", 30, 150, 50)
max_len = st.slider("Maximum summary length", 100, 300, 150)

if st.button("Analyze Article"):
    if not url.strip():
        st.warning("Please enter a valid URL.")
    else:
        with st.spinner("Processing article..."):
            try:
                result = process_from_url(url, min_len, max_len)
                st.subheader("🗞 Headline")
                st.write(result["headline"])

                st.subheader("🏷 Category")
                st.success(result["predicted_class"])

                st.subheader("🧠 Summary")
                st.write(result["summary"])

                st.caption(f"Confidence: {result.get('confidence', 'N/A')}")
                st.caption(f"Source: {result['url']}")

            except Exception as e:
                st.error(f"❌ Error: {e}")
