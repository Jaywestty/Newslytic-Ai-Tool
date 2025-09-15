import streamlit as st
import requests

st.set_page_config(page_title="News Classifier & Summarizer", layout="wide")

st.title("📰 News Classifier & Summarizer")
st.write("Paste a news article below to classify and generate a summary.")

headline = st.text_input("Enter Headline")
article = st.text_area("Paste Full Article", height=300)

if st.button("Process Article"):
    if headline.strip() and article.strip():
        payload = {
            "headline": headline.strip(),
            "article": article.strip()
        }

        try:
            response = requests.post("http://127.0.0.1:8000/process", json=payload)

            if response.status_code == 200:
                result = response.json()

                st.success("✅ Analysis Complete!")

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Classification Result")
                    st.write(f"**Predicted Class:** {result['predicted_class']}")

                with col2:
                    st.subheader("Summary")
                    st.write(result['summary'])

            else:
                st.error(f"❌ Error {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"Failed to connect to FastAPI backend: {e}")
    else:
        st.warning("Please enter both a headline and article.")
