# streamlit_app.py
import streamlit as st
from utils.api_client import process_from_url, health_check

# --- Page Config ---
st.set_page_config(
    page_title="📰 Newslytic",
    page_icon="🧠",
    layout="centered",
)

# --- Header Section ---
st.markdown(
    """
    <h1 style='text-align:center; color:#2E4053;'>📰 Newslytic</h1>
    <p style='text-align:center; color:gray;'>
        AI-powered news summarizer and classifier — just enter a news URL below.
    </p>
    <hr style='margin-top: 10px; margin-bottom: 25px;'>
    """,
    unsafe_allow_html=True
)

# --- API Health Check ---
try:
    health_check()
    st.toast("✅ API Connected")
except Exception:
    st.warning("⚠️ Unable to connect to the API. Please ensure it's running.")

# --- Input Section ---
st.markdown("### 🔗 Enter News Article URL")
url = st.text_input(
    "News URL",
    placeholder="https://www.example.com/news-article",
    label_visibility="collapsed"  # 👈 hides the label visually but keeps it accessible
)

# --- Action Button ---
analyze = st.button("✨ Analyze Article", use_container_width=True)

# --- Result Section ---
if analyze:
    if not url.strip():
        st.warning("Please enter a valid news article URL.")
    else:
        with st.spinner("Analyzing and summarizing article..."):
            try:
                # Using default length parameters inside process function
                result = process_from_url(url, None, None)

                # --- ERROR CHECK ---
                if result.get("status") == "error":
                    # Display a user-friendly message with possible reasons
                    st.warning(
                        f"⚠️ Unable to process this article.\n\n"
                        f"Reason: {result.get('message', 'Unknown error occurred.')}\n\n"
                        "This may be because the site is blocking access, the URL is incomplete, "
                        "or the page structure is unsupported. Please try a different news source."
                    )
                else:
                    st.markdown("---")
                    st.markdown("### 🗞 Headline")
                    st.markdown(f"**{result['headline']}**")

                    st.markdown("### 🏷 Category")
                    st.markdown(
                        f"<p style='background-color:#E8F6F3; color:#117864; padding:8px 15px; border-radius:8px; display:inline-block; font-weight:600;'>{result['predicted_class']}</p>",
                        unsafe_allow_html=True
                    )

                    st.markdown("### 🧠 Summary")
                    st.markdown(f"{result['summary']}")

                    st.markdown(
                        f"<p style='color:gray; font-size:0.9em;'>Source: <a href='{result['url']}' target='_blank'>{result['url']}</a></p>",
                        unsafe_allow_html=True
                    )

            except Exception as e:
                st.error(f"❌ Error: {e}")
