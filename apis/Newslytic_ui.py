import streamlit as st
from utils.api_client import process_news, classify_headline, summarize_article, health_check

# --- CONSTANTS ---
# Fixed summary lengths for the API calls (used to replace the removed sliders)
DEFAULT_MIN_LEN = 50
DEFAULT_MAX_LEN = 150

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="Newslytic — Intelligent News Analysis",
    page_icon="🗞️",
    layout="wide"
)

# Apply custom styles for better visual appeal
st.markdown("""
<style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    /* Hide the default Streamlit footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# SIDEBAR / SETTINGS (Now Collapsible)
# -----------------------------
st.sidebar.header("⚙️ App Configuration")

# 1. Collapsible Settings Section (for Analysis Mode)
with st.sidebar.expander("🔬 Analysis Settings", expanded=True):
    # Keep the mode selection
    mode = st.radio(
        "Select Processing Mode",
        ["Full Process (Classify + Summarize)", "Classify Only", "Summarize Only"],
        index=0, # Default to Full Process
    )
    # Min/Max Summary Length sliders are removed from the UI.

st.sidebar.markdown("---")

# 2. API Health Check Status
if st.sidebar.button("🔌 Check API Status"):
    try:
        status = health_check()
        st.sidebar.success(f"✅ API is **{status['status'].upper()}** on **{status['device']}**")
    except Exception as e:
        st.sidebar.error(f"❌ API Unreachable. Error: {e}")

# -----------------------------
# MAIN PAGE (Beautification)
# -----------------------------
st.title("📰 Newslytic — Intelligent News Analysis")
st.markdown("""
Analyze news content by **classifying headlines** (Crime or Non-Crime) and generating a **concise summary**.
""")

# Input Container
with st.container(border=True):
    st.markdown('<p class="big-font">Input Content</p>', unsafe_allow_html=True)
    
    # Using specific labels for better accessibility, though keeping the placeholder visual focus
    headline = st.text_input("Headline", placeholder="e.g. Nigeria’s Central Bank Introduces New Policy...", key="headline_input")
    article = st.text_area("Full Article", height=250, placeholder="Paste the full article text here...", key="article_input")


if st.button("🚀 Run Newslytic Analysis", type="primary", use_container_width=True):
    
    # -----------------------------
    # VALIDATION
    # -----------------------------
    if not headline.strip() and mode != "Summarize Only":
        st.error("🛑 Please enter a headline when not in 'Summarize Only' mode.")
        st.stop()
    if not article.strip() and mode != "Classify Only":
        st.error("🛑 Please enter the article body when not in 'Classify Only' mode.")
        st.stop()
        
    # -----------------------------
    # PROCESSING LOGIC
    # -----------------------------
    try:
        with st.spinner(f"Processing in **{mode}** mode... ⏳"):
            
            st.markdown("## ✨ Analysis Results")
            
            if mode == "Full Process (Classify + Summarize)":
                # Use hardcoded lengths
                result = process_news(headline, article, DEFAULT_MIN_LEN, DEFAULT_MAX_LEN)
                data = result["results"]
                
                # Use columns to neatly separate classification and summary
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.success("✅ Classification & Summarization Complete!")
                    # 3. Rename Predicted Class to Headline Type. 4. Hide Confidence.
                    st.metric(label="Headline Type", value=data['predicted_class'].capitalize())
                    st.caption(f"Classification based on the headline.")
                
                with col2:
                    st.subheader("✂️ Article Summary")
                    st.info(data["summary"])


            elif mode == "Classify Only":
                result = classify_headline(headline)
                
                st.success("✅ Classification Complete!")
                
                # 3. Rename Predicted Class to Headline Type. 4. Hide Confidence.
                st.metric(label="Headline Type", value=result['predicted_class'].capitalize())
                st.caption("Classification based on the headline.")


            elif mode == "Summarize Only":
                # Use hardcoded lengths
                result = summarize_article(article, DEFAULT_MIN_LEN, DEFAULT_MAX_LEN)
                
                st.success("✅ Summarization Complete!")
                
                st.markdown("### ✂️ Article Summary")
                st.info(result["summary"])
                
                # Displaying summary length information nicely
                col_len1, col_len2, col_len3 = st.columns(3)
                
                original_len = int(result.get('article_length', 1))
                summary_len = int(result.get('summary_length', 1))
                
                reduction = f"{100 - (summary_len / original_len * 100):.1f}%" if original_len > 0 else "0%"

                col_len1.metric("Original Length (words)", original_len)
                col_len2.metric("Summary Length (words)", summary_len)
                col_len3.metric("Length Reduction", reduction)

    except Exception as e:
        # Generic error handling for API issues
        st.error(f"⚠️ An API error occurred during processing. Please check the API status in the sidebar. Details: {e}")

st.markdown("---")
st.markdown("<sub>*Newslytic utilizes an external API for AI classification and summarization. Fixed summary length range: {DEFAULT_MIN_LEN}-{DEFAULT_MAX_LEN} words.*</sub>", unsafe_allow_html=True)
