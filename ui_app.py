import streamlit as st
import requests
import time

# Page configuration
st.set_page_config(
    page_title="News Classifier & Summarizer", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        background-color: #f8f9fa;
    }
    .classification-crime {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
    }
    .classification-non-crime {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">📰 News Classifier & Summarizer</h1>', unsafe_allow_html=True)
st.markdown("**Classify news headlines and generate intelligent summaries using AI**")

# API configuration
API_URL = "http://127.0.0.1:8000"

# Sidebar for configuration (optional)
with st.sidebar:
    st.header("⚙️ Settings")
    api_url = st.text_input("API URL", value=API_URL, help="FastAPI backend URL")
    st.markdown("---")
    st.markdown("### About")
    st.info("This app uses a Bernoulli classifier for crime detection and BART for summarization.")

# Main input section
st.header("📝 Input Your News Article")

headline = st.text_input(
    "📄 Enter News Headline", 
    placeholder="e.g., Local bank robbed in downtown area...",
    help="Enter the main headline of your news article"
)

article = st.text_area(
    "📰 Paste Full Article", 
    height=250,
    placeholder="Paste the complete news article content here...",
    help="Paste the full body of the news article for summarization"
)

# Process button with better spacing
st.markdown("---")
process_col1, process_col2, process_col3 = st.columns([1, 2, 1])
with process_col2:
    process_button = st.button("🚀 Process Article", type="primary", use_container_width=True)

# Processing logic
if process_button:
    if headline.strip() and article.strip():
        # Input validation
        if len(headline.strip()) < 5:
            st.warning("⚠️ Headline seems too short. Please enter a meaningful headline.")
        elif len(article.strip()) < 50:
            st.warning("⚠️ Article seems too short. Please enter a complete article.")
        else:
            # Show processing indicator
            with st.spinner("🔄 Processing your article... This may take a few seconds."):
                payload = {
                    "headline": headline.strip(),
                    "article": article.strip()
                }

                try:
                    # Make API request
                    response = requests.post(f"{api_url}/process", json=payload, timeout=30)

                    if response.status_code == 200:
                        result = response.json()
                        
                        # Success message
                        st.success("✅ Analysis Complete!")
                        
                        # Results section
                        st.markdown("---")
                        st.header("📊 Results")

                        # Classification and Summary in columns
                        result_col1, result_col2 = st.columns([1, 2])

                        with result_col1:
                            st.subheader("🏷️ Classification")
                            
                            # Style based on classification result
                            classification = result['predicted_class']
                            confidence = result.get('confidence', 'N/A')
                            
                            if 'crime' in classification.lower():
                                class_style = "classification-crime"
                                class_emoji = "🚨"
                            else:
                                class_style = "classification-non-crime" 
                                class_emoji = "✅"
                            
                            st.markdown(f"""
                            <div class="result-box {class_style}">
                                <h3>{class_emoji} {classification}</h3>
                                <p><strong>Confidence:</strong> {confidence}</p>
                            </div>
                            """, unsafe_allow_html=True)

                        with result_col2:
                            st.subheader("📄 Summary")
                            summary = result['summary']
                            
                            st.markdown(f"""
                            <div class="result-box">
                                <p style="font-size: 1.1em; line-height: 1.6;">{summary}</p>
                            </div>
                            """, unsafe_allow_html=True)

                        # Additional info section
                        with st.expander("📈 Processing Details"):
                            st.json({
                                "Original Headline": result['headline'],
                                "Classification": result['predicted_class'],
                                "Confidence": confidence,
                                "Processing Status": result.get('status', 'success'),
                                "Original Article Length": f"{len(article)} characters",
                                "Summary Length": f"{len(summary)} characters"
                            })

                    else:
                        st.error(f"❌ API Error {response.status_code}")
                        try:
                            error_detail = response.json().get('detail', 'Unknown error')
                            st.error(f"Details: {error_detail}")
                        except:
                            st.error(f"Response: {response.text}")
                            
                except requests.exceptions.Timeout:
                    st.error("⏰ Request timed out. The API might be processing a large article.")
                except requests.exceptions.ConnectionError:
                    st.error("🔌 Cannot connect to the API. Make sure the FastAPI server is running.")
                    st.info("To start the server, run: `python your_api_file.py`")
                except Exception as e:
                    st.error(f"❌ Unexpected error: {str(e)}")

    else:
        st.warning("⚠️ Please enter both a headline and article before processing.")

# Footer with example
st.markdown("---")
with st.expander("💡 Example Usage"):
    st.markdown("""
    **Sample Headline:** `Local bank robbed in downtown area, suspect still at large`
    
    **Sample Article:** `Police are investigating a bank robbery that occurred this morning at First National Bank on Main Street. The suspect, described as a male in his 30s wearing a black hoodie, demanded cash from tellers before fleeing on foot. No injuries were reported, but the investigation is ongoing. Police are asking anyone with information to come forward.`
    """)

# API status check
if st.checkbox("🔍 Check API Status"):
    try:
        health_response = requests.get(f"{api_url}/health", timeout=5)
        if health_response.status_code == 200:
            health_data = health_response.json()
            st.success("✅ API is healthy and ready")
            st.json(health_data)
        else:
            st.warning("⚠️ API is running but might have issues")
    except:
        st.error("❌ API is not accessible")
        st.info("Make sure your FastAPI server is running on the specified URL.")