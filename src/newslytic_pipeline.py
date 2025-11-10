#%%
# src/newslytic_pipeline.py
import os
import re
import logging
from typing import Optional, Tuple, Dict, Any, List, Union
import joblib
from groq import Groq
from newspaper import Article
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class NewslyticProcessor:
    """
    Class that encapsulates the headline classifier and the API-based summarizer.
    Instantiate once at app startup and reuse for every request.
    """

    def __init__(
        self,
        classifier_path: Optional[str] = None,
        groq_api_key: Optional[str] = None,
    ):
        """
        Args:
            classifier_path: Path to joblib classifier file. If None, tries a sensible default.
            groq_api_key: Groq API key. If None, reads from GROQ_API_KEY env var.
        """
        self.classifier_path = classifier_path or self._default_classifier_path()
        
        # Initialize Groq client
        api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY must be set in environment or passed as argument")
        self.groq_client = Groq(api_key=api_key)

        # placeholders
        self.classifier = None
        self._has_predict_proba = False

        # load classifier
        self._load_classifier()

        # label mapping
        self.label_map = {0: "non-crime", 1: "crime"}

    def _default_classifier_path(self) -> str:
        base = os.path.dirname(os.path.dirname(__file__)) if os.path.dirname(__file__) else os.getcwd()
        return os.path.join(base, "Model", "classifier_model_20251003_154212.pkl")

    def _load_classifier(self):
        """Load joblib classifier with error handling."""
        try:
            logger.info("Loading classifier from %s", self.classifier_path)
            self.classifier = joblib.load(self.classifier_path)
            self._has_predict_proba = hasattr(self.classifier, "predict_proba")
            logger.info("✅ Headline classifier loaded (predict_proba=%s)", self._has_predict_proba)
        except Exception as e:
            logger.exception("Failed to load classifier: %s", e)
            raise RuntimeError(f"Failed to load classifier at {self.classifier_path}: {e}")

    # Text cleaning utilities
    @staticmethod
    def normalize_raw_text(text: str) -> str:
        if not text:
            return ""
        text = text.replace(""", '"').replace(""", '"').replace("'", "'").replace("'", "'")
        text = text.replace("–", "-").replace("—", "-")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def clean_and_merge_article(self, article: str) -> str:
        """Basic cleanup and merging of sentences to prepare for summarization."""
        if not article:
            return ""
        article = self.normalize_raw_text(article)
        article = re.sub(r"[^\x00-\x7F]+", " ", article)
        article = article.replace(" ,", ",").replace(" .", ".").replace(" !", "!").replace(" ?", "?")
        article = re.sub(r"\s+", " ", article).strip()
        sentences = re.split(r"(?<=[.!?])\s+", article)
        merged = " ".join(s.strip() for s in sentences if s.strip())
        return merged

    # Classification
    def classify_headline(self, headline: str) -> Tuple[Any, Optional[float]]:
        """
        Returns: (predicted_label, confidence [0..1] or None)
        """
        if not headline or not headline.strip():
            raise ValueError("Headline cannot be empty")

        try:
            pred = self.classifier.predict([headline])[0]
            pred_label = self.label_map.get(int(pred), str(pred))
            confidence = None
            if self._has_predict_proba:
                try:
                    proba = self.classifier.predict_proba([headline])[0]
                    confidence = float(max(proba))
                except Exception:
                    confidence = None
            return pred_label, confidence
        except Exception as e:
            logger.exception("Headline classification failed: %s", e)
            raise
    
    # Summarization with Groq API
    def abstractive_summary(self, article: str, min_len: int = 50, max_len: int = 150) -> str:
        """
        Use Groq API to generate abstractive summary.
        Falls back to truncation if API fails.
        """
        if not article or not article.strip():
            return ""

        try:
            prompt = f"""Write a brief 2-3 sentence summary of this news article. Write in paragraph form, not as a list. Focus on: what happened, who was involved, and the key outcome.

Article: {article}

Write only the summary, nothing else:"""

            # Call Groq API
            completion = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",  # Fast and efficient
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,  # Lower temp for factual consistency
                max_tokens=max_len,
            )
            
            summary = completion.choices[0].message.content.strip()
            
            # Remove any numbered list formatting if LLM still tries it
            summary = re.sub(r'^\d+\.\s*', '', summary, flags=re.MULTILINE)
            summary = re.sub(r'\n+', ' ', summary)  # Collapse newlines
            summary = re.sub(r'\s+', ' ', summary).strip()
            
            return summary

        except Exception as e:
            logger.exception("Groq API summarization failed: %s", e)
            # Fallback: return truncated cleaned article
            fallback = (article[:max_len * 2] + "...") if len(article) > max_len * 2 else article
            logger.warning("Using fallback truncation for summary")
            return fallback

    # 🕸️ URL Extraction Utility
    def extract_from_url(self, url: str) -> Tuple[str, str]:
        """
        Given a news article URL, downloads and extracts the headline and body text.
        Returns (headline, article_body)
        """
        if not url or not url.strip():
            raise ValueError("URL cannot be empty")

        try:
            art = Article(url)
            art.download()
            art.parse()
            headline = self.normalize_raw_text(art.title)
            article = self.clean_and_merge_article(art.text)
            if not article:
                raise ValueError("Failed to extract article content from the URL.")
            return headline, article
        except Exception as e:
            logger.exception("URL extraction failed: %s", e)
            raise RuntimeError(f"Failed to extract from URL: {e}")

    # Combined workflow
    def process_single(
        self, headline: str, article: str, min_len: int = 50, max_len: int = 150
    ) -> Dict[str, Any]:
        """
        Full pipeline: classify headline and summarize article.
        Returns a dict with keys: headline, predicted_class, confidence, summary, status
        """
        if not headline or not article:
            raise ValueError("Headline and article are required")

        # 1. classify
        pred_label, confidence = self.classify_headline(headline)
        confidence_str = f"{confidence:.2%}" if (confidence is not None) else "N/A"

        # 2. clean article
        cleaned = self.clean_and_merge_article(article)
        if not cleaned or len(cleaned) < 20:
            raise ValueError("Article too short or invalid after cleaning")

        # 3. summarize with Groq
        summary = self.abstractive_summary(cleaned, min_len=min_len, max_len=max_len)

        return {
            "headline": headline,
            "predicted_class": pred_label,
            "confidence": confidence_str,
            "summary": summary,
            "status": "success",
        }
    
    def process(
        self, headlines: Union[str, List[str]], articles: Union[str, List[str]], min_len: int = 50, max_len: int = 150
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Automatically handles both single and batch inputs.
        """
        # Single input
        if isinstance(headlines, str) and isinstance(articles, str):
            return self.process_single(headlines, articles, min_len, max_len)

        # Batch input
        elif isinstance(headlines, list) and isinstance(articles, list):
            results = []
            for h, a in zip(headlines, articles):
                try:
                    result = self.process_single(h, a, min_len, max_len)
                    results.append(result)
                except Exception as e:
                    logger.error(f"❌ Error processing article: {e}")
                    results.append({
                        "headline": h,
                        "error": str(e),
                        "status": "failed"
                    })
            return results

        # Mismatched types
        else:
            raise ValueError("headline and article must both be strings or both be lists")


if __name__ == "__main__":
    processor = NewslyticProcessor()

    headline = "Nigeria's Central Bank Introduces New Policy to Boost Small Business Loans"
    article = """The Central Bank of Nigeria has announced a new initiative aimed at increasing access to affordable credit for small and medium-sized enterprises. The programme will provide low-interest loans and technical support to help entrepreneurs grow their businesses and stimulate the economy."""

    results = processor.process_single(headline, article)
    print(results)
# %%