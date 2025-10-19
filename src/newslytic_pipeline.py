#%%
# src/newslytic_pipeline.py
import os
import re
import logging
from typing import Optional, Tuple, Dict, Any, List, Union
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class NewslyticProcessor:
    """
    Class that encapsulates the headline classifier and the abstractive summarizer.
    Instantiate once at app startup and reuse for every request.
    """

    def __init__(
        self,
        classifier_path: Optional[str] = None,
        hf_model_name: str = "Jayywestty/bart-summarizer-epoch2",
        device: Optional[str] = None,
    ):
        """
        Args:
            classifier_path: Path to joblib classifier file. If None, tries a sensible default.
            hf_model_name: Hugging Face repo id or local path for the BART summarizer.
            device: e.g. "cuda" | "cpu". If None, auto-detects.
        """
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.classifier_path = classifier_path or self._default_classifier_path()
        self.hf_model_name = hf_model_name

        # placeholders
        self.classifier = None
        self._has_predict_proba = False
        self.tokenizer = None
        self.summarizer = None

        # load models now
        self._load_classifier()
        self._load_summarizer()

        #label mapping
        self.label_map = {0: "non-crime", 1: "crime"}


    def _default_classifier_path(self) -> str:
        # Default tries to load from a sibling Model folder 
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

    def _load_summarizer(self):
        """Load HF tokenizer and model - tries cache first, downloads if needed."""
        try:
            logger.info("Loading summarizer: %s (device=%s)", self.hf_model_name, self.device)
            
            # Try local cache first, fallback to download
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.hf_model_name, 
                    local_files_only=True
                )
                self.summarizer = AutoModelForSeq2SeqLM.from_pretrained(
                    self.hf_model_name, 
                    local_files_only=True
                )
                logger.info("✅ Summarizer loaded from cache")
            except (OSError, ValueError, Exception):
                logger.info("📥 Cache miss - downloading from HuggingFace (this may take a few minutes)...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name)
                self.summarizer = AutoModelForSeq2SeqLM.from_pretrained(self.hf_model_name)
                logger.info("✅ Summarizer downloaded and cached successfully")
            
            # Move model to device and set to eval mode
            self.summarizer.to(self.device)
            self.summarizer.eval()
            logger.info("✅ Summarizer ready on %s", self.device)
            
        except Exception as e:
            logger.exception("❌ Failed to load summarizer: %s", e)
            raise RuntimeError(f"Failed to load summarizer: {e}")

   
    # Text cleaning utilities
    @staticmethod
    def normalize_raw_text(text: str) -> str:
        if not text:
            return ""
        # normalize smart quotes and dashes; collapse whitespace
        text = text.replace(""", '"').replace(""", '"').replace("'", "'").replace("'", "'")
        text = text.replace("–", "-").replace("—", "-")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def clean_and_merge_article(self, article: str) -> str:
        """Basic cleanup and merging of sentences to prepare for summarization."""
        if not article:
            return ""
        article = self.normalize_raw_text(article)
        # remove weird control characters but keep punctuation
        article = re.sub(r"[^\x00-\x7F]+", " ", article)
        # remove stray spaces before punctuation
        article = article.replace(" ,", ",").replace(" .", ".").replace(" !", "!").replace(" ?", "?")
        # collapse multiple newlines and spaces
        article = re.sub(r"\s+", " ", article).strip()
        # optionally split/merge sentences intelligently (simple approach)
        sentences = re.split(r"(?<=[.!?])\s+", article)
        merged = " ".join(s.strip() for s in sentences if s.strip())
        return merged

    # Classification
    def classify_headline(self, headline: str) -> Tuple[Any, Optional[float]]:
        """
        Returns: (predicted_label, confidence [0..1] or None)
        predicted_label is whatever classifier.predict returns (often 0/1 or string labels)
        """
        if not headline or not headline.strip():
            raise ValueError("Headline cannot be empty")

        try:
            pred = self.classifier.predict([headline])[0]
            pred_label = self.label_map.get(int(pred), str(pred))  # convert 0/1 → crime/non-crime
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
    
    # Summarization
    def abstractive_summary(self, article: str, min_len: int = 50, max_len: int = 150) -> str:
        """Run the model.generate to produce an abstractive summary."""
        if not article or not article.strip():
            return ""

        inputs = self.tokenizer(
            article,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
        )
        # move tensors to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        try:
            with torch.no_grad():
                summary_ids = self.summarizer.generate(
                    **inputs,
                    max_length=max_len,
                    min_length=min_len,
                    length_penalty=2.0,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                )
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            # post-clean
            summary = re.sub(r"\s+", " ", summary).strip()
            return summary
        except Exception as e:
            logger.exception("Summarization failed: %s", e)
            # fallback: return truncated cleaned article
            return (article[: max_len * 2] + "...") if len(article) > max_len * 2 else article

    
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

        # convert confidence to percentage string if present
        confidence_str = f"{confidence:.2%}" if (confidence is not None) else "N/A"

        # 2. clean article
        cleaned = self.clean_and_merge_article(article)
        if not cleaned or len(cleaned) < 20:
            raise ValueError("Article too short or invalid after cleaning")

        # 3. summarize
        summary = self.abstractive_summary(cleaned, min_len=min_len, max_len=max_len)

        # return structured result
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
        - If given strings -> processes one article.
        - If given lists -> processes each pair in batch.
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
