# 📰 Newslytic — Real-Time AI News Summarizer & Classifier

**Turning news into insights — instantly.**

A real-time AI web app that classifies news headlines and generates concise human-like summaries from any article URL.

---

## 🌍 Overview

In today’s information-dense world, understanding what matters shouldn’t take hours. **Newslytic** instantly analyzes any news article — identifying whether it’s crime-related or not and producing a readable, context-aware summary in seconds.

Originally inspired by the challenge of information overload, Newslytic helps individuals and businesses transform unstructured news data into actionable insights.

---

## 🚀 Key Features

* ✅ **Headline Classification** — Distinguishes crime vs non-crime stories using a **Bernoulli Naive Bayes** model.
* ✅ **Real-Time Summarization** — Leverages **Groq GenAI API** for lightning-fast, coherent summaries.
* ✅ **URL-to-Insight Pipeline** — Processes any valid news link into structured insights in one call.
* ✅ **Web & API Access** — Available through a **FastAPI** backend and **Streamlit** web UI.
* ✅ **Lightweight & Scalable** — Optimized for deployment on Render and Hugging Face Spaces.

---

## 📊 Results

* **Headline classification accuracy:** ~96% on held-out test data.
* **Average summary length:** 2–3 sentences with >90% factual retention (human-verified).
* **Latency:** Under **3 seconds** per full request (classification + summarization).

### 🧠 Example Output

> **Input:** “Police uncover major fraud ring in Lagos tech firm.”
> **Classification:** Crime
> **Summary:** “Authorities have dismantled a tech-based fraud operation in Lagos, recovering evidence linked to financial scams.”

---

## 🧩 Tech Stack

### Programming & Frameworks
* **Python**
* **FastAPI** (backend API)
* **Streamlit** (web interface)
* **Google Colab** (model experimentation)

### Machine Learning / NLP
* **Scikit-learn** — BernoulliNB headline classifier
* **Groq GenAI API** — real-time summarization
* **Pandas, NumPy, Regex** — text preprocessing

### Deployment & Tools
* **Render** (FastAPI backend hosting)
* **Hugging Face Spaces** (model API hosting)
* Git, VS Code, Virtual Environments

---

## ⚙️ How It Works

1.  **Input** — User submits a valid news article URL.
2.  **Text Extraction** — The article body and headline are parsed using the internal scraper.
3.  **Headline Classification** — BernoulliNB model predicts whether it’s crime or non-crime.
4.  **Summarization** — The article body is passed to the **Groq GenAI API** for real-time summarization.
5.  **Response Delivery** — FastAPI returns both classification and summary via JSON, which Streamlit displays neatly on the UI.

---

## 💻 Setup & Installation

Follow these steps to run Newslytic locally:

### 1. Clone the repository

```bash
git clone [https://github.com/jaywestty/newslytic.git](https://github.com/jaywestty/newslytic.git)
cd newslytic```

---

### 2. Create and activate a virtual environment
