# Daily News Digest - Gradio App

A Gradio-powered web application that fetches, classifies, and summarizes news articles from RSS feeds based on a selected date.

**🔗 Live Demo**: [Visit App on Hugging Face](https://huggingface.co/spaces/Deejay888/News_Classifier)  

---

## What It Does

- Accepts a date input (`YYYY-MM-DD`)
- Fetches top 3 articles per category from **The Hindu** RSS feeds:
  - 🌍 World
  - 💼 Business
  - 🏏 Sports
  - 🔬 Sci/Tech
- Uses a trained **Naive Bayes classifier** to predict the article's category
- Summarizes each article using **TF-IDF + PageRank-based Extractive Summarizer**

---

## Backend Model

- **Classifier**: `news_classifier.joblib`  
  - Built using TF-IDF vectors + Naive Bayes
- **Summarizer**: 
  - Extractive summarization based on cosine similarity and PageRank
  - Generates short summaries (2 lines or up to 40 words)

---

## Tech Stack

-  Python
-  Scikit-learn
-  Newspaper3k (for scraping articles)
-  NetworkX (for PageRank graph)
-  Feedparser (for parsing RSS feeds)
-  NLTK (for text preprocessing)
-  Gradio (for interactive UI)
-  Hugging Face Spaces (for deployment)

---

## Project Files

- `app.py` – Main application code (Gradio UI + logic)
- `news_classifier.joblib` – Pretrained text classification model
- `requirements.txt` – All dependencies for setting up the environment

---

