# Daily News Digest - Web App

A Streamlit web application that classifies and summarizes news articles based on their URL input.

**Live Demo**: [Visit App on Hugging Face](https://huggingface.co/spaces/Deejay888/News_Classifier)

## What It Does

- Takes a live news article URL
- Classifies the category (e.g., World, Sports, Business, Sci/Tech)
- Generates a 2-line summary using TF-IDF + PageRank

## Backend Model

- Pretrained Naive Bayes Classifier (`news_classifier.joblib`)
- Custom Extractive Summarizer

## Tech Stack

- Python
- Streamlit
- Scikit-learn
- NLTK (replaced with custom cleaner)
- BeautifulSoup (for scraping)
- Hugging Face Spaces (deployment)

## Project Files
- app.py # Web interface
- requirements.txt # All dependencies
- news_classifier.joblib # Trained ML model
