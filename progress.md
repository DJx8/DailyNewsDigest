# Weekly Progress Log

## Week 1: Problem Formulation & Dataset
- Defined problem: Daily News Summarizer + Category Classifier
- Justified use-case: Time-saving, topic-wise news reader
- Dataset: GNews dataset (4 categories)
- Did basic EDA and preprocessing
- Decided model: Naive Bayes for classification, TF-IDF + PageRank for summarization

## Week 2: Model Building
- Trained Naive Bayes classifier with TF-IDF vectorizer
- Achieved ~88% accuracy
- Built extractive summarizer using sentence ranking (PageRank)
- Evaluated with classification_report
- Faced `newspaper3k` issue; switched to BeautifulSoup

## Week 3: Finalization & Deployment
- Created working Streamlit UI (Hugging Face Spaces)
- Cleaned code, improved UI experience
- Added model to app (joblib)
- Created `README.md` in both repos
- Documented team contributions and project structure
