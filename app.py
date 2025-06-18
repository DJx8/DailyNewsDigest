import streamlit as st
import nltk
import joblib
import requests
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

# 👇 Add local NLTK data path
nltk.data.path.append('nltk_data')

# Load model
model = joblib.load("news_classifier.joblib")

# Helper Functions
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    text = str(text).lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha()]
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [stemmer.stem(t) for t in tokens]
    return ' '.join(tokens)

def summarize(text, top_n=2, max_words=40):
    sentences = sent_tokenize(text)
    if len(sentences) <= top_n:
        return ' '.join(sentences[:top_n])

    tfidf = TfidfVectorizer().fit_transform(sentences)
    sim_matrix = cosine_similarity(tfidf)
    nx_graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(nx_graph)
    ranked = sorted(((scores[i], s, i) for i, s in enumerate(sentences)), reverse=True)

    summary = []
    total_words = 0
    for _, sentence, idx in sorted(ranked[:len(sentences)], key=lambda x: x[2]):
        word_count = len(sentence.split())
        if total_words + word_count <= max_words:
            summary.append(sentence)
            total_words += word_count
        if len(summary) >= top_n or total_words >= max_words:
            break
    return ' '.join(summary)

# ✅ Updated fetch function (no newspaper lib)
def fetch_article(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])
        title = soup.title.string if soup.title else "No title found"
        return title, text
    except Exception as e:
        return "Error", f"Failed to fetch article: {e}"

# Streamlit UI
st.title("📰 Daily News Summarizer")
url = st.text_input("Paste a news article URL:")

if st.button("Summarize"):
    if url:
        try:
            title, text = fetch_article(url)
            cleaned = clean_text(title + ". " + text)
            category = model.predict([cleaned])[0]
            summary = summarize(text)

            st.subheader("📰 Title")
            st.write(title)
            st.subheader("📂 Category")
            st.write(category)
            st.subheader("📝 Summary")
            st.write(summary)
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please paste a valid news URL.")
