import gradio as gr
from newspaper import Article
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from joblib import load
import networkx as nx
import re

# Download stopwords
import nltk
nltk.download('stopwords')

# Set up stopwords and stemmer
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Clean and tokenize text
def clean_text(text):
    text = str(text).lower()
    tokens = re.findall(r'\b\w+\b', text)
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [stemmer.stem(t) for t in tokens]
    return ' '.join(tokens)

# Simple sentence tokenizer
def split_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

# Summarizer function using PageRank
def summarize(text, top_n=2, max_words=40):
    sentences = split_sentences(text)
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

# News article scraper
def fetch_article(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.title, article.text

# Load the trained classifier model
clf = load("news_classifier.joblib")

# Main prediction function
def classify_and_summarize(url):
    title, text = fetch_article(url)
    cleaned = clean_text(title + ". " + text)
    category = clf.predict([cleaned])[0]
    summary = summarize(text)
    return title, category, summary

# Gradio UI
demo = gr.Interface(
    fn=classify_and_summarize,
    inputs=gr.Textbox(label="Enter News URL"),
    outputs=[
        gr.Textbox(label="News Title"),
        gr.Textbox(label="Predicted Category"),
        gr.Textbox(label="Summary")
    ],
    title="📰 News Article Classifier and Summarizer",
    description="Paste a news article URL. Get its category and a short summary."
)

# Launch the app
demo.launch()
