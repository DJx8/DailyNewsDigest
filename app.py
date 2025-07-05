import gradio as gr
import feedparser
from datetime import datetime
from newspaper import Article
from joblib import load
import nltk
import networkx as nx
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download only what's needed
nltk.download('stopwords')

# Load classifier
clf = load("news_classifier.joblib")

# Preprocessing tools
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Clean and tokenize text
def clean_text(text):
    text = str(text).lower()
    tokens = re.findall(r'\b\w+\b', text)
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [stemmer.stem(t) for t in tokens]
    return ' '.join(tokens)

# Regex-based sentence splitting
def split_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

# Summarizer using PageRank
def summarize(text, top_n=2, max_words=40):
    sentences = split_sentences(text)
    if len(sentences) <= top_n:
        return ' '.join(sentences)
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

# Article fetching using newspaper3k
def fetch_article_details(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        title, text = article.title, article.text
        cleaned = clean_text(title + ". " + text)
        predicted_category = clf.predict([cleaned])[0]
        summary = summarize(text)
        return title, url, predicted_category, summary
    except Exception as e:
        return "Error", url, "N/A", str(e)

# Parse RSS feed
def fetch_top_articles_from_rss(date_str, category_rss_path):
    url = f"https://www.thehindu.com/{category_rss_path}/?service=rss"
    feed = feedparser.parse(url)
    target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    articles = []
    for entry in feed.entries:
        if hasattr(entry, 'published_parsed'):
            pub_date = datetime(*entry.published_parsed[:6]).date()
            if pub_date == target_date:
                details = fetch_article_details(entry.link)
                if details:
                    articles.append(details)
        if len(articles) >= 3:
            break
    return articles

# Main function for Gradio
def classify_rss_by_date(date_input):
    date_input = date_input.strip()
    category_mapping = {
        "World": "news/international",
        "Business": "business",
        "Sports": "sport",
        "Sci/Tech": "sci-tech"
    }

    output = ""
    for category, rss_path in category_mapping.items():
        output += f"### 🔹 {category}\n\n"
        articles = fetch_top_articles_from_rss(date_input, rss_path)
        if not articles:
            output += "_No articles found._\n\n"
            continue
        for i, (title, url, pred_cat, summary) in enumerate(articles, 1):
            output += f"**{i}. {title}**\n"
            output += f"- 🔗 [Link]({url})\n"
            
            output += f"- 📝 Summary: {summary}\n\n"
        output += "---\n\n"
    return output

# UI
demo = gr.Interface(
    fn=classify_rss_by_date,
    inputs=gr.Textbox(label="📅 Enter Date ", placeholder="YYYY-MM-DD"),
    outputs=gr.Markdown(label="📋 Results"),
    title="📰 Daily News Digest",
    description="Enter a date to fetch top 3 articles per category (World, Business, Sports, Sci/Tech) and gives a short summary"
)

# Launch
demo.launch()
