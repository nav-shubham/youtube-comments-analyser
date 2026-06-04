# YouTube Comment Intelligence Analyzer

An advanced NLP and sentiment analysis platform designed specifically to fetch, preprocess, and interpret YouTube comment sections. It helps creators, marketers, and researchers understand audience sentiment, extract key topics, identify common phrases (bigrams), and generate executive-ready reports.

---

## ✨ Features

- **High-Speed YouTube Scraping**: Integrates `scrapetube` or API-free fetching to pull comment threads directly from videos or channels.
- **Multilingual Sentiment Analysis (VADER)**: Enriches comments with polarity scores (Positive, Negative, Neutral) using an optimized VADER engine tailored with custom Hinglish (Hindi/English) and social-media stopwords.
- **Topic Modeling & Keyword Mining**: Performs bigram parsing (two-word phrases) and keyword tokenization to discover trending feedback categories.
- **Interactive Streamlit UI**: Premium dashboard built with Plotly graphs, word clouds, sentiment filters, and comment data tables.
- **Automated Report Generator**: Compiles metrics and saves them as executive-ready reports:
  - 📊 **Excel sheets** with detailed comment-level analytics.
  - 📄 **HTML documents** with clean CSS formatting.
  - 💾 **SQLite DB cache** to prevent re-fetching and save API quotas.

---

## 📁 Project Structure

- `app.py` - Core Streamlit web dashboard.
- `analyzer.py` - NLP text tokenization, stopwords definitions, and VADER sentiment enrichment.
- `database.py` - SQLite schema definition and query handlers.
- `youtube_fetcher.py` - Script to retrieve YouTube video data and comments.
- `reporting.py` - Compile results and export PDF, Excel, and HTML summaries.
- `run.bat` - Executable batch file to start the Streamlit server.
- `pyproject.toml` - Declares standard Python packaging dependencies (including `streamlit`, `scrapetube`, `vadersentiment`, `plotly`, `wordcloud`, etc.).

---

## 🚀 Setup & Execution

### 1. Setup Environment & Install Dependencies
Initialize and sync using the fast `uv` project manager:
```bash
uv venv
uv sync
```

### 2. Run Streamlit Application
Start the local dashboard:
```bash
uv run streamlit run app.py
```
Or simply double-click the `run.bat` batch file in Windows.

---

## ⚙️ How it Works
1. **Fetch**: Paste a YouTube Video URL in the dashboard.
2. **Analyze**: The backend pulls comments and passes them through VADER sentiment vectors.
3. **Visualize**: View sentiment distributions, bigram charts, and word clouds in real-time.
4. **Export**: Generate and download your `.xlsx` or `.html` executive reports with one click.
