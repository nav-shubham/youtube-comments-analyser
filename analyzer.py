import re
import pandas as pd
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize VADER Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# A self-contained list of common English stopwords + YouTube/social media terms + Hinglish stopwords
STOPWORDS = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
    'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
    'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
    'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should',
    "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
    'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
    'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
    'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", 'like', 'get', 'would', 'video', 
    'youtube', 'videos', 'channel', 'love', 'one', 'good', 'great', 'really', 'much', 'best', 'make', 'people',
    'always', 'think', 'see', 'view', 'watch', 'watching', 'guys', 'sub', 'subscriber', 'subscribers', 'subscribe',
    # Hinglish & generic noise words
    'bhai', 'hai', 'bhi', 'ki', 'ke', 'ko', 'ka', 'me', 'se', 'hi', 'vlog', 'vlogs', 'aur', 'di', 'ek', 'par', 'ne', 
    'main', 'yeh', 'toh', 'tha', 'thi', 'the', 'ho', 'h', 'vlogger', 'vloger', 'karo', 'kar', 'karna', 'karke', 
    'aap', 'tum', 'mera', 'meri', 'mere', 'apna', 'apni', 'apne', 'sab', 'kuch', 'hoga', 'hogi', 'aaj', 'kal', 
    'din', 'baat', 'yaar', 'yaaro', 'bhaiya', 'bhabhi', 'didi', 'anu', 'div', 'anuanddiv', 'comment', 'comments',
    'video', 'videos', 'please', 'pls', 'watching', 'watch', 'like', 'likes'
])

def enrich_dataframe(df):
    """
    Dynamically enriches a comments DataFrame on-the-fly with sentiment scores and labels.
    Uses pandas vectorization for extremely high-speed CPU calculations.
    """
    if df is None or df.empty:
        return df
        
    # Apply sentiment analysis on-the-fly
    def get_vader_metrics(text):
        if not text or not isinstance(text, str):
            return 0.0, 'Neutral'
        scores = analyzer.polarity_scores(text)
        compound = scores['compound']
        if compound >= 0.05:
            return compound, 'Positive'
        elif compound <= -0.05:
            return compound, 'Negative'
        else:
            return compound, 'Neutral'
            
    # Apply function along column
    results = df['text'].apply(get_vader_metrics)
    
    # Extract compound score and label lists
    df['sentiment_score'] = [r[0] for r in results]
    df['sentiment_label'] = [r[1] for r in results]
    
    return df

def clean_and_tokenize(text):
    """
    Cleans text for keyword frequency analysis (lowercasing, removing punctuation).
    Retains alphanumeric characters.
    """
    if not text:
        return []
    # Convert to lowercase and replace non-alphanumeric chars with space
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = text.split()
    # Filter stopwords and short tokens
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]

def get_top_keywords(comments_list, top_n=50):
    """
    Returns a dictionary of top keyword counts.
    """
    word_counter = Counter()
    for comment in comments_list:
        tokens = clean_and_tokenize(comment.get('text', ''))
        word_counter.update(tokens)
    return dict(word_counter.most_common(top_n))

def get_top_bigrams(comments_list, top_n=30):
    """
    Returns a list of top 2-word phrase counts.
    """
    bigram_counter = Counter()
    for comment in comments_list:
        tokens = clean_and_tokenize(comment.get('text', ''))
        if len(tokens) >= 2:
            bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]
            bigram_counter.update(bigrams)
    return dict(bigram_counter.most_common(top_n))
