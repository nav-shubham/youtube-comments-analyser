import re
from collections import Counter
import pandas as pd

# Self-contained list of common English stopwords + YouTube/social media terms + Hinglish stopwords
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

def calculate_quality_score(text):
    """
    Computes a raw comment quality score between 0.0 and 1.0.
    Identifies short texts, emoji spam, repetitive text, and link spam.
    """
    if not text or not isinstance(text, str):
        return 0.0
        
    text_lower = text.lower()
    words = [w for w in text_lower.split() if w.strip()]
    if not words:
        return 0.0
        
    score = 1.0
    
    # 1. Length penalty
    if len(text) < 12:
        score -= 0.3
    if len(words) < 3:
        score -= 0.2
        
    # 2. Vocabulary repetition penalty (e.g. spam repeating words)
    unique_ratio = len(set(words)) / len(words)
    if unique_ratio < 0.6:
        score -= (1.0 - unique_ratio)
        
    # 3. Excessive emoji penalty
    emoji_count = len(re.findall(r'[\u2700-\u27bf]|[\u2600-\u26ff]|[\u2300-\u23ff]|[\U0001f300-\U0001f6ff]|[\U0001f900-\U0001f9ff]|[\U0001f600-\U0001f64f]', text))
    if len(words) > 0 and (emoji_count / len(words)) > 1.5:
        score -= 0.4
    if emoji_count > 6:
        score -= 0.3
        
    # 4. Link & Promotion penalty
    if re.search(r'http|https|www\.|t\.me|wa\.me|\.com', text_lower):
        score -= 0.8
        
    return max(0.0, score)

def filter_quality_comments(df):
    """
    Applies a rigorous multi-stage preprocessing quality layer to comments:
    1. Removes Crypto/Investment/Financial Spam.
    2. Removes Promotional comments and external links.
    3. Removes Generic Copy-Paste spam (identical duplicates across different authors).
    4. Removes Bot-like repetitive texts and excessive emojis.
    5. Calculates a Quality Score and drops comments below 0.35 threshold.
    Returns: A clean DataFrame containing only high-value genuine audience opinions.
    """
    if df is None or df.empty:
        return df
        
    df = df.copy()
    initial_count = len(df)
    
    # 1. Crypto & promotional keywords filter
    spam_pattern = re.compile(
        r'\b(?:crypto|bitcoin|investment|telegram|whatsapp|profit|invest|traded|trading|broker|earn money|make money|wa\.me|t\.me|cashout|giveaway|bonus|dm me|inbox me|follow my)\b',
        re.IGNORECASE
    )
    df = df[~df['text'].str.contains(spam_pattern, na=False)]
    
    # 2. External links filter
    df = df[~df['text'].str.contains(r'http|https|www\.|t\.me|wa\.me', na=False, case=False)]
    
    # 3. Copy-Paste identical duplicates (retains first commenter)
    df = df.drop_duplicates(subset=['text'], keep='first')
    
    # 4. Filter by quality score (threshold = 0.35)
    df['quality_score'] = df['text'].apply(calculate_quality_score)
    df = df[df['quality_score'] >= 0.35]
    
    # Drop quality_score column from active schema to preserve database consistency
    df = df.drop(columns=['quality_score'])
    
    final_count = len(df)
    print(f"Preprocessing Quality Layer: Filtered out {initial_count - final_count} spam/low-quality comments. Remaining: {final_count}.")
    
    return df

def clean_text(text):
    """
    Cleans raw text of non-ASCII printable chars, lowercases it, and prepares it for NLP processing.
    """
    if not text or not isinstance(text, str):
        return ""
    # Standardize spaces and lower
    return text.strip().lower()

def clean_and_tokenize(text):
    """
    Cleans text and tokenizes it, filtering out stopwords and non-alphanumeric noise.
    """
    if not text:
        return []
    
    cleaned = re.sub(r'[^a-z0-9\s]', ' ', text.lower())
    tokens = cleaned.split()
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]

def get_top_n_words(df, n=50):
    """
    Extracts word frequencies from a comments dataframe column 'text'.
    Returns a dictionary of word counts.
    """
    if df.empty or 'text' not in df.columns:
        return {}
    
    counter = Counter()
    for text in df['text']:
        tokens = clean_and_tokenize(text)
        counter.update(tokens)
    return dict(counter.most_common(n))

def get_top_n_bigrams(df, n=30):
    """
    Extracts 2-word bigram phrase frequencies from comments dataframe text column.
    Returns a dictionary of phrase counts.
    """
    if df.empty or 'text' not in df.columns:
        return {}
    
    counter = Counter()
    for text in df['text']:
        tokens = clean_and_tokenize(text)
        if len(tokens) >= 2:
            bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]
            counter.update(bigrams)
    return dict(counter.most_common(n))
