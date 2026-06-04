import logging
import re
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

# Initialize VADER
vader_analyzer = SentimentIntensityAnalyzer()

# Suggestion patterns from the advanced reference repository
SUGGESTION_PATTERNS = [
    r"\b(?:you\s+)?(?:should|could|would|might)\s+(?:try|consider|add|include|make|do|use|show|explain|cover)",
    r"\b(?:please|pls)\s+(?:add|make|do|try|show|include|explain|cover)",
    r"\b(?:would\s+be\s+(?:nice|great|cool|awesome|better)\s+(?:if|to))",
    r"\b(?:i\s+(?:wish|hope|suggest|recommend|think\s+you\s+should))",
    r"\b(?:can\s+you|could\s+you)\s+(?:please\s+)?(?:add|make|do|show|include|explain|cover)",
    r"\b(?:it\s+would\s+(?:help|be\s+helpful))",
    r"\b(?:next\s+(?:video|time|episode))",
    r"\b(?:feature\s+request|suggestion|idea)",
    r"\bpourriez[- ]vous\b",
    r"\bvous\s+(?:devriez|pourriez|pouvez)\b",
    r"\bce\s+serait\s+(?:bien|super|cool|genial)\b",
    r"\bje\s+(?:suggere|propose|recommande|souhaite)\b",
    r"\bserait[- ]il\s+possible\b",
    r"\bune\s+(?:suggestion|idee|proposition)\b",
]

COMPILED_SUGGESTION_PATTERNS = [re.compile(p, re.IGNORECASE) for p in SUGGESTION_PATTERNS]

def check_is_suggestion(text):
    if not text or not isinstance(text, str):
        return False
    for pattern in COMPILED_SUGGESTION_PATTERNS:
        if pattern.search(text):
            return True
    return False

# Lazy-loaded BERT variables
_tokenizer = None
_model = None
_device = None
_bert_failed = False

def get_bert_model_and_tokenizer():
    global _tokenizer, _model, _device, _bert_failed
    if _bert_failed:
        return None, None, None
        
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer, _device
        
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        logger.info("Loading local BERT sentiment model 'nlptown/bert-base-multilingual-uncased-sentiment'...")
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
        _model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
        _model.to(_device)
        logger.info("BERT model loaded successfully!")
        return _model, _tokenizer, _device
    except Exception as e:
        logger.error(f"Failed to load BERT model/tokenizer: {e}")
        _bert_failed = True
        return None, None, None

def get_vader_sentiment(text):
    """
    Analyzes single comment text and returns (compound_score, label).
    Classifies comments into Positive, Neutral, Negative, or Mixed.
    """
    if check_is_suggestion(text):
        return 0.5, 'Suggestion'
        
    if not text or not isinstance(text, str):
        return 0.0, 'Neutral'
    
    scores = vader_analyzer.polarity_scores(text)
    compound = scores['compound']
    pos = scores['pos']
    neg = scores['neg']
    
    if pos >= 0.15 and neg >= 0.15:
        label = 'Mixed'
    elif compound >= 0.05:
        label = 'Positive'
    elif compound <= -0.05:
        label = 'Negative'
    else:
        label = 'Neutral'
        
    return compound, label

def enrich_comments_sentiment(df, mode="vader"):
    """
    Enriches a comments DataFrame on-the-fly with 'sentiment_score' and 'sentiment_label'.
    Supports 'vader' and 'bert' modes.
    """
    if df is None or df.empty or 'text' not in df.columns:
        return df
        
    df = df.copy()
    df['is_suggestion'] = df['text'].apply(check_is_suggestion)
    
    if mode == "bert":
        try:
            import torch
            model, tokenizer, device = get_bert_model_and_tokenizer()
            if model is not None:
                texts = df['text'].fillna("").tolist()
                suggestions = df['is_suggestion'].tolist()
                
                scores = []
                labels = []
                
                batch_size = 32
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i : i + batch_size]
                    batch_sugg = suggestions[i : i + batch_size]
                    
                    # Check if all are suggestions to skip tokenizer passes
                    if all(batch_sugg):
                        scores.extend([0.5] * len(batch_texts))
                        labels.extend(['Suggestion'] * len(batch_texts))
                        continue
                        
                    inputs = tokenizer(
                        batch_texts,
                        return_tensors="pt",
                        truncation=True,
                        max_length=128,
                        padding=True
                    )
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                        probabilities = torch.softmax(outputs.logits, dim=1)
                        predicted_classes = torch.argmax(probabilities, dim=1).tolist()
                        confidences = [probabilities[j][predicted_classes[j]].item() for j in range(len(predicted_classes))]
                        
                    for pred_class, conf, sugg in zip(predicted_classes, confidences, batch_sugg):
                        if sugg:
                            labels.append('Suggestion')
                            scores.append(0.5)
                        elif pred_class <= 1:
                            labels.append('Negative')
                            scores.append(-conf)
                        elif pred_class >= 3:
                            labels.append('Positive')
                            scores.append(conf)
                        else:
                            labels.append('Neutral')
                            scores.append(0.0)
                            
                df['sentiment_score'] = scores
                df['sentiment_label'] = labels
                return df
        except Exception as e:
            logger.error(f"Error in BERT sentiment prediction, falling back to VADER: {e}")
            
    # Fallback to VADER
    results = []
    for text, sugg in zip(df['text'], df['is_suggestion']):
        if sugg:
            results.append((0.5, 'Suggestion'))
            continue
            
        if not text or not isinstance(text, str):
            results.append((0.0, 'Neutral'))
            continue
            
        scores = vader_analyzer.polarity_scores(text)
        compound = scores['compound']
        pos = scores['pos']
        neg = scores['neg']
        
        if pos >= 0.15 and neg >= 0.15:
            label = 'Mixed'
        elif compound >= 0.05:
            label = 'Positive'
        elif compound <= -0.05:
            label = 'Negative'
        else:
            label = 'Neutral'
            
        results.append((compound, label))
        
    df['sentiment_score'] = [r[0] for r in results]
    df['sentiment_label'] = [r[1] for r in results]
    
    return df

