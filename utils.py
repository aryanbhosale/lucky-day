from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import datetime
import math

device = "cuda:0" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)
labels = ["positive", "negative", "neutral"]

def estimate_sentiment(news_items):
    """
    Estimate sentiment from a list of news items.
    Each item is a dict with keys: 'headline', 'timestamp', and 'source'.
    Applies recency decay and source credibility weighting before aggregating FinBERT logits.
    """
    if not news_items:
        return 0.0, "neutral"
    
    aggregated_logits = None
    now = datetime.datetime.utcnow()
    total_weight = 0.0
    # Calibrated source credibility scores.
    source_credibility = {"Finnhub": 0.9, "NewsAPI": 0.8, "MarketAux": 0.7}
    alpha_decay = 0.1  # Decay rate for recency weighting

    for item in news_items:
        headline = item.get("headline")
        ts = item.get("timestamp")
        if ts is not None:
            delta_hours = (now - ts).total_seconds() / 3600.0
            recency_weight = math.exp(-alpha_decay * delta_hours)
        else:
            recency_weight = 0.5
        credibility = source_credibility.get(item.get("source"), 0.5)
        weight = recency_weight * credibility
        total_weight += weight

        tokens = tokenizer([headline], return_tensors="pt", padding=True).to(device)
        logits = model(tokens["input_ids"], attention_mask=tokens["attention_mask"])["logits"]
        weighted_logits = weight * logits[0]
        aggregated_logits = weighted_logits if aggregated_logits is None else aggregated_logits + weighted_logits

    aggregated_logits = aggregated_logits / total_weight
    aggregated_softmax = torch.nn.functional.softmax(aggregated_logits, dim=-1)
    max_index = torch.argmax(aggregated_softmax).item()
    probability = aggregated_softmax[max_index].item()
    sentiment = labels[max_index]
    return probability, sentiment
