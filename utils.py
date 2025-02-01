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
    Batch processes headlines using FinBERT.
    Applies exponential recency decay and source credibility weighting.
    """
    if not news_items:
        return 0.0, "neutral"
    
    headlines = []
    weights = []
    now = datetime.datetime.utcnow()
    total_weight = 0.0
    # Adjust credibility as needed.
    source_credibility = {"Finnhub": 0.9, "NewsAPI": 0.8, "MarketAux": 0.7, "EventRegistry": 0.8}
    alpha_decay = 0.1

    for item in news_items:
        headline = item.get("headline")
        ts = item.get("timestamp")
        if ts:
            delta_hours = (now - ts).total_seconds() / 3600.0
            recency_weight = math.exp(-alpha_decay * delta_hours)
        else:
            recency_weight = 0.5
        credibility = source_credibility.get(item.get("source"), 0.5)
        weight = recency_weight * credibility
        headlines.append(headline)
        weights.append(weight)
        total_weight += weight

    # Batch tokenize headlines.
    tokens = tokenizer(headlines, return_tensors="pt", padding=True, truncation=True).to(device)
    logits = model(tokens["input_ids"], attention_mask=tokens["attention_mask"])["logits"]
    weights_tensor = torch.tensor(weights, device=device).unsqueeze(1)
    aggregated_logits = torch.sum(logits * weights_tensor, dim=0) / total_weight
    aggregated_softmax = torch.nn.functional.softmax(aggregated_logits, dim=-1)
    max_index = torch.argmax(aggregated_softmax).item()
    probability = aggregated_softmax[max_index].item()
    sentiment = labels[max_index]
    return probability, sentiment
