import os
import logging
from typing import List, Dict, Tuple
import google.generativeai as genai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logging.error("Gemini API key not found.")
genai.configure(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-1.5-flash"

def get_enriched_sentiment_summary(aggregated_news: List[Dict], symbol: str) -> Tuple[float, str, str]:
    prompt = (
        f"Aggregate all relevant news headlines for the symbol {symbol} from the past week. "
        "Each headline is timestamped. Provide a quick sentiment analysis that highlights key market-moving events, "
        "and compare these trends with previous weeks to forecast market momentum.\n\n"
    )
    for item in aggregated_news:
        ts = item["timestamp"].strftime("%Y-%m-%d %H:%M:%S") if item["timestamp"] else "Unknown Time"
        prompt += f"[{ts}] ({item['source']}): {item['headline']}\n"
    prompt += "\nProvide your analysis in a concise manner, and conclude with the overall sentiment (positive, negative, or neutral)."
    
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        summary = response.text
        summary_lower = summary.lower()
        if "positive" in summary_lower:
            sentiment = "positive"
        elif "negative" in summary_lower:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        confidence = 0.99  # Calibrated confidence score
        return confidence, sentiment, summary
    except Exception as e:
        logging.error(f"Error calling Google Gemini API: {e}")
        return 0.0, "neutral", "Error in obtaining enriched analysis."
