import os
import datetime
import logging
import time
import requests
from eventregistry import EventRegistry, QueryArticlesIter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Cache dictionary with TTL (seconds)
NEWS_CACHE = {}
CACHE_TTL = 300  # 5 minutes

def parse_timestamp_from_string(time_str, fmt="%Y-%m-%dT%H:%M:%SZ"):
    try:
        return datetime.datetime.strptime(time_str, fmt)
    except Exception:
        return None

def get_news_from_finnhub(symbol: str, start: str, end: str):
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        logging.error("Finnhub API key not found.")
        return []
    url = "https://finnhub.io/api/v1/company-news"
    params = {"symbol": symbol, "from": start, "to": end, "token": api_key}
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        news_items = []
        for item in data:
            published_at = None
            if "datetime" in item:
                try:
                    published_at = datetime.datetime.utcfromtimestamp(item["datetime"])
                except Exception:
                    published_at = None
            news_items.append({
                "headline": item.get("headline"),
                "timestamp": published_at,
                "source": "Finnhub"
            })
        logging.info(f"Finnhub: Retrieved {len(news_items)} news items.")
        return news_items
    except Exception as e:
        logging.error(f"Error fetching news from Finnhub: {e}")
        return []

def get_news_from_eventregistry(symbol: str, start: str, end: str):
    api_key = os.getenv("EVENTREGISTRY_API_KEY")
    if not api_key:
        logging.error("EventRegistry API key not found.")
        return []
    er = EventRegistry(apiKey=api_key)
    q = QueryArticlesIter(keywords=symbol, dateStart=start, dateEnd=end, lang="eng")
    try:
        articles = q.execQuery(er, sortBy="date", maxItems=100)
        news_items = []
        for art in articles:
            published_at = parse_timestamp_from_string(art.get("date"))
            news_items.append({
                "headline": art.get("title"),
                "timestamp": published_at,
                "source": "EventRegistry"
            })
        logging.info(f"EventRegistry: Retrieved {len(news_items)} news items.")
        return news_items
    except Exception as e:
        logging.error(f"Error fetching news from EventRegistry: {e}")
        return []

def get_news_from_marketaux(symbol: str, start: str, end: str):
    api_key = os.getenv("MARKETAUX_KEY")
    if not api_key:
        logging.error("MarketAux API key not found.")
        return []
    url = "https://api.marketaux.com/v1/news/all"
    params = {
        "api_token": api_key,
        "symbols": symbol,
        "published_after": start,
        "published_before": end,
        "language": "en"
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        news_items = []
        for article in data.get("data", []):
            published_at = parse_timestamp_from_string(article.get("published_at"))
            news_items.append({
                "headline": article.get("title"),
                "timestamp": published_at,
                "source": "MarketAux"
            })
        logging.info(f"MarketAux: Retrieved {len(news_items)} news items.")
        return news_items
    except Exception as e:
        logging.error(f"Error fetching news from MarketAux: {e}")
        return []

def get_aggregated_news(symbol: str, start: str, end: str):
    global NEWS_CACHE
    cache_key = (symbol, start, end)
    now = time.time()
    if cache_key in NEWS_CACHE and now - NEWS_CACHE[cache_key]["timestamp"] < CACHE_TTL:
        logging.info("Returning cached aggregated news data.")
        return NEWS_CACHE[cache_key]["data"]
    news = []
    news.extend(get_news_from_finnhub(symbol, start, end))
    news.extend(get_news_from_eventregistry(symbol, start, end))
    news.extend(get_news_from_marketaux(symbol, start, end))
    unique_news = {}
    for item in news:
        if item.get("headline") and item.get("headline") not in unique_news:
            unique_news[item["headline"]] = item
    aggregated_news = list(unique_news.values())
    aggregated_news.sort(key=lambda x: x["timestamp"] if x["timestamp"] else datetime.datetime.min, reverse=True)
    logging.info(f"Aggregated a total of {len(aggregated_news)} unique news items.")
    NEWS_CACHE[cache_key] = {"timestamp": now, "data": aggregated_news}
    return aggregated_news

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    symbol = "AAPL"
    start = "2023-01-01"
    end = "2023-01-07"
    news = get_aggregated_news(symbol, start, end)
    for item in news:
        print(item)
