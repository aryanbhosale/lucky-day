import os
import datetime
import logging
import requests

from eventregistry import EventRegistry, QueryArticlesIter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def get_news_from_finnhub(symbol: str, start: str, end: str):
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        logging.error("Finnhub API key not found.")
        return []
    url = "https://finnhub.io/api/v1/company-news"
    params = {
        "symbol": symbol,
        "from": start,
        "to": end,
        "token": api_key
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        news_items = []
        for item in data:
            headline = item.get("headline")
            time_stamp = item.get("datetime")
            try:
                published_at = datetime.datetime.utcfromtimestamp(time_stamp)
            except Exception:
                published_at = None
            news_items.append({
                "headline": headline,
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
    # Use the symbol as keyword; adjust as needed (e.g. converting a symbol to a company name)
    q = QueryArticlesIter(
        keywords=symbol,
        dateStart=start,
        dateEnd=end,
        lang="eng"
    )
    try:
        articles = q.execQuery(er, sortBy="date", maxItems=100)
        news_items = []
        for art in articles:
            headline = art.get("title")
            date_str = art.get("date")
            try:
                published_at = datetime.datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")
            except Exception:
                published_at = None
            news_items.append({
                "headline": headline,
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
            headline = article.get("title")
            time_str = article.get("published_at")
            try:
                published_at = datetime.datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%SZ")
            except Exception:
                published_at = None
            news_items.append({
                "headline": headline,
                "timestamp": published_at,
                "source": "MarketAux"
            })
        logging.info(f"MarketAux: Retrieved {len(news_items)} news items.")
        return news_items
    except Exception as e:
        logging.error(f"Error fetching news from MarketAux: {e}")
        return []

def get_aggregated_news(symbol: str, start: str, end: str):
    news = []
    news.extend(get_news_from_finnhub(symbol, start, end))
    news.extend(get_news_from_eventregistry(symbol, start, end))
    news.extend(get_news_from_marketaux(symbol, start, end))
    unique_news = {}
    for item in news:
        headline = item.get("headline")
        if headline and headline not in unique_news:
            unique_news[headline] = item
    aggregated_news = list(unique_news.values())
    aggregated_news.sort(key=lambda x: x["timestamp"] if x["timestamp"] else datetime.datetime.min, reverse=True)
    logging.info(f"Aggregated a total of {len(aggregated_news)} unique news items.")
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
