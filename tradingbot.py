import os
from dotenv import load_dotenv
from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from datetime import datetime, timedelta
from alpaca_trade_api import REST
import logging
import numpy as np

from news_collector import get_aggregated_news
from utils import estimate_sentiment
from chatgpt_client import get_enriched_sentiment_summary
from rl_agent import DQNAgent

os.environ.pop("ALPACA_API_KEY", None)
os.environ.pop("ALPACA_API_SECRET", None)
os.environ.pop("ALPACA_BASE_URL", None)
os.environ.pop("FINNHUB_API_KEY", None)
os.environ.pop("EVENTREGISTRY_API_KEY", None)
os.environ.pop("MARKETAUX_KEY", None)
os.environ.pop("GEMINIAI_API_KEY", None)
os.environ.pop("GEMINI_API_URL", None)

load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")
BASE_URL = os.getenv("ALPACA_BASE_URL")

ALPACA_CREDS = {
    "API_KEY": API_KEY,
    "API_SECRET": API_SECRET,
    "PAPER": True
}

logging.basicConfig(level=logging.INFO)

class MLTrader(Strategy):
    def initialize(self, symbol: str = "SPY", cash_at_risk: float = 0.5):
        self.symbol = symbol
        self.sleeptime = "24H"
        self.cash_at_risk = cash_at_risk
        self.api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)
        # State vector: [sentiment probability, sentiment value, price momentum, current position]
        # Actions: 0 = hold, 1 = buy, 2 = sell.
        self.rl_agent = DQNAgent(state_size=4, action_size=3)
        self.prev_state = None
        self.prev_price = None
        self.prev_action = None
        self.current_position = 0  # 1 for long, -1 for short, 0 for flat

    def position_sizing(self):
        cash = self.get_cash()
        last_price = self.get_last_price(self.symbol)
        quantity = round(cash * self.cash_at_risk / last_price, 0)
        return cash, last_price, quantity
    
    def get_dates(self):
        today = self.get_datetime()
        three_days_prior = today - timedelta(days=3)
        return today.strftime('%Y-%m-%d'), three_days_prior.strftime('%Y-%m-%d')

    def get_sentiment(self):
        today, three_days_prior = self.get_dates()
        aggregated_news = get_aggregated_news(self.symbol, three_days_prior, today)
        probability_f, sentiment_f = estimate_sentiment(aggregated_news)
        probability_c, sentiment_c, enriched_summary = get_enriched_sentiment_summary(aggregated_news, self.symbol)
        if sentiment_f == sentiment_c:
            combined_probability = (probability_f + probability_c) / 2.0
            combined_sentiment = sentiment_f
        else:
            if probability_f > probability_c + 0.1:
                combined_probability = probability_f
                combined_sentiment = sentiment_f
            elif probability_c > probability_f + 0.1:
                combined_probability = probability_c
                combined_sentiment = sentiment_c
            else:
                combined_probability = min(probability_f, probability_c)
                combined_sentiment = "neutral"
        logging.info(f"FinBERT: {probability_f:.3f} ({sentiment_f}) | Gemini: {probability_c:.3f} ({sentiment_c})")
        logging.info("Enriched Analysis Summary: " + enriched_summary)
        return combined_probability, combined_sentiment

    def compute_state(self, last_price, sentiment_probability, sentiment_label):
        # Encode sentiment label numerically: positive=1, negative=-1, neutral=0.
        sentiment_value = 1 if sentiment_label == "positive" else (-1 if sentiment_label == "negative" else 0)
        price_momentum = 0.0
        if self.prev_price is not None and self.prev_price > 0:
            price_momentum = (last_price - self.prev_price) / self.prev_price
        return [sentiment_probability, sentiment_value, price_momentum, self.current_position]

    def on_trading_iteration(self):
        cash, last_price, quantity = self.position_sizing()
        sentiment_probability, sentiment_label = self.get_sentiment()
        current_state = self.compute_state(last_price, sentiment_probability, sentiment_label)
        
        if self.prev_state is not None and self.prev_price is not None:
            reward = 0.0
            if self.current_position == 1:  # Long position
                reward = (last_price - self.prev_price) / self.prev_price
            elif self.current_position == -1:  # Short position
                reward = (self.prev_price - last_price) / self.prev_price
            done = False
            self.rl_agent.remember(self.prev_state, self.prev_action, reward, current_state, done)
            self.rl_agent.train()
        
        action = self.rl_agent.choose_action(current_state)
        
        if cash > last_price:
            if action == 1 and self.current_position != 1:
                if self.current_position == -1:
                    self.sell_all()  # Cover short position.
                order = self.create_order(self.symbol, quantity, "buy", type="market")
                self.submit_order(order)
                self.current_position = 1
            elif action == 2 and self.current_position != -1:
                if self.current_position == 1:
                    self.sell_all()  # Exit long position.
                order = self.create_order(self.symbol, quantity, "sell", type="market")
                self.submit_order(order)
                self.current_position = -1
            # If action == 0 (hold), no order is executed.
        
        self.prev_state = current_state
        self.prev_price = last_price
        self.prev_action = action

start_date = datetime(2020, 1, 1)
end_date = datetime(2023, 12, 31)

broker = Alpaca(ALPACA_CREDS)
strategy = MLTrader(name="mlstrat", broker=broker, parameters={
                    "symbol": "SPY", "cash_at_risk": 0.5})
strategy.backtest(YahooDataBacktesting, start_date, end_date,
                  parameters={"symbol": "SPY", "cash_at_risk": 0.5})
