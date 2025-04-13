import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from datetime import timedelta


# --------------------------
# Helper Functions
# --------------------------

def load_ticker_options(filename, default_list):
    """
    Load tickers from an external file (one ticker per line).
    If the file does not exist, return the default_list.
    """
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            tickers = [line.strip() for line in f if line.strip()]
        return tickers
    else:
        return default_list


@st.cache_data
def get_data(ticker, start_date, end_date):
    """
    Load historical data for a given ticker between start_date and end_date.
    """
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        # If the data has a MultiIndex on the columns, drop the lower level:
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()


def set_page_config():
    """
    Set page configuration and Dark Mode CSS.
    """
    st.set_page_config(page_title='RiskReturnX Portfolio Analyzer', layout='wide', page_icon='📈')
    st.markdown(
        """
        <style>
        body { background-color: #111; color: #fff; }
        h1, h2, h3, h4 { color: #0f9d58; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_portfolio_data(portfolio, start_date, end_date):
    """
    For each asset in the portfolio (Ticker, Invested Money), load historical Close data and compute the
    portfolio value time series.
    """
    value_series = {}
    latest_values = {}
    for idx, row in portfolio.iterrows():
        ticker = row["Ticker"]
        invested_money = row["Invested Money"]
        data = get_data(ticker, start_date, end_date)
        if not data.empty and "Close" in data.columns:
            current_price = data["Close"].iloc[-1]
            shares = invested_money / current_price  # Convert invested money to shares
            value_series[ticker] = data["Close"] * shares
            latest_values[ticker] = data["Close"].iloc[-1] * shares
    if not value_series:
        return None, None
    portfolio_df = pd.concat(value_series, axis=1)
    portfolio_df = portfolio_df.fillna(method='ffill')
    portfolio_df["Total Value"] = portfolio_df.sum(axis=1)
    return portfolio_df, latest_values


def calculate_portfolio_metrics(portfolio_series):
    """
    Calculate key portfolio metrics: annualized volatility, Sharpe Ratio, VaR and maximum drawdown.
    """
    returns = portfolio_series.pct_change().dropna()
    volatility = np.std(returns) * np.sqrt(252)
    sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) != 0 else 0
    var_95 = np.percentile(returns, 5)
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
    return volatility, sharpe, var_95, max_drawdown


def get_rating(value, thresholds, rating_labels):
    """
    Return a rating based on a list of thresholds (assumed in ascending order) and corresponding labels.
    """
    for i, thresh in enumerate(thresholds):
        if value <= thresh:
            return rating_labels[i]
    return rating_labels[-1]


def get_colored_rating(rating):
    """
    Return the rating as HTML with matching colored emoji and text.
    """
    rating_map = {
        "Excellent": "<span style='color:green; font-weight:bold;'>✅ Excellent</span>",
        "Good": "<span style='color:green; font-weight:bold;'>✔️ Good</span>",
        "Medium": "<span style='color:orange; font-weight:bold;'>⚠️ Medium</span>",
        "Bad": "<span style='color:red; font-weight:bold;'>❌ Bad</span>"
    }
    return rating_map.get(rating, rating)


def filter_time_window(df, time_window):
    """
    Filter the DataFrame based on the chosen time window.
    """
    if time_window == "max":
        return df
    last_date = df.index.max()
    if time_window == "1 week":
        delta = timedelta(weeks=1)
    elif time_window == "1 month":
        delta = timedelta(days=30)
    elif time_window == "6 months":
        delta = timedelta(days=182)
    elif time_window == "1 year":
        delta = timedelta(days=365)
    else:
        delta = timedelta(days=0)
    start_date = last_date - delta
    return df[df.index >= start_date]


def generate_feedback(volatility, sharpe, max_drawdown, liquidity):
    """
    Generate detailed feedback based on portfolio metrics with nuanced ratings and matching colored emojis.
    """
    feedback = ""
    # Define Emoji als HTML (matching Farben)
    emoji_excellent = "<span style='color:green'>✅</span>"
    emoji_good = "<span style='color:green'>✔️</span>"
    emoji_medium = "<span style='color:orange'>⚠️</span>"
    emoji_bad = "<span style='color:red'>❌</span>"

    # Volatility Feedback
    if volatility > 0.6:
        feedback += f"{emoji_bad} Volatility is very high. Your portfolio experiences extreme swings. Consider more stable assets.<br>"
    elif volatility > 0.4:
        feedback += f"{emoji_medium} Volatility is high. Significant fluctuations observed.<br>"
    elif volatility > 0.2:
        feedback += f"{emoji_good} Volatility is moderate. Risk is within an acceptable range.<br>"
    else:
        feedback += f"{emoji_excellent} Volatility is very low. Your portfolio is very stable, though it might miss out on higher returns.<br>"

    # Sharpe Ratio Feedback
    if sharpe < 0.5:
        feedback += f"{emoji_bad} Sharpe Ratio is very low. Returns do not sufficiently compensate for the risk.<br>"
    elif sharpe < 1.0:
        feedback += f"{emoji_medium} Sharpe Ratio is low. There is room for better diversification.<br>"
    elif sharpe < 2.0:
        feedback += f"{emoji_good} Sharpe Ratio is good. Decent risk-adjusted returns.<br>"
    else:
        feedback += f"{emoji_excellent} Sharpe Ratio is excellent. Outstanding performance relative to the risk taken!<br>"

    # Maximum Drawdown Feedback (smaller negative ist besser)
    if max_drawdown < -0.8:
        feedback += f"{emoji_bad} Max Drawdown is extremely high. Severe losses observed.<br>"
    elif max_drawdown < -0.5:
        feedback += f"{emoji_medium} Max Drawdown is high. Consider strategies to protect against downside risk.<br>"
    elif max_drawdown < -0.2:
        feedback += f"{emoji_good} Max Drawdown is moderate. Losses are contained.<br>"
    else:
        feedback += f"{emoji_excellent} Max Drawdown is very low. Excellent downside protection.<br>"

    # Liquidity Feedback
    if liquidity < 0.1:
        feedback += f"{emoji_bad} Liquidity is very low. Consider increasing liquid assets.<br>"
    elif liquidity < 0.2:
        feedback += f"{emoji_medium} Liquidity is low. Increasing liquidity could be beneficial.<br>"
    else:
        feedback += f"{emoji_excellent} Liquidity is high. Well-positioned for short-term needs.<br>"

    return feedback


def plot_portfolio_value(portfolio_df, title="Portfolio Value Over Time"):
    """
    Create a line chart showing the evolution of the portfolio's total value.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=portfolio_df.index, y=portfolio_df["Total Value"], mode='lines', name='Portfolio Value'
    ))
    fig.update_layout(template='plotly_dark', title=title, xaxis_title='Date', yaxis_title='Total Value (USD)')
    st.plotly_chart(fig, use_container_width=True)


def plot_allocation(latest_values):
    """
    Create a pie chart showing the current portfolio allocation.
    """
    labels = list(latest_values.keys())
    values = list(latest_values.values())
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
    fig.update_layout(title='Current Portfolio Allocation', template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)


def time_window_buttons_inline(prefix=""):
    selected = "max"  # Default
    time_options = {
        "1W": "1 week",
        "1M": "1 month",
        "6M": "6 months",
        "1Y": "1 year",
        "Max": "max"
    }

    cols = st.columns([1,1,1,1,1,8])  # Last column takes up more space to push buttons to the right
    for i, (label, value) in enumerate(time_options.items()):
        if cols[i].button(label, key=f"{prefix}_{label}"):
            selected = value
    return selected



# --------------------------
# Main Application
# --------------------------

def main():
    set_page_config()
    st.title('📈 RiskReturnX - Portfolio Analyzer & Optimizer')

    # Load ticker options from external files or fallback lists
    default_stock_options = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA"]
    default_crypto_options = ["BTC-USD", "ETH-USD", "ADA-USD"]

    stock_options = load_ticker_options("stocks_etfs.txt", default_stock_options)
    crypto_options = load_ticker_options("cryptos.txt", default_crypto_options)

    # --------------------------
    # Stock Portfolio Input with Searchable Dropdown
    # --------------------------
    st.markdown("## Stock Portfolio Input")
    st.write("Select your stocks from the dropdown (type to search) and enter the invested amount for each:")

    selected_stocks = st.multiselect("Search and select stocks", options=stock_options, default=["AAPL", "MSFT"])

    stocks_investments = {}
    for ticker in selected_stocks:
        amount = st.number_input(f"Invested Amount for {ticker} (in USD)", min_value=0.0, value=1000.0,
                                 key=f"stock_{ticker}")
        stocks_investments[ticker] = amount

    if stocks_investments:
        stocks_portfolio = pd.DataFrame({
            "Ticker": list(stocks_investments.keys()),
            "Invested Money": list(stocks_investments.values())
        })
    else:
        stocks_portfolio = pd.DataFrame(columns=["Ticker", "Invested Money"])

    # --------------------------
    # Crypto Portfolio Input with Searchable Dropdown
    # --------------------------
    st.markdown("## Crypto Portfolio Input")
    st.write("Select your cryptocurrencies from the dropdown (type to search) and enter the invested amount for each:")

    selected_cryptos = st.multiselect("Search and select cryptocurrencies", options=crypto_options,
                                      default=["BTC-USD", "ETH-USD"])

    crypto_investments = {}
    for ticker in selected_cryptos:
        amount = st.number_input(f"Invested Amount for {ticker} (in USD)", min_value=0.0, value=1000.0,
                                 key=f"crypto_{ticker}")
        crypto_investments[ticker] = amount

    if crypto_investments:
        crypto_portfolio = pd.DataFrame({
            "Ticker": list(crypto_investments.keys()),
            "Invested Money": list(crypto_investments.values())
        })
    else:
        crypto_portfolio = pd.DataFrame(columns=["Ticker", "Invested Money"])

    # --------------------------
    # Analysis Data
    # --------------------------
    st.markdown("## Analysis Data")
    start_date = st.date_input('Analysis Start Date', pd.to_datetime('2022-01-01'))
    end_date = st.date_input('Analysis End Date', pd.to_datetime('today'))

    # --------------------------
    # Retrieve Portfolio Data
    # --------------------------
    stocks_df, stocks_latest = None, None
    if not stocks_portfolio.empty:
        stocks_df, stocks_latest = get_portfolio_data(stocks_portfolio, start_date, end_date)

    crypto_df, crypto_latest = get_portfolio_data(crypto_portfolio, start_date, end_date)

    if stocks_df is None and crypto_df is None:
        st.error("No valid data found for the entered positions.")
        return

    # Merge portfolios
    if stocks_df is not None and crypto_df is not None:
        combined_df = stocks_df.join(crypto_df, how="outer", lsuffix="_stocks", rsuffix="_crypto").fillna(
            method='ffill')
        combined_df["Total Value"] = combined_df.filter(like="Total Value").sum(axis=1)
    elif stocks_df is not None:
        combined_df = stocks_df
    else:
        combined_df = crypto_df

    # --------------------------
    # Key Metrics and Ratings
    # --------------------------
    liquidity = 0.2  # Example liquidity value
    volatility, sharpe, var_95, max_drawdown = calculate_portfolio_metrics(combined_df["Total Value"])
    st.markdown("## Key Portfolio Metrics")

    # Define Ratings based on Schwellenwerte:
    vol_rating = get_rating(volatility, [0.2, 0.4, 0.6], ["Excellent", "Good", "Medium", "Bad"])
    sharpe_rating = get_rating(sharpe, [1.0, 2.0], ["Bad", "Good", "Excellent"])
    var_rating = get_rating(var_95, [-0.1, -0.3], ["Excellent", "Good", "Bad"])
    dd_rating = get_rating(max_drawdown, [-0.2, -0.5], ["Excellent", "Good", "Bad"])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Annualized Volatility", f"{volatility:.2%}")
        st.markdown(get_colored_rating(vol_rating), unsafe_allow_html=True)
    with col2:
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
        st.markdown(get_colored_rating(sharpe_rating), unsafe_allow_html=True)
    with col3:
        st.metric("Value at Risk (95%)", f"{var_95:.2%}")
        st.markdown(get_colored_rating(var_rating), unsafe_allow_html=True)
    with col4:
        st.metric("Max Drawdown", f"{max_drawdown:.2%}")
        st.markdown(get_colored_rating(dd_rating), unsafe_allow_html=True)

    st.markdown("## Portfolio Feedback")
    feedback = generate_feedback(volatility, sharpe, max_drawdown, liquidity)
    st.write(feedback, unsafe_allow_html=True)

    # --------------------------
    # Portfolio Allocation
    # --------------------------
    combined_latest_dict = {}
    if stocks_latest is not None:
        combined_latest_dict.update(stocks_latest)
    if crypto_latest is not None:
        combined_latest_dict.update(crypto_latest)

    st.markdown("## Portfolio Allocation")
    plot_allocation(combined_latest_dict)

    # === Combined Portfolio ===
    st.markdown("## Combined Portfolio Development")
    selected_combined = time_window_buttons_inline(prefix="combined")
    plot_portfolio_value(filter_time_window(combined_df, selected_combined), title="Combined Portfolio Value Over Time")

    # === Stocks Portfolio ===
    if stocks_df is not None:
        st.markdown("## Stocks Portfolio Development")
        selected_stocks = time_window_buttons_inline(prefix="stocks")
        plot_portfolio_value(filter_time_window(stocks_df, selected_stocks), title="Stocks Portfolio Value Over Time")

    # === Crypto Portfolio ===
    if crypto_df is not None:
        st.markdown("## Crypto Portfolio Development")
        selected_crypto = time_window_buttons_inline(prefix="crypto")
        plot_portfolio_value(filter_time_window(crypto_df, selected_crypto), title="Crypto Portfolio Value Over Time")


if __name__ == "__main__":
    main()
