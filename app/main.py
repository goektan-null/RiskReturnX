import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --------------------------
# Hilfsfunktionen
# --------------------------

@st.cache_data
def get_data(ticker, start_date, end_date):
    """
    Lädt historische Daten für einen Ticker innerhalb eines bestimmten Zeitraums.
    """
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        # Falls die Daten einen MultiIndex haben, wird die untere Ebene entfernt:
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

def set_page_config():
    """
    Setzt die Seitenkonfiguration und das Dark Mode-CSS.
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
    Lädt für jedes Portfolio-Asset (Ticker, Shares) die historischen Close-Daten und berechnet den
    Gesamtwert des Portfolios.
    """
    value_series = {}
    latest_values = {}
    for idx, row in portfolio.iterrows():
        ticker = row["Ticker"]
        invested_money = row["Invested Money"]  # Investiertes Geld
        data = get_data(ticker, start_date, end_date)
        if not data.empty and "Close" in data.columns:
            current_price = data["Close"].iloc[-1]  # Aktueller Preis
            shares = invested_money / current_price  # Berechne die Anzahl der Shares
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
    Berechnet Kennzahlen für das Portfolio: annualisierte Volatilität, Sharpe Ratio, VaR und maximaler Drawdown.
    """
    returns = portfolio_series.pct_change().dropna()
    volatility = np.std(returns) * np.sqrt(252)
    sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
    var_95 = np.percentile(returns, 5)
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
    return volatility, sharpe, var_95, max_drawdown

def generate_feedback(volatility, sharpe, max_drawdown, liquidity):
    """
    Generiert ein Feedback basierend auf den Portfolio-Kennzahlen.
    """
    feedback = ""
    if volatility > 0.4:
        feedback += "⚠️ Dein Portfolio ist sehr volatil. Mehr Diversifikation könnte helfen, das Risiko zu senken.\n"
    else:
        feedback += "👍 Die Volatilität liegt im moderaten Bereich.\n"

    if sharpe < 1.0:
        feedback += "⚠️ Die risikoadjustierte Rendite ist niedrig. Eine Überprüfung der Positionen könnte sinnvoll sein.\n"
    else:
        feedback += "🚀 Gute risikoadjustierte Performance.\n"

    if max_drawdown < -0.5:
        feedback += "⚠️ Es wurde ein erheblicher Drawdown festgestellt. Eine Risikomanagement-Strategie wird empfohlen.\n"
    else:
        feedback += "✅ Drawdown liegt im akzeptablen Bereich.\n"

    if liquidity < 0.1:
        feedback += "⚠️ Deine Portfolio-Liquidität ist gering. Eine Erhöhung der Liquidität könnte hilfreich sein.\n"
    else:
        feedback += "✅ Die Liquidität deines Portfolios ist ausreichend.\n"

    return feedback

def plot_portfolio_value(portfolio_df, title="Portfolio Value Over Time"):
    """
    Erstellt ein Liniendiagramm der Portfolio-Wertentwicklung.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=portfolio_df.index, y=portfolio_df["Total Value"], mode='lines', name='Portfolio Value'
    ))
    fig.update_layout(template='plotly_dark', title=title, xaxis_title='Date', yaxis_title='Total Value (USD)')
    st.plotly_chart(fig, use_container_width=True)

def plot_allocation(latest_values):
    """
    Erstellt ein Tortendiagramm der aktuellen Portfolio-Allokation.
    """
    labels = list(latest_values.keys())
    values = list(latest_values.values())
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
    fig.update_layout(title='Current Portfolio Allocation', template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)

# --------------------------
# Hauptfunktion
# --------------------------

def main():
    set_page_config()
    st.title('📈 RiskReturnX - Portfolio Analyzer & Optimizer')

    st.markdown("## Portfolio Input")
    st.write("Bearbeite dein Aktien-Portfolio in der Tabelle unten:")
    default_stocks = pd.DataFrame({
        "Ticker": ["AAPL", "MSFT", "TSLA"],
        "Invested Money": [1300, 2200, 3500]  # Investiertes Geld (z.B. 1300$ für AAPL)
    })
    stocks_portfolio = st.data_editor(default_stocks, num_rows="dynamic", use_container_width=True)

    st.markdown("## Krypto Portfolio Input")
    st.write("Bearbeite dein Krypto-Portfolio in der Tabelle unten (nutze z.B. 'BTC-USD' für Bitcoin, 'ETH-USD' für Ethereum):")
    default_crypto = pd.DataFrame({
        "Ticker": ["BTC-USD", "ETH-USD"],
        "Invested Money": [15000, 4000]  # Investiertes Geld
    })
    crypto_portfolio = st.data_editor(default_crypto, num_rows="dynamic", use_container_width=True)

    st.markdown("## Analysedaten")
    start_date = st.date_input('Analyse Start Date', pd.to_datetime('2022-01-01'))
    end_date = st.date_input('Analyse End Date', pd.to_datetime('today'))

    # Daten für Aktien abrufen:
    stocks_df, stocks_latest = get_portfolio_data(stocks_portfolio, start_date, end_date)
    # Daten für Krypto abrufen:
    crypto_df, crypto_latest = get_portfolio_data(crypto_portfolio, start_date, end_date)

    if stocks_df is None and crypto_df is None:
        st.error("Keine gültigen Daten für die eingegebenen Positionen gefunden.")
        return

    # Gesamtportfolio zusammenführen:
    if stocks_df is not None and crypto_df is not None:
        combined_df = stocks_df.join(crypto_df, how="outer", lsuffix="_stocks", rsuffix="_crypto").fillna(method='ffill')
        combined_df["Total Value"] = combined_df.filter(like="Total Value").sum(axis=1)
    elif stocks_df is not None:
        combined_df = stocks_df
    else:
        combined_df = crypto_df

    st.markdown("## Portfolio Value Over Time")
    plot_portfolio_value(stocks_df, title="Stocks Portfolio Value Over Time") if stocks_df is not None else None
    plot_portfolio_value(crypto_df, title="Crypto Portfolio Value Over Time") if crypto_df is not None else None
    plot_portfolio_value(combined_df, title="Combined Portfolio Value Over Time")

    # Portfolio Allokation (für Gesamtportfolio):
    combined_latest = {}
    if stocks_latest is not None:
        combined_latest.update(stocks_latest)
    if crypto_latest is not None:
        combined_latest.update(crypto_latest)
    st.markdown("## Portfolio Allocation")
    plot_allocation(combined_latest)

    # Berechnung der Portfolio-Metriken für das Gesamtportfolio:
    liquidity = 0.2  # Beispielwert für Liquidität
    volatility, sharpe, var_95, max_drawdown = calculate_portfolio_metrics(combined_df["Total Value"])
    st.markdown("## Key Portfolio Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Annualized Volatility", f"{volatility:.2%}")
    col2.metric("Sharpe Ratio", f"{sharpe:.2f}")
    col3.metric("Value at Risk (95%)", f"{var_95:.2%}")
    col4.metric("Max Drawdown", f"{max_drawdown:.2%}")

    st.markdown("## Portfolio Feedback")
    feedback = generate_feedback(volatility, sharpe, max_drawdown, liquidity)
    st.write(feedback)

if __name__ == "__main__":
    main()
