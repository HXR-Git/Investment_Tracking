import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import sqlite3
from datetime import datetime, timedelta
import requests
import numpy as np
import time

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Stock Portfolio Tracker")

# Hide Streamlit's default menu and deploy button
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- Custom Plotly Template ---
custom_template = {
    "layout": {
        "font": {"family": "Arial", "size": 12, "color": "#FFFFFF"},
        "paper_bgcolor": "#1E1E1E",
        "plot_bgcolor": "#2D2D2D",
        "xaxis": {"gridcolor": "#444444", "zerolinecolor": "#444444", "title_font": {"size": 14}},
        "yaxis": {"gridcolor": "#444444", "zerolinecolor": "#444444", "title_font": {"size": 14}},
        "title": {"font": {"size": 18, "color": "#FFFFFF"}},
        "legend": {"font": {"color": "#FFFFFF"}, "bgcolor": "rgba(0,0,0,0)"},
        "margin": {"l": 50, "r": 50, "t": 80, "b": 50},
    }
}

# --- Database Setup ---
def init_db():
    conn = sqlite3.connect("portfolio.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS transactions
                 (stock_name TEXT, symbol TEXT, transaction_type TEXT, exchange TEXT,
                  purchase_date TEXT, purchase_price REAL, quantity REAL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS stocks
                 (stock_name TEXT, symbol TEXT UNIQUE, exchange TEXT,
                  purchase_price REAL, quantity REAL, total_cost REAL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS broker_transactions
                 (transaction_type TEXT, date TEXT, amount REAL)''')
    conn.commit()
    conn.close()

# --- Yahoo Finance API Functions ---
def search_stock(query):
    base_url = "https://query1.finance.yahoo.com/v1/finance/search"
    params = {'q': query, 'quotesCount': 10, 'newsCount': 0}
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(base_url, params=params, headers=headers)
        response.raise_for_status()
        return response.json().get('quotes', [])
    except Exception as e:
        st.error(f"Error searching for stocks: {str(e)}")
        return []

def fetch_data(symbol, retries=3):
    for attempt in range(retries):
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period="1d", interval="1m")
            if data.empty:
                return None
            return data['Close'].iloc[-1]
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1)
                continue
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            return None

def fetch_historical_data(symbol, start_date, end_date, include_volume=False):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(start=start_date, end=end_date, interval='1d')
        if data.empty:
            return None
        if include_volume:
            return data.reset_index()[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        return data.reset_index()[['Date', 'Close']]
    except Exception as e:
        st.error(f"Error fetching historical data for {symbol}: {str(e)}")
        return None

def fetch_historical_pe(symbol, start_date, end_date):
    try:
        ticker = yf.Ticker(symbol)
        hist_data = fetch_historical_data(symbol, start_date, end_date)
        if hist_data is None:
            return None
        eps = ticker.info.get('trailingEps', None)
        if eps is None or eps == 0:
            st.warning(f"No EPS data available for {symbol}.")
            return None
        hist_data['P/E Ratio'] = hist_data['Close'] / eps
        hist_data['Date'] = pd.to_datetime(hist_data['Date'])
        return hist_data[['Date', 'P/E Ratio']]
    except Exception as e:
        st.error(f"Error fetching historical P/E for {symbol}: {str(e)}")
        return None

# --- Technical Indicators ---
def calculate_rsi(df, period=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(df, period=20):
    df['SMA'] = df['Close'].rolling(window=period).mean()
    df['STD'] = df['Close'].rolling(window=period).std()
    df['Upper Band'] = df['SMA'] + (df['STD'] * 2)
    df['Lower Band'] = df['SMA'] - (df['STD'] * 2)
    return df

# --- Portfolio Calculations ---
def calculate_portfolio_return(df_transactions, start_date, end_date, benchmark="^NSEI"):
    date_range = pd.date_range(start=start_date, end=end_date)
    portfolio_returns = []
    benchmark_returns = []
    df_transactions['purchase_date'] = pd.to_datetime(df_transactions['purchase_date'])
    hist_data = {symbol: fetch_historical_data(symbol, start_date, end_date) for symbol in df_transactions['symbol'].unique()}
    benchmark_data = fetch_historical_data(benchmark, start_date, end_date)

    if benchmark_data is None or benchmark_data.empty:
        st.warning("Could not fetch NIFTY 50 data. Showing portfolio returns only.")
        benchmark_data = pd.DataFrame(columns=['Date', 'Close'])

    last_valid_return = 0
    for date in date_range:
        df_up_to_date = df_transactions[df_transactions['purchase_date'].dt.date <= date.date()]
        total_cost = 0
        total_value = 0
        has_data = False

        for symbol in hist_data:
            if hist_data[symbol] is not None and not hist_data[symbol].empty:
                df_symbol = df_up_to_date[df_up_to_date['symbol'] == symbol]
                buys = df_symbol[df_symbol['transaction_type'] == 'Buy']
                sells = df_symbol[df_symbol['transaction_type'] == 'Sell']
                quantity = buys['quantity'].sum() - sells['quantity'].sum()
                if quantity > 0:
                    price_row = hist_data[symbol][hist_data[symbol]['Date'].dt.date == date.date()]
                    if not price_row.empty:
                        price = price_row['Close'].iloc[0]
                        total_value += quantity * price
                        total_cost += quantity * (buys['purchase_price'] * buys['quantity']).sum() / buys['quantity'].sum() if buys['quantity'].sum() > 0 else 0
                        has_data = True

        if has_data and total_cost > 0 and total_value > 0:
            return_pct = (total_value - total_cost) / total_cost * 100
            last_valid_return = return_pct
            portfolio_returns.append({'Date': date, 'Return %': return_pct, 'Type': 'Portfolio'})
        else:
            portfolio_returns.append({'Date': date, 'Return %': last_valid_return, 'Type': 'Portfolio'})

    if not benchmark_data.empty:
        benchmark_data['Return %'] = (benchmark_data['Close'] - benchmark_data['Close'].iloc[0]) / benchmark_data['Close'].iloc[0] * 100
        benchmark_returns = [{'Date': row['Date'], 'Return %': row['Return %'], 'Type': 'NIFTY 50'} for _, row in benchmark_data.iterrows()]

    portfolio_df = pd.DataFrame(portfolio_returns)
    benchmark_df = pd.DataFrame(benchmark_returns)
    combined_df = pd.concat([portfolio_df, benchmark_df], ignore_index=True)
    return combined_df

def calculate_one_day_change(df_current_stocks):
    symbols = df_current_stocks['symbol'].unique()
    today = datetime.today().strftime('%Y-%m-%d')
    yesterday = (datetime.today() - timedelta(days=5)).strftime('%Y-%m-%d')
    hist_data = {}

    for symbol in symbols:
        try:
            data = fetch_historical_data(symbol, yesterday, today)
            if data is not None and len(data) >= 2:
                hist_data[symbol] = data
            else:
                st.warning(f"Insufficient data for {symbol} to calculate 1-day change.")
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")

    if not hist_data:
        return None, None

    today_value = 0
    yesterday_value = 0

    for symbol in symbols:
        if symbol in hist_data and len(hist_data[symbol]) >= 2:
            quantity = df_current_stocks[df_current_stocks['symbol'] == symbol]['quantity'].iloc[0]
            if quantity > 0:
                hist_df = hist_data[symbol]
                today_price = hist_df['Close'].iloc[-1]
                yesterday_price = hist_df['Close'].iloc[-2]
                today_value += quantity * today_price
                yesterday_value += quantity * yesterday_price

    if yesterday_value > 0:
        one_day_change = today_value - yesterday_value
        one_day_change_pct = (one_day_change / yesterday_value) * 100
        return one_day_change, one_day_change_pct
    return None, None

def calculate_realized_profit_loss(df_transactions):
    df_transactions['purchase_date'] = pd.to_datetime(df_transactions['purchase_date'])
    realized_pl = 0
    for symbol in df_transactions['symbol'].unique():
        df_symbol = df_transactions[df_transactions['symbol'] == symbol].sort_values('purchase_date')
        buys = df_symbol[df_symbol['transaction_type'] == 'Buy']
        sells = df_symbol[df_symbol['transaction_type'] == 'Sell']
        if not sells.empty and not buys.empty:
            total_bought = buys['quantity'].sum()
            avg_buy_price = (buys['purchase_price'] * buys['quantity']).sum() / total_bought
            for _, sell in sells.iterrows():
                sell_value = sell['purchase_price'] * sell['quantity']
                cost_basis = avg_buy_price * sell['quantity']
                realized_pl += sell_value - cost_basis
    return realized_pl

def calculate_realized_pl_table(df_transactions):
    df_transactions['purchase_date'] = pd.to_datetime(df_transactions['purchase_date'])
    realized_pl_data = []
    for symbol in df_transactions['symbol'].unique():
        df_symbol = df_transactions[df_transactions['symbol'] == symbol].sort_values('purchase_date')
        buys = df_symbol[df_symbol['transaction_type'] == 'Buy']
        sells = df_symbol[df_symbol['transaction_type'] == 'Sell']
        if not buys.empty:
            avg_buy_price = (buys['purchase_price'] * buys['quantity']).sum() / buys['quantity'].sum()
            for _, buy in buys.iterrows():
                realized_pl_data.append({
                    'Symbol': symbol,
                    'Stock Name': buy['stock_name'],
                    'Type': 'Buy',
                    'Date': buy['purchase_date'],
                    'Price (₹)': buy['purchase_price'],
                    'Quantity': buy['quantity'],
                    'Realized P/L (₹)': 0.0
                })
        if not sells.empty and not buys.empty:
            for _, sell in sells.iterrows():
                sell_value = sell['purchase_price'] * sell['quantity']
                cost_basis = avg_buy_price * sell['quantity']
                realized_pl = sell_value - cost_basis
                realized_pl_data.append({
                    'Symbol': symbol,
                    'Stock Name': sell['stock_name'],
                    'Type': 'Sell',
                    'Date': sell['purchase_date'],
                    'Price (₹)': sell['purchase_price'],
                    'Quantity': sell['quantity'],
                    'Realized P/L (₹)': realized_pl
                })
    return pd.DataFrame(realized_pl_data)

# --- New Graph Calculations ---
def calculate_volatility_return(df_valid):
    start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
    end_date = datetime.today().strftime('%Y-%m-%d')
    volatility_return_data = []

    for _, row in df_valid.iterrows():
        symbol = row['symbol']
        hist_df = fetch_historical_data(symbol, start_date, end_date)
        if hist_df is not None and len(hist_df) > 1:
            daily_returns = hist_df['Close'].pct_change().dropna()
            annualized_volatility = daily_returns.std() * np.sqrt(252) * 100  # Annualized volatility (%)
            total_return = ((hist_df['Close'].iloc[-1] - hist_df['Close'].iloc[0]) / hist_df['Close'].iloc[0]) * 100
            volatility_return_data.append({
                'Symbol': symbol,
                'Stock Name': row['stock_name'],
                'Volatility (%)': annualized_volatility,
                'Return (%)': total_return,
                'Value (₹)': row['value']
            })
    return pd.DataFrame(volatility_return_data)

def calculate_price_change_distribution(symbol, start_date, end_date):
    hist_df = fetch_historical_data(symbol, start_date, end_date)
    if hist_df is not None:
        hist_df['Price Change (%)'] = hist_df['Close'].pct_change() * 100
        return hist_df.dropna()
    return None

def calculate_pl_breakdown(df_valid):
    pl_data = df_valid[['stock_name', 'unrealized_pl']].copy()
    pl_data.columns = ['Stock Name', 'Unrealized P/L (₹)']
    total_pl = pl_data['Unrealized P/L (₹)'].sum()
    pl_data = pd.concat([pl_data, pd.DataFrame([{'Stock Name': 'Total', 'Unrealized P/L (₹)': total_pl}])])
    return pl_data

def calculate_correlation_matrix(df_valid):
    start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
    end_date = datetime.today().strftime('%Y-%m-%d')
    symbols = df_valid['symbol'].tolist()
    close_prices = pd.DataFrame()

    for symbol in symbols:
        hist_df = fetch_historical_data(symbol, start_date, end_date)
        if hist_df is not None:
            close_prices[symbol] = hist_df['Close']

    if len(close_prices.columns) > 1:
        correlation_matrix = close_prices.pct_change().corr()
        return correlation_matrix
    return None

# --- Database Operations ---
def update_stocks(stock_name, symbol, exchange, transaction_type, purchase_price, quantity):
    try:
        conn = sqlite3.connect("portfolio.db")
        c = conn.cursor()
        c.execute("SELECT quantity, purchase_price, total_cost FROM stocks WHERE symbol=?", (symbol,))
        existing = c.fetchone()

        if existing:
            existing_quantity, existing_price, existing_cost = existing
            if transaction_type == "Buy":
                new_quantity = existing_quantity + quantity
                new_total_cost = existing_cost + (purchase_price * quantity)
                new_avg_price = new_total_cost / new_quantity if new_quantity > 0 else 0
                c.execute("UPDATE stocks SET stock_name=?, exchange=?, purchase_price=?, quantity=?, total_cost=? WHERE symbol=?",
                          (stock_name, exchange, new_avg_price, new_quantity, new_total_cost, symbol))
            elif transaction_type == "Sell":
                if existing_quantity >= quantity:
                    new_quantity = existing_quantity - quantity
                    new_total_cost = existing_cost if new_quantity == 0 else existing_price * new_quantity
                    c.execute("UPDATE stocks SET stock_name=?, exchange=?, purchase_price=?, quantity=?, total_cost=? WHERE symbol=?",
                              (stock_name, exchange, existing_price, new_quantity, new_total_cost, symbol))
                    if new_quantity == 0:
                        c.execute("DELETE FROM stocks WHERE symbol=?", (symbol,))
                else:
                    st.error(f"Cannot sell {quantity} of {stock_name} ({symbol}) - Only {existing_quantity} available.")
                    return False
        else:
            if transaction_type == "Buy":
                total_cost = purchase_price * quantity
                c.execute("INSERT INTO stocks VALUES (?, ?, ?, ?, ?, ?)",
                          (stock_name, symbol, exchange, purchase_price, quantity, total_cost))
            else:
                st.error(f"Cannot sell {stock_name} ({symbol}) - No existing position.")
                return False
        conn.commit()
        return True
    except sqlite3.Error as e:
        st.error(f"Database error: {str(e)}")
        return False
    finally:
        conn.close()

# --- Add Transaction Forms ---
def add_stock_transaction_form():
    st.subheader("Add Stock Transaction")
    stock_query = st.text_input("Search Stock")
    symbol_options = []
    selected_symbol = None
    selected_exchange = None
    selected_name = ""

    if stock_query and len(stock_query) >= 2:
        with st.spinner("Searching..."):
            results = search_stock(stock_query)
        if results:
            symbol_options = [f"{r.get('shortname', 'Unknown')} - {r['symbol']} ({r.get('exchDisp', 'Unknown')})"
                              for r in results if 'symbol' in r]
            selected_result = st.selectbox("Select Stock:", ["Select..."] + symbol_options)
            if selected_result != "Select...":
                selected_symbol = selected_result.split(" - ")[1].split(" (")[0]
                selected_name = selected_result.split(" - ")[0]
                selected_exchange = "NSE" if ".NS" in selected_symbol else "BSE" if ".BO" in selected_symbol else "Other"
                current_price = fetch_data(selected_symbol)
                if current_price:
                    st.success(f"Current Price: ₹{current_price:.2f}")
                else:
                    st.warning("Could not fetch price.")

    stock_name = st.text_input("Company Name", value=selected_name)
    symbol = st.text_input("Symbol", value=selected_symbol or "")
    exchange = st.selectbox("Exchange", ["NSE", "BSE", "Other"], index=["NSE", "BSE", "Other"].index(selected_exchange) if selected_exchange else 0)
    transaction_type = st.selectbox("Type", ["Buy", "Sell"])
    purchase_date = st.date_input("Date", max_value=datetime.today())
    purchase_price = st.number_input("Price", min_value=0.0, step=0.01)
    quantity = st.number_input("Quantity", min_value=0.0, step=0.01, value=1.0)

    if st.button("Add Stock"):
        if not stock_name or not symbol:
            st.error("Enter company name and symbol.")
        elif purchase_price <= 0 or quantity <= 0:
            st.error("Price and quantity must be positive.")
        elif fetch_data(symbol) is None:
            st.error("Invalid symbol.")
        elif purchase_date > datetime.today().date():
            st.error("Date cannot be in the future.")
        else:
            conn = sqlite3.connect("portfolio.db")
            c = conn.cursor()
            c.execute("INSERT INTO transactions VALUES (?, ?, ?, ?, ?, ?, ?)",
                      (stock_name, symbol, transaction_type, exchange, str(purchase_date), purchase_price, quantity))
            conn.commit()
            conn.close()
            if update_stocks(stock_name, symbol, exchange, transaction_type, purchase_price, quantity):
                st.success(f"{transaction_type} of {stock_name} added.")
                st.session_state.show_sidebar = False
                st.rerun()

def add_broker_transaction_form():
    st.subheader("Add Broker Transaction")
    transaction_type = st.selectbox("Type", ["Deposit", "Withdraw"], key="broker_type")
    date = st.date_input("Date", max_value=datetime.today(), key="broker_date")
    amount = st.number_input("Amount", min_value=0.0, step=0.01, key="broker_amount")
    if st.button("Add Transaction"):
        if amount <= 0:
            st.error("Amount must be positive.")
        elif date > datetime.today().date():
            st.error("Date cannot be in the future.")
        else:
            conn = sqlite3.connect("portfolio.db")
            c = conn.cursor()
            c.execute("INSERT INTO broker_transactions VALUES (?, ?, ?)",
                      (transaction_type, str(date), amount))
            conn.commit()
            conn.close()
            st.success(f"{transaction_type} of ₹{amount} added.")
            st.session_state.show_sidebar = False
            st.rerun()

# --- Styling Functions ---
def color_pl(val):
    color = '#00CC96' if val > 0 else '#EF553B' if val < 0 else '#FFFFFF'
    return f'color: {color}'

def color_transaction(val):
    color = '#00CC96' if val == 'Buy' else '#EF553B' if val == 'Sell' else '#FFFFFF'
    return f'color: {color}'

def color_fund(val):
    color = '#00CC96' if val == 'Deposit' else '#EF553B' if val == 'Withdraw' else '#FFFFFF'
    return f'color: {color}'

# --- Main App ---
def main():
    init_db()
    st.title("Stock Portfolio Tracker", anchor="top")

    # Navigation with Add and Refresh Buttons
    col1, col2, col3, col4, col5, col6, col7 = st.columns([1, 1, 1, 1, 1, 1, 0.5])
    with col1:
        dashboard_clicked = st.button("Dashboard")
    with col2:
        stock_activity_clicked = st.button("Stock Activity Ledger")
    with col3:
        fund_monitor_clicked = st.button("Fund Monitor")
    with col4:
        unrealized_pl_clicked = st.button("Unrealized P/L")
    with col5:
        realized_pl_clicked = st.button("Realized P/L")
    with col6:
        add_clicked = st.button("Add")
    with col7:
        if 'refreshing' not in st.session_state:
            st.session_state.refreshing = False
        refresh_clicked = st.button("🔄", key="refresh_button")
        if refresh_clicked:
            st.session_state.refreshing = True
            with st.spinner("Refreshing data..."):
                try:
                    st.cache_data.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Refresh failed: {str(e)}")
            st.session_state.refreshing = False

    # CSS for spinning refresh button
    st.markdown("""
        <style>
        button[kind="primary"][key="refresh_button"] {
            transition: transform 0.5s;
        }
        button[kind="primary"][key="refresh_button"]:active {
            transform: rotate(360deg);
        }
        </style>
    """, unsafe_allow_html=True)

    # View and Sidebar State Management
    if 'view' not in st.session_state:
        st.session_state.view = "Dashboard"
    if 'show_sidebar' not in st.session_state:
        st.session_state.show_sidebar = False

    if dashboard_clicked:
        st.session_state.view = "Dashboard"
        st.session_state.show_sidebar = False
    elif stock_activity_clicked:
        st.session_state.view = "Stock Activity Ledger"
        st.session_state.show_sidebar = False
    elif fund_monitor_clicked:
        st.session_state.view = "Fund Monitor"
        st.session_state.show_sidebar = False
    elif unrealized_pl_clicked:
        st.session_state.view = "Unrealized P/L"
        st.session_state.show_sidebar = False
    elif realized_pl_clicked:
        st.session_state.view = "Realized P/L"
        st.session_state.show_sidebar = False
    elif add_clicked:
        current_state = st.session_state.show_sidebar
        st.session_state.show_sidebar = not current_state

    # Fetch Data
    conn = sqlite3.connect("portfolio.db")
    df_stocks = pd.read_sql_query("SELECT * FROM stocks", conn)
    df_transactions = pd.read_sql_query("SELECT * FROM transactions", conn)
    df_broker = pd.read_sql_query("SELECT * FROM broker_transactions", conn)
    conn.close()

    # Sidebar with Add Dropdown
    if st.session_state.show_sidebar:
        with st.sidebar:
            with st.expander("Add Transactions", expanded=True):
                add_stock_transaction_form()
                st.markdown("---")
                add_broker_transaction_form()
            st.write(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Password Setup for Clear All
    if 'clear_password' not in st.session_state:
        st.session_state.clear_password = None
    if 'is_unlocked' not in st.session_state:
        st.session_state.is_unlocked = False

    if st.session_state.clear_password is None:
        with st.form("set_password_form"):
            new_password = st.text_input("Set a password to protect 'Clear All'", type="password")
            if st.form_submit_button("Set Password"):
                if new_password:
                    st.session_state.clear_password = new_password
                    st.success("Password set successfully!")
                    st.rerun()
                else:
                    st.error("Password cannot be empty.")

    # Color Mapping for Stocks
    stock_colors = {symbol: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
                    for i, symbol in enumerate(df_stocks['symbol'].unique())}

    with st.container():
        if st.session_state.view == "Dashboard":
            if not df_stocks.empty:
                df_current_stocks = df_stocks[df_stocks['quantity'] > 0].copy()
                if not df_current_stocks.empty:
                    with st.spinner("Updating Prices..."):
                        prices = [fetch_data(symbol) for symbol in df_current_stocks['symbol']]
                        df_current_stocks['current_price'] = prices

                    df_valid = df_current_stocks.dropna(subset=['current_price']).copy()
                    if not df_valid.empty:
                        df_valid['value'] = df_valid['current_price'] * df_valid['quantity']
                        df_valid['unrealized_pl'] = (df_valid['current_price'] - df_valid['purchase_price']) * df_valid['quantity']
                        df_valid['return_percentage'] = df_valid.apply(
                            lambda row: (row['unrealized_pl'] / row['total_cost'] * 100) if row['total_cost'] > 0 else 0, axis=1)
                        df_valid['sector'] = [yf.Ticker(symbol).info.get('sector', 'Unknown') for symbol in df_valid['symbol']]

                        total_investment = df_valid['total_cost'].sum()
                        total_value = df_valid['value'].sum()
                        total_unrealized_pl = df_valid['unrealized_pl'].sum()
                        realized_pl = calculate_realized_profit_loss(df_transactions)
                        one_day_change, one_day_change_pct = calculate_one_day_change(df_valid)

                        # Metrics
                        col1, col2, col3, col4, col5 = st.columns(5)
                        col1.metric("Total Investment", f"₹{total_investment:.2f}")
                        col2.metric("Current Value", f"₹{total_value:.2f}")
                        col3.metric("1-Day Change", f"₹{one_day_change:.2f}" if one_day_change is not None else "N/A",
                                    f"{one_day_change_pct:.2f}%" if one_day_change_pct is not None else "N/A",
                                    delta_color="normal" if one_day_change is None or one_day_change >= 0 else "inverse")
                        col4.metric("Return", f"₹{total_unrealized_pl:.2f}",
                                    f"{(total_unrealized_pl / total_investment) * 100:.2f}%" if total_investment > 0 else "N/A")
                        col5.metric("Realized P/L", f"₹{realized_pl:.2f}")

                        # Graphs
                        col1, col2 = st.columns(2)

                        # Graph 1: Portfolio Allocation
                        with col1:
                            st.subheader("Portfolio Allocation")
                            allocation_type = st.selectbox("View:", ["Stock", "Sector"], index=0, key="alloc", label_visibility="collapsed")
                            if allocation_type == "Stock":
                                fig_pie = px.pie(df_valid, values='value', names='stock_name', title="Stock Allocation",
                                                 color='symbol', color_discrete_map=stock_colors)
                            else:
                                sector_values = df_valid.groupby('sector')['value'].sum().reset_index()
                                fig_pie = px.pie(sector_values, values='value', names='sector', title="Sector Diversification",
                                                 color_discrete_sequence=px.colors.qualitative.Plotly)
                            fig_pie.update_layout(template=custom_template, height=400)
                            st.plotly_chart(fig_pie, use_container_width=True)
                            st.markdown("""
                                <style>
                                div[data-testid="stSelectbox"]:nth-child(2) {
                                    position: absolute;
                                    bottom: 10px;
                                    right: 10px;
                                    width: 100px;
                                }
                                </style>
                                """, unsafe_allow_html=True)

                        # Graph 2: Stock Performance
                        with col2:
                            st.subheader("Stock Performance")
                            fig_bar = px.bar(df_valid, x='stock_name', y='return_percentage', title="Stock Returns (%)",
                                             labels={'return_percentage': 'Return (%)'}, color='symbol',
                                             color_discrete_map=stock_colors)
                            fig_bar.update_traces(width=0.5)
                            if len(df_valid) > 4:
                                fig_bar.update_layout(xaxis=dict(autorange=False, range=[-0.5, 3.5], rangeslider_visible=True))
                            fig_bar.update_layout(template=custom_template, height=400)
                            st.plotly_chart(fig_bar, use_container_width=True)

                        # Graph 3: Portfolio vs NIFTY 50 Returns
                        st.subheader("Portfolio vs NIFTY 50 Returns (1-Year)")
                        start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
                        end_date = datetime.today().strftime('%Y-%m-%d')
                        portfolio_df = calculate_portfolio_return(df_transactions, start_date, end_date)
                        if not portfolio_df.empty and len(portfolio_df['Date'].unique()) > 1:
                            fig_portfolio = go.Figure()
                            portfolio_data = portfolio_df[portfolio_df['Type'] == 'Portfolio']
                            nifty_data = portfolio_df[portfolio_df['Type'] == 'NIFTY 50']

                            fig_portfolio.add_trace(go.Scatter(
                                x=portfolio_data['Date'],
                                y=portfolio_data['Return %'],
                                mode='lines',
                                name='Portfolio',
                                line_color='#00CC96',
                                customdata=nifty_data['Return %'].reindex(portfolio_data.index, method='ffill'),
                                hovertemplate='<b>Date</b>: %{x}<br><b>Portfolio Return</b>: %{y:.2f}%<br><b>NIFTY 50 Return</b>: %{customdata:.2f}%'
                            ))

                            if not nifty_data.empty:
                                fig_portfolio.add_trace(go.Scatter(
                                    x=nifty_data['Date'],
                                    y=nifty_data['Return %'],
                                    mode='lines',
                                    name='NIFTY 50',
                                    line_color='#EF553B',
                                    customdata=portfolio_data['Return %'].reindex(nifty_data.index, method='ffill'),
                                    hovertemplate='<b>Date</b>: %{x}<br><b>Portfolio Return</b>: %{customdata:.2f}%<br><b>NIFTY 50 Return</b>: %{y:.2f}%'
                                ))

                            fig_portfolio.update_layout(
                                title="Portfolio vs NIFTY 50 Returns (1-Year)",
                                yaxis_title="Return (%)",
                                template=custom_template,
                                legend=dict(x=0.01, y=0.99),
                                hovermode='x unified'
                            )
                            st.plotly_chart(fig_portfolio, use_container_width=True)
                        else:
                            st.warning("Insufficient data to plot Portfolio vs NIFTY 50 returns.")

                        # New Graph 4: Selectable Graph Type
                        st.subheader("Risk Analysis")
                        graph4_options = ["Volatility vs. Return (Scatter)", "Price Change Distribution (Histogram)"]
                        graph4_type = st.selectbox("Select Graph Type:", graph4_options, key="graph4")
                        if graph4_type == "Volatility vs. Return (Scatter)":
                            volatility_return_df = calculate_volatility_return(df_valid)
                            if not volatility_return_df.empty:
                                fig_scatter = px.scatter(volatility_return_df, x="Volatility (%)", y="Return (%)",
                                                         size="Value (₹)", color="Symbol", text="Stock Name",
                                                         title="Volatility vs. Return (1-Year)",
                                                         color_discrete_map=stock_colors,
                                                         hover_data={"Value (₹)": ":.2f"})
                                fig_scatter.update_traces(textposition='top center')
                                fig_scatter.update_layout(template=custom_template, height=400)
                                st.plotly_chart(fig_scatter, use_container_width=True)
                            else:
                                st.warning("Insufficient data for Volatility vs. Return plot.")
                        elif graph4_type == "Price Change Distribution (Histogram)":
                            stock_options = [f"{row['stock_name']} ({row['symbol']})" for _, row in df_valid.iterrows()]
                            selected_stock = st.selectbox("Select Stock:", stock_options, key="hist_stock")
                            selected_symbol = selected_stock.split(" (")[1][:-1]
                            hist_df = calculate_price_change_distribution(selected_symbol, start_date, end_date)
                            if hist_df is not None:
                                fig_hist = px.histogram(hist_df, x="Price Change (%)", nbins=50,
                                                        title=f"Price Change Distribution: {selected_stock}",
                                                        color_discrete_sequence=[stock_colors[selected_symbol]])
                                fig_hist.update_layout(template=custom_template, height=400, bargap=0.2)
                                st.plotly_chart(fig_hist, use_container_width=True)
                            else:
                                st.warning("Insufficient data for Price Change Distribution.")

                        # Graph 5: Historical Price Trend with Volume (Candlestick)
                        st.subheader("Technical Analysis")
                        stock_options = [f"{row['stock_name']} ({row['symbol']})" for _, row in df_valid.iterrows()]
                        selected_stock = st.selectbox("Stock for Candlestick:", stock_options, key="price")
                        selected_symbol = selected_stock.split(" (")[1][:-1]
                        hist_df = fetch_historical_data(selected_symbol, start_date, end_date, include_volume=True)
                        if hist_df is not None:
                            fig_candle = go.Figure()
                            fig_candle.add_trace(go.Bar(x=hist_df['Date'], y=hist_df['Volume'], name='Volume', opacity=0.5, yaxis='y2', marker_color='#636EFA'))
                            fig_candle.add_trace(go.Candlestick(x=hist_df['Date'], open=hist_df['Open'], high=hist_df['High'],
                                                                low=hist_df['Low'], close=hist_df['Close'], name='Price',
                                                                increasing_line_color='#00CC96', decreasing_line_color='#EF553B'))
                            fig_candle.update_layout(title=f"Candlestick: {selected_stock} (1-Year)", yaxis_title="Price (₹)",
                                                     yaxis2=dict(title="Volume", overlaying='y', side='right'),
                                                     template=custom_template, legend=dict(x=0.01, y=0.99))
                            st.plotly_chart(fig_candle, use_container_width=True)

                        # Graph 6: RSI Trend
                        selected_rsi_stock = st.selectbox("Stock for RSI:", stock_options, key="rsi")
                        selected_symbol = selected_rsi_stock.split(" (")[1][:-1]
                        hist_df = fetch_historical_data(selected_symbol, start_date, end_date)
                        if hist_df is not None:
                            hist_df['RSI'] = calculate_rsi(hist_df)
                            fig_rsi = go.Figure()
                            for i in range(1, len(hist_df)):
                                color = '#FFFFFF' if 30 <= hist_df['RSI'].iloc[i] <= 70 else '#EF553B' if hist_df['RSI'].iloc[i] > 70 else '#00CC96'
                                fig_rsi.add_trace(go.Scatter(x=[hist_df['Date'].iloc[i-1], hist_df['Date'].iloc[i]],
                                                             y=[hist_df['RSI'].iloc[i-1], hist_df['RSI'].iloc[i]],
                                                             mode='lines', line=dict(color=color), showlegend=False))
                            fig_rsi.add_hline(y=70, line_dash="dash", line_color="#EF553B", annotation_text="Overbought")
                            fig_rsi.add_hline(y=30, line_dash="dash", line_color="#00CC96", annotation_text="Oversold")
                            fig_rsi.update_layout(title=f"RSI: {selected_rsi_stock}", yaxis_title="RSI", template=custom_template)
                            st.plotly_chart(fig_rsi, use_container_width=True)

                        # Graph 7: Bollinger Bands
                        st.subheader("Bollinger Bands")
                        selected_bb_stock = st.selectbox("Stock for Bollinger Bands:", stock_options, key="bb")
                        selected_symbol = selected_bb_stock.split(" (")[1][:-1]
                        hist_df = fetch_historical_data(selected_symbol, start_date, end_date)
                        if hist_df is not None:
                            sma_options = [5, 10, 20, 50, 100]
                            sma_period = st.selectbox("SMA Period:", sma_options, index=2, key="bb_sma")
                            hist_df = calculate_bollinger_bands(hist_df, period=sma_period)
                            fig_bb = go.Figure()
                            fig_bb.add_trace(go.Scatter(x=hist_df['Date'], y=hist_df['Close'], name='Price', line_color='#00CC96',
                                                        hovertemplate='Date: %{x}<br>Price: ₹%{y:.2f}<br>SMA: ₹%{customdata[0]:.2f}<br>Upper: ₹%{customdata[1]:.2f}<br>Lower: ₹%{customdata[2]:.2f}',
                                                        customdata=hist_df[['SMA', 'Upper Band', 'Lower Band']]))
                            fig_bb.add_trace(go.Scatter(x=hist_df['Date'], y=hist_df['Upper Band'], name='Upper', line=dict(dash='dash', color='#EF553B')))
                            fig_bb.add_trace(go.Scatter(x=hist_df['Date'], y=hist_df['Lower Band'], name='Lower', line=dict(dash='dash', color='#00CC96')))
                            fig_bb.add_trace(go.Scatter(x=hist_df['Date'], y=hist_df['SMA'], name=f'SMA-{sma_period}', line_color='#FFFF00'))
                            fig_bb.update_layout(title=f"Bollinger Bands: {selected_bb_stock}", yaxis_title="Price (₹)",
                                                 template=custom_template, legend=dict(x=0.01, y=0.99))
                            st.plotly_chart(fig_bb, use_container_width=True)

                        # Graph 8: P/E Ratio Trend
                        st.subheader("P/E Ratio Trend")
                        selected_pe_stock = st.selectbox("Stock for P/E Trend:", stock_options, key="pe")
                        selected_symbol = selected_pe_stock.split(" (")[1][:-1]
                        hist_pe_df = fetch_historical_pe(selected_symbol, start_date, end_date)
                        if hist_pe_df is not None:
                            fig_pe_trend = px.line(hist_pe_df, x='Date', y='P/E Ratio', title=f"P/E Trend: {selected_pe_stock}",
                                                   color_discrete_sequence=[stock_colors[selected_symbol]])
                            fig_pe_trend.add_hline(y=20, line_dash="dash", line_color="#FFFF00", annotation_text="Avg")
                            fig_pe_trend.update_layout(
                                template=custom_template,
                                height=400,
                                yaxis=dict(
                                    dtick=10,
                                    range=[0, max(50, hist_pe_df['P/E Ratio'].max() * 1.1)]
                                )
                            )
                            st.plotly_chart(fig_pe_trend, use_container_width=True)

                        # New Graph 9: Selectable Graph Type
                        st.subheader("Portfolio Insights")
                        graph9_options = ["Profit/Loss Breakdown (Waterfall)", "Correlation Matrix (Heatmap)"]
                        graph9_type = st.selectbox("Select Graph Type:", graph9_options, key="graph9")
                        if graph9_type == "Profit/Loss Breakdown (Waterfall)":
                            pl_df = calculate_pl_breakdown(df_valid)
                            fig_waterfall = go.Figure(go.Waterfall(
                                name="P/L", orientation="v",
                                x=pl_df['Stock Name'], y=pl_df['Unrealized P/L (₹)'],
                                text=[f"₹{val:.2f}" for val in pl_df['Unrealized P/L (₹)']],
                                textposition="auto",
                                connector={"line": {"color": "rgb(63, 63, 63)"}},
                            ))
                            fig_waterfall.update_layout(
                                title="Unrealized Profit/Loss Breakdown",
                                yaxis_title="P/L (₹)",
                                template=custom_template,
                                height=400
                            )
                            st.plotly_chart(fig_waterfall, use_container_width=True)
                        elif graph9_type == "Correlation Matrix (Heatmap)":
                            corr_matrix = calculate_correlation_matrix(df_valid)
                            if corr_matrix is not None:
                                fig_heatmap = px.imshow(corr_matrix, text_auto=".2f",
                                                        title="Stock Correlation Matrix",
                                                        color_continuous_scale="RdBu_r",
                                                        range_color=[-1, 1])
                                fig_heatmap.update_layout(template=custom_template, height=400)
                                st.plotly_chart(fig_heatmap, use_container_width=True)
                            else:
                                st.warning("Need at least 2 stocks with sufficient data for correlation matrix.")

                        # Clear All with Password Lock
                        st.subheader("Clear Data")
                        if st.session_state.clear_password:
                            if not st.session_state.is_unlocked:
                                with st.form("unlock_form"):
                                    password_input = st.text_input("Enter password to unlock 'Clear All'", type="password")
                                    if st.form_submit_button("Unlock"):
                                        if password_input == st.session_state.clear_password:
                                            st.session_state.is_unlocked = True
                                            st.success("Unlocked successfully!")
                                            st.rerun()
                                        else:
                                            st.error("Incorrect password.")
                            else:
                                if st.button("Clear All Data"):
                                    conn = sqlite3.connect("portfolio.db")
                                    c = conn.cursor()
                                    c.execute("DELETE FROM transactions")
                                    c.execute("DELETE FROM stocks")
                                    c.execute("DELETE FROM broker_transactions")
                                    conn.commit()
                                    conn.close()
                                    st.success("Data cleared!")
                                    st.session_state.is_unlocked = False
                                    st.rerun()
                                if st.button("Lock"):
                                    st.session_state.is_unlocked = False
                                    st.success("Locked again!")
                                    st.rerun()
                    else:
                        st.info("No current holdings with valid price data.")
                else:
                    st.info("No stocks currently held (quantity > 0).")
            else:
                st.info("Add stocks to view your dashboard.")

        elif st.session_state.view == "Stock Activity Ledger":
            if not df_transactions.empty:
                st.subheader("Stock Activity Ledger")
                ledger_df = df_transactions.copy()
                ledger_df['purchase_date'] = pd.to_datetime(ledger_df['purchase_date'])
                ledger_df['Transaction Value'] = ledger_df.apply(
                    lambda row: row['purchase_price'] * row['quantity'] if row['transaction_type'] == 'Buy' else -row['purchase_price'] * row['quantity'], axis=1)
                realized_pl_col = []
                for idx, row in ledger_df.iterrows():
                    if row['transaction_type'] == 'Sell':
                        symbol = row['symbol']
                        buys = ledger_df[(ledger_df['symbol'] == symbol) & (ledger_df['transaction_type'] == 'Buy') & (ledger_df['purchase_date'] <= row['purchase_date'])]
                        if not buys.empty:
                            avg_buy_price = (buys['purchase_price'] * buys['quantity']).sum() / buys['quantity'].sum()
                            realized_pl = (row['purchase_price'] - avg_buy_price) * row['quantity']
                            realized_pl_col.append(realized_pl)
                        else:
                            realized_pl_col.append(0)
                    else:
                        realized_pl_col.append(0)
                ledger_df['Realized P/L (₹)'] = realized_pl_col
                ledger_df = ledger_df[['symbol', 'stock_name', 'transaction_type', 'exchange', 'purchase_date', 'purchase_price', 'quantity', 'Transaction Value', 'Realized P/L (₹)']]
                ledger_df.columns = ['Symbol', 'Stock Name', 'Type', 'Exchange', 'Date', 'Price (₹)', 'Qty', 'Transaction Value (₹)', 'Realized P/L (₹)']
                ledger_df = ledger_df.sort_values('Date', ascending=False)
                styled_ledger = ledger_df.style.format({
                    'Price (₹)': '₹{:.2f}',
                    'Transaction Value (₹)': '₹{:.2f}',
                    'Realized P/L (₹)': '₹{:.2f}'
                }).map(color_transaction, subset=['Type']).map(color_pl, subset=['Realized P/L (₹)'])
                st.dataframe(styled_ledger, use_container_width=True, height=400)
                total_buys = ledger_df[ledger_df['Type'] == 'Buy']['Transaction Value (₹)'].sum()
                total_sells = -ledger_df[ledger_df['Type'] == 'Sell']['Transaction Value (₹)'].sum()
                net_investment = total_buys - total_sells
                st.write(f"**Summary:** Total Buys: ₹{total_buys:.2f} | Total Sells: ₹{total_sells:.2f} | Net Investment: ₹{net_investment:.2f}")
            else:
                st.info("No transactions yet.")

        elif st.session_state.view == "Fund Monitor":
            if not df_broker.empty:
                st.subheader("Fund Monitor")
                broker_df = df_broker.copy()
                broker_df['date'] = pd.to_datetime(broker_df['date'])
                broker_df['Deposit'] = broker_df.apply(lambda row: row['amount'] if row['transaction_type'] == 'Deposit' else 0, axis=1)
                broker_df['Withdraw'] = broker_df.apply(lambda row: row['amount'] if row['transaction_type'] == 'Withdraw' else 0, axis=1)
                broker_df = broker_df[['transaction_type', 'date', 'Deposit', 'Withdraw']]
                broker_df.columns = ['Type', 'Date', 'Deposit (₹)', 'Withdraw (₹)']
                broker_df = broker_df.sort_values('Date', ascending=False)
                styled_broker = broker_df.style.format({'Deposit (₹)': '₹{:.2f}', 'Withdraw (₹)': '₹{:.2f}'}).map(color_fund, subset=['Type'])
                st.dataframe(styled_broker, use_container_width=True, height=400)
                total_deposits = broker_df['Deposit (₹)'].sum()
                total_withdrawals = broker_df['Withdraw (₹)'].sum()
                net_balance = total_deposits - total_withdrawals
                st.write(f"**Summary:** Total Deposits: ₹{total_deposits:.2f} | Total Withdrawals: ₹{total_withdrawals:.2f} | Net Balance: ₹{net_balance:.2f}")
            else:
                st.info("No broker transactions yet.")

        elif st.session_state.view == "Unrealized P/L":
            if not df_stocks.empty:
                df_current_stocks = df_stocks[df_stocks['quantity'] > 0].copy()
                if not df_current_stocks.empty:
                    with st.spinner("Updating Prices..."):
                        prices = [fetch_data(symbol) for symbol in df_current_stocks['symbol']]
                        df_current_stocks['current_price'] = prices

                    df_valid = df_current_stocks.dropna(subset=['current_price']).copy()
                    if not df_valid.empty:
                        df_valid['value'] = df_valid['current_price'] * df_valid['quantity']
                        df_valid['unrealized_pl'] = (df_valid['current_price'] - df_valid['purchase_price']) * df_valid['quantity']
                        df_valid['return_percentage'] = df_valid.apply(
                            lambda row: (row['unrealized_pl'] / row['total_cost'] * 100) if row['total_cost'] > 0 else 0, axis=1)
                        df_valid['sector'] = [yf.Ticker(symbol).info.get('sector', 'Unknown') for symbol in df_valid['symbol']]

                        st.subheader("Unrealized Profit/Loss")
                        holdings_df = df_valid[['symbol', 'stock_name', 'quantity', 'current_price', 'value', 'purchase_price', 'unrealized_pl', 'return_percentage', 'sector', 'exchange']]
                        holdings_df.columns = ['Symbol', 'Stock Name', 'Qty', 'Current Price (₹)', 'Value (₹)', 'Avg Buy Price (₹)', 'Return (₹)', 'Return %', 'Sector', 'Exchange']
                        styled_df = holdings_df.style.format({
                            'Current Price (₹)': '₹{:.2f}',
                            'Value (₹)': '₹{:.2f}',
                            'Avg Buy Price (₹)': '₹{:.2f}',
                            'Return (₹)': '₹{:.2f}',
                            'Return %': '{:.2f}%'
                        }).map(color_pl, subset=['Return (₹)', 'Return %'])
                        st.dataframe(styled_df, use_container_width=True)
                        if st.download_button("Export Holdings", holdings_df.to_csv(index=False), "holdings.csv"):
                            st.success("Holdings exported!")
                    else:
                        st.info("No current holdings with valid price data.")
                else:
                    st.info("No stocks currently held (quantity > 0).")
            else:
                st.info("Add stocks to view unrealized P/L.")

        elif st.session_state.view == "Realized P/L":
            if not df_transactions.empty:
                st.subheader("Realized Profit/Loss")
                realized_pl_df = calculate_realized_pl_table(df_transactions)
                if not realized_pl_df.empty:
                    realized_pl_df = realized_pl_df.sort_values('Date', ascending=False)
                    styled_realized_pl = realized_pl_df.style.format({
                        'Price (₹)': '₹{:.2f}',
                        'Realized P/L (₹)': '₹{:.2f}'
                    }).map(color_transaction, subset=['Type']).map(color_pl, subset=['Realized P/L (₹)'])
                    st.dataframe(styled_realized_pl, use_container_width=True, height=400)
                    total_realized_pl = realized_pl_df['Realized P/L (₹)'].sum()
                    st.write(f"**Total Realized P/L:** ₹{total_realized_pl:.2f}")
                else:
                    st.info("No realized profit/loss from sales yet.")
            else:
                st.info("No transactions yet.")

if __name__ == "__main__":
    main()
