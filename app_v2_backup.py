import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO
from datetime import datetime

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="NSE Pre-Breakout Scanner V2",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 NSE Pre-Breakout Scanner V2")

st.caption(
    "Automatic NSE/NIFTY 500 technical scanner | "
    "Pre-breakout + VCP + trend + volume analysis"
)

# =========================================================
# SETTINGS
# =========================================================

st.sidebar.header("⚙️ Scanner Settings")

universe_size = st.sidebar.selectbox(
    "Stock Universe",
    ["NIFTY 500", "NIFTY 250", "NIFTY 100"],
    index=0
)

breakout_distance = st.sidebar.slider(
    "Maximum distance from 20D breakout (%)",
    1.0,
    10.0,
    5.0,
    0.5
)

min_score = st.sidebar.slider(
    "Minimum Setup Score",
    0,
    100,
    50,
    5
)

batch_size = st.sidebar.selectbox(
    "Download batch size",
    [50, 75, 100],
    index=1
)

st.sidebar.markdown("---")

st.sidebar.info(
    "This scanner uses daily historical market data. "
    "It is designed to identify stocks approaching technical breakouts, "
    "not to guarantee future returns."
)

# =========================================================
# LOAD NSE STOCK UNIVERSE
# =========================================================

@st.cache_data(ttl=86400)
def get_nifty500():

    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Linux; Android 10; Mobile) "
            "AppleWebKit/537.36 Chrome/120 Safari/537.36"
        ),
        "Accept": "text/csv,text/plain,*/*"
    }

    try:

        response = requests.get(
            url,
            headers=headers,
            timeout=20
        )

        response.raise_for_status()

        df = pd.read_csv(StringIO(response.text))

        if "Symbol" not in df.columns:
            return []

        symbols = (
            df["Symbol"]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )

        return symbols

    except Exception:

        return []


@st.cache_data(ttl=86400)
def get_nifty250():

    # Fallback subset of highly liquid NSE stocks
    symbols = [
        "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY",
        "SBIN", "BHARTIARTL", "ITC", "LT", "AXISBANK",
        "KOTAKBANK", "HINDUNILVR", "BAJFINANCE", "MARUTI",
        "SUNPHARMA", "TITAN", "NTPC", "POWERGRID",
        "M&M", "ADANIENT", "ADANIPORTS", "TATASTEEL",
        "JSWSTEEL", "HINDALCO", "COALINDIA", "ONGC",
        "WIPRO", "TECHM", "HCLTECH", "TATAMOTORS",
        "TATA", "BEL", "HAL", "TRENT", "INDUSINDBK",
        "BANKBARODA", "CANBK", "PNB", "DLF", "SIEMENS",
        "ABB", "PIDILITIND", "CIPLA", "DRREDDY",
        "DIVISLAB", "EICHERMOT", "HEROMOTOCO",
        "BAJAJ-AUTO", "TVSMOTOR", "APOLLOHOSP"
    ]

    return symbols


@st.cache_data(ttl=86400)
def get_universe(selection):

    if selection == "NIFTY 500":

        symbols = get_nifty500()

        if len(symbols) >= 300:
            return symbols

        return get_nifty250()

    elif selection == "NIFTY 250":

        return get_nifty250()

    else:

        return [
            "RELIANCE",
            "TCS",
            "HDFCBANK",
            "ICICIBANK",
            "INFY",
            "SBIN",
            "BHARTIARTL",
            "ITC",
            "LT",
            "AXISBANK",
            "KOTAKBANK",
            "HINDUNILVR",
            "BAJFINANCE",
            "MARUTI",
            "SUNPHARMA",
            "TITAN",
            "NTPC",
            "POWERGRID",
            "M&M",
            "ADANIENT",
            "ADANIPORTS",
            "TATASTEEL",
            "JSWSTEEL",
            "HINDALCO",
            "COALINDIA",
            "ONGC",
            "WIPRO",
            "TECHM",
            "HCLTECH",
            "TATAMOTORS"
        ]


# =========================================================
# TECHNICAL INDICATORS
# =========================================================

def calculate_indicators(df):

    df = df.copy()

    if len(df) < 250:
        return None

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # Moving averages
    df["SMA50"] = close.rolling(50).mean()
    df["SMA150"] = close.rolling(150).mean()
    df["EMA220"] = close.ewm(span=220, adjust=False).mean()

    # 52 week levels
    df["52W_HIGH"] = high.rolling(252).max()
    df["52W_LOW"] = low.rolling(252).min()

    # True Range / ATR
    previous_close = close.shift(1)

    tr1 = high - low
    tr2 = abs(high - previous_close)
    tr3 = abs(low - previous_close)

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    df["ATR14"] = true_range.rolling(14).mean()

    # RSI
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["RSI14"] = 100 - (100 / (1 + rs))

    # Volume
    df["AVG_VOL20"] = volume.rolling(20).mean()
    df["AVG_VOL50"] = volume.rolling(50).mean()

    df["VOL_RATIO"] = (
        volume / df["AVG_VOL20"].replace(0, np.nan)
    )

    # Ranges
    df["RANGE5"] = (
        (high.rolling(5).max() -
         low.rolling(5).min())
        / close
        * 100
    )

    df["RANGE10"] = (
        (high.rolling(10).max() -
         low.rolling(10).min())
        / close
        * 100
    )

    df["RANGE20"] = (
        (high.rolling(20).max() -
         low.rolling(20).min())
        / close
        * 100
    )

    # 20 day resistance
    df["BREAKOUT20"] = high.shift(1).rolling(20).max()

    # Previous 90 session dip below EMA220
    df["BELOW_EMA220"] = low < df["EMA220"]

    df["DIP90"] = (
        df["BELOW_EMA220"]
        .rolling(90)
        .max()
    )

    return df


# =========================================================
# STOCK ANALYSIS
# =========================================================

def analyze_stock(symbol, df, max_breakout_distance):

    try:

        df = calculate_indicators(df)

        if df is None:
            return None

        row = df.iloc[-1]

        price = float(row["Close"])

        sma50 = float(row["SMA50"])
        sma150 = float(row["SMA150"])
        ema220 = float(row["EMA220"])

        high52 = float(row["52W_HIGH"])
        low52 = float(row["52W_LOW"])

        atr = float(row["ATR14"])
        rsi = float(row["RSI14"])

        vol_ratio = float(row["VOL_RATIO"])

        range5 = float(row["RANGE5"])
        range10 = float(row["RANGE10"])
        range20 = float(row["RANGE20"])

        breakout = float(row["BREAKOUT20"])

        dip90 = bool(row["DIP90"])

        if any(
            pd.isna(x)
            for x in [
                price,
                sma50,
                sma150,
                ema220,
                high52,
                low52,
                atr,
                rsi,
                vol_ratio,
                range5,
                range10,
                range20,
                breakout
            ]
        ):
            return None

        # =================================================
        # CORE TREND CONDITIONS
        # =================================================

        trend1 = sma150 > ema220
        trend2 = price > sma50
        trend3 = sma50 > sma150

        distance_from_low = (
            price / low52
        )

        above_25pct_low = (
            price > 1.25 * low52
        )

        # =================================================
        # BREAKOUT PROXIMITY
        # =================================================

        breakout_distance_pct = (
            (breakout - price)
            / price
            * 100
        )

        near_breakout = (
            breakout_distance_pct >= 0
            and
            breakout_distance_pct <= max_breakout_distance
        )

        # =================================================
        # RANGE CONTRACTION
        # =================================================

        range_contracting = (
            range5 <= range10
            and
            range10 <= range20
        )

        tight_range = (
            range20 <= 20
        )

        # =================================================
        # VOLUME CONTRACTION
        # =================================================

        volume_contraction = (
            vol_ratio < 1.0
        )

        # =================================================
        # VCP SCORE
        # =================================================

        vcp_score = 0

        if range5 < range10:
            vcp_score += 1

        if range10 < range20:
            vcp_score += 1

        if volume_contraction:
            vcp_score += 1

        if near_breakout:
            vcp_score += 1

        # =================================================
        # SCORE
        # =================================================

        score = 0

        if trend1:
            score += 15

        if trend2:
            score += 15

        if trend3:
            score += 15

        if above_25pct_low:
            score += 10

        if dip90:
            score += 10

        if near_breakout:
            score += 15

        if range_contracting:
            score += 10

        if tight_range:
            score += 5

        if volume_contraction:
            score += 5

        # RSI sweet spot
        if 45 <= rsi <= 70:
            score += 5

        score = min(score, 100)

        # =================================================
        # QUALITY
        # =================================================

        if score >= 85:
            quality = "🔥 VERY HIGH"

        elif score >= 75:
            quality = "🟢 HIGH"

        elif score >= 65:
            quality = "🟡 MEDIUM"

        elif score >= 50:
            quality = "⚪ WATCH"

        else:
            quality = "🔴 LOW"

        # =================================================
        # ENTRY / SL / TARGETS
        # =================================================

        entry = breakout

        stop_loss = max(
            ema220,
            entry - 2 * atr
        )

        risk = entry - stop_loss

        if risk <= 0:
            return None

        target1 = entry + risk * 1.5
        target2 = entry + risk * 2.0
        target3 = entry + risk * 3.0

        return {
            "Symbol": symbol,
            "Score": round(score, 1),
            "Quality": quality,

            "Close": round(price, 2),

            "Breakout": round(breakout, 2),

            "Breakout Distance %":
                round(breakout_distance_pct, 2),

            "Entry": round(entry, 2),

            "Stop Loss": round(stop_loss, 2),

            "Target 1": round(target1, 2),

            "Target 2": round(target2, 2),

            "Target 3": round(target3, 2),

            "RSI": round(rsi, 1),

            "Volume Ratio":
                round(vol_ratio, 2),

            "Range 5D %":
                round(range5, 2),

            "Range 10D %":
                round(range10, 2),

            "Range 20D %":
                round(range20, 2),

            "VCP":
                f"{vcp_score}/4",

            "Dip Below EMA220":
                "YES" if dip90 else "NO",

            "Trend":
                "YES"
                if trend1 and trend2 and trend3
                else "NO"
        }

    except Exception:

        return None


# =========================================================
# DOWNLOAD DATA
# =========================================================

@st.cache_data(ttl=1800, show_spinner=False)
def download_market_data(symbols, batch_size):

    all_results = {}

    tickers = [
        f"{symbol}.NS"
        for symbol in symbols
    ]

    for start in range(
        0,
        len(tickers),
        batch_size
    ):

        batch = tickers[
            start:start + batch_size
        ]

        try:

            data = yf.download(
                batch,
                period="2y",
                interval="1d",
                auto_adjust=False,
                group_by="ticker",
                threads=True,
                progress=False
            )

            if data is None or data.empty:
                continue

            if isinstance(
                data.columns,
                pd.MultiIndex
            ):

                for ticker in batch:

                    try:

                        if ticker not in data.columns.levels[0]:
                            continue

                        stock_df = data[ticker].copy()

                        stock_df = stock_df.dropna(
                            how="all"
                        )

                        if len(stock_df) > 0:
                            symbol = ticker.replace(
                                ".NS",
                                ""
                            )

                            all_results[
                                symbol
                            ] = stock_df

                    except Exception:
                        continue

        except Exception:
            continue

    return all_results


# =========================================================
# SCANNER
# =========================================================

st.markdown("---")

col1, col2, col3 = st.columns(3)

universe = get_universe(
    universe_size
)

with col1:
    st.metric(
        "Universe",
        len(universe)
    )

with col2:
    st.metric(
        "Minimum Score",
        min_score
    )

with col3:
    st.metric(
        "Breakout Zone",
        f"{breakout_distance}%"
    )

run_scanner = st.button(
    "🔎 RUN V2 SCANNER",
    type="primary",
    use_container_width=True
)

if run_scanner:

    progress = st.progress(0)

    status = st.empty()

    status.info(
        f"Loading {len(universe)} NSE stocks..."
    )

    market_data = download_market_data(
        tuple(universe),
        batch_size
    )

    status.info(
        f"Analysing {len(market_data)} stocks..."
    )

    results = []

    total = len(market_data)

    for i, (symbol, df) in enumerate(
        market_data.items()
    ):

        result = analyze_stock(
            symbol,
            df,
            breakout_distance
        )

        if result is not None:

            if result["Score"] >= min_score:
                results.append(result)

        if total > 0:
            progress.progress(
                int((i + 1) / total * 100)
            )

    progress.empty()

    status.success(
        f"Scan completed — {len(results)} qualifying stocks found."
    )

    if results:

        results_df = pd.DataFrame(
            results
        )

        results_df = results_df.sort_values(
            "Score",
            ascending=False
        ).reset_index(drop=True)

        # =================================================
        # TOP SETUPS
        # =================================================

        st.subheader(
            "🔥 Highest Conviction Pre-Breakout Setups"
        )

        top = results_df.head(10)

        st.dataframe(
            top,
            use_container_width=True,
            hide_index=True
        )

        # =================================================
        # BEST STOCK
        # =================================================

        best = results_df.iloc[0]

        st.markdown("---")

        st.subheader(
            f"🎯 #1 Setup — {best['Symbol']}"
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Setup Score",
                f"{best['Score']}/100"
            )

        with c2:
            st.metric(
                "Breakout",
                f"₹{best['Breakout']:,.2f}"
            )

        with c3:
            st.metric(
                "Stop Loss",
                f"₹{best['Stop Loss']:,.2f}"
            )

        with c4:
            st.metric(
                "VCP",
                best["VCP"]
            )

        # =================================================
        # ENTRY / TARGETS
        # =================================================

        st.subheader(
            "📌 Trade Levels"
        )

        level1, level2, level3, level4, level5 = st.columns(5)

        with level1:
            st.metric(
                "Entry",
                f"₹{best['Entry']:,.2f}"
            )

        with level2:
            st.metric(
                "Stop Loss",
                f"₹{best['Stop Loss']:,.2f}"
            )

        with level3:
            st.metric(
                "Target 1",
                f"₹{best['Target 1']:,.2f}"
            )

        with level4:
            st.metric(
                "Target 2",
                f"₹{best['Target 2']:,.2f}"
            )

        with level5:
            st.metric(
                "Target 3",
                f"₹{best['Target 3']:,.2f}"
            )

        # =================================================
        # MARKET DATA
        # =================================================

        st.subheader(
            "📊 Setup Diagnostics"
        )

        d1, d2, d3, d4, d5, d6 = st.columns(6)

        with d1:
            st.metric(
                "RSI",
                best["RSI"]
            )

        with d2:
            st.metric(
                "Volume",
                f"{best['Volume Ratio']}x"
            )

        with d3:
            st.metric(
                "5D Range",
                f"{best['Range 5D %']}%"
            )

        with d4:
            st.metric(
                "10D Range",
                f"{best['Range 10D %']}%"
            )

        with d5:
            st.metric(
                "20D Range",
                f"{best['Range 20D %']}%"
            )

        with d6:
            st.metric(
                "Dip 220 EMA",
                best["Dip Below EMA220"]
            )

        # =================================================
        # FULL RESULTS
        # =================================================

        st.markdown("---")

        st.subheader(
            "📋 All Qualifying Stocks"
        )

        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True
        )

        # =================================================
        # DOWNLOAD
        # =================================================

        csv = results_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download Scanner Results",
            csv,
            "nse_prebreakout_v2.csv",
            "text/csv",
            use_container_width=True
        )

    else:

        st.warning(
            "No stocks currently meet the selected score."
        )

else:

    st.info(
        "Press 🔎 RUN V2 SCANNER to scan the NSE universe."
    )
