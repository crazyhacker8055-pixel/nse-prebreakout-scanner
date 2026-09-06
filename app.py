import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="NSE Pre-Breakout Scanner",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 NSE Pre-Breakout Stock Scanner")

st.caption(
    "Technical screening dashboard for Indian NSE stocks. "
    "For research and educational purposes."
)

# ============================================================
# SETTINGS
# ============================================================

DEFAULT_STOCKS = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "AXISBANK.NS",
    "LT.NS",
    "BHARTIARTL.NS",
    "ITC.NS",
    "MARUTI.NS",
    "TATAMOTORS.NS",
    "TATASTEEL.NS",
    "SUNPHARMA.NS",
    "HINDALCO.NS",
    "JSWSTEEL.NS",
    "ADANIENT.NS",
    "ADANIPORTS.NS",
    "BAJFINANCE.NS",
    "BAJAJFINSV.NS",
]

# ============================================================
# TECHNICAL FUNCTIONS
# ============================================================

def calculate_indicators(df):

    df = df.copy()

    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA150"] = df["Close"].rolling(150).mean()
    df["EMA220"] = df["Close"].ewm(
        span=220,
        adjust=False
    ).mean()

    df["52W_LOW"] = df["Low"].rolling(252).min()
    df["52W_HIGH"] = df["High"].rolling(252).max()

    df["AVG_VOL20"] = df["Volume"].rolling(20).mean()

    # ATR
    high_low = df["High"] - df["Low"]
    high_close = abs(df["High"] - df["Close"].shift())
    low_close = abs(df["Low"] - df["Close"].shift())

    tr = pd.concat(
        [high_low, high_close, low_close],
        axis=1
    ).max(axis=1)

    df["ATR14"] = tr.rolling(14).mean()

    # RSI
    delta = df["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["RSI14"] = 100 - (
        100 / (1 + rs)
    )

    # Range measurements
    df["RANGE5"] = (
        df["High"].rolling(5).max()
        -
        df["Low"].rolling(5).min()
    ) / df["Close"]

    df["RANGE10"] = (
        df["High"].rolling(10).max()
        -
        df["Low"].rolling(10).min()
    ) / df["Close"]

    df["RANGE20"] = (
        df["High"].rolling(20).max()
        -
        df["Low"].rolling(20).min()
    ) / df["Close"]

    return df


# ============================================================
# 90-DAY DIP BELOW 220 EMA
# ============================================================

def dipped_below_ema(df):

    recent = df.tail(90)

    if len(recent) < 90:
        return False

    return bool(
        (recent["Low"] < recent["EMA220"]).any()
    )


# ============================================================
# VCP / CONTRACTION CHECK
# ============================================================

def vcp_score(df):

    if len(df) < 30:
        return 0

    r5 = df["RANGE5"].iloc[-1]
    r10 = df["RANGE10"].iloc[-1]
    r20 = df["RANGE20"].iloc[-1]

    score = 0

    if r5 < r10:
        score += 1

    if r10 < r20:
        score += 1

    if r5 < 0.08:
        score += 1

    if r10 < 0.15:
        score += 1

    return score


# ============================================================
# STOCK ANALYSIS
# ============================================================

def analyze_stock(ticker):

    try:

        df = yf.download(
            ticker,
            period="2y",
            interval="1d",
            auto_adjust=False,
            progress=False
        )

        if df.empty:
            return None

        # Handle yfinance multi-index columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.dropna()

        if len(df) < 260:
            return None

        df = calculate_indicators(df)

        last = df.iloc[-1]

        price = float(last["Close"])

        # ----------------------------------------------------
        # CONDITIONS
        # ----------------------------------------------------

        condition_1 = (
            last["SMA150"] >
            last["EMA220"]
        )

        condition_2 = (
            price >
            last["SMA50"]
        )

        condition_3 = (
            last["SMA50"] >
            last["SMA150"]
        )

        condition_4 = (
            price >
            1.25 * last["52W_LOW"]
        )

        condition_5 = dipped_below_ema(df)

        breakout_level = float(
            df["High"].tail(20).max()
        )

        # Distance from 20-day breakout
        breakout_distance = (
            breakout_level - price
        ) / price

        condition_6 = (
            breakout_distance <= 0.05
        )

        # Tight range
        condition_7 = (
            last["RANGE5"] < 0.08
            and
            last["RANGE10"] < 0.15
        )

        # Volume
        volume_ratio = (
            last["Volume"] /
            last["AVG_VOL20"]
        )

        # VCP
        vcp = vcp_score(df)

        # ----------------------------------------------------
        # SCORE
        # ----------------------------------------------------

        score = 0

        if condition_1:
            score += 15

        if condition_2:
            score += 10

        if condition_3:
            score += 10

        if condition_4:
            score += 10

        if condition_5:
            score += 10

        if condition_6:
            score += 15

        if condition_7:
            score += 10

        # VCP
        score += min(vcp * 2.5, 10)

        # RSI
        rsi = float(last["RSI14"])

        if 50 <= rsi <= 70:
            score += 5

        # Volume
        if volume_ratio > 1.2:
            score += 5

        # ----------------------------------------------------
        # ENTRY / SL / TARGET
        # ----------------------------------------------------

        entry = breakout_level

        atr = float(last["ATR14"])

        technical_stop = (
            entry - 1.5 * atr
        )

        # 15% maximum risk rule
        max_loss_stop = entry * 0.85

        stop_loss = max(
            technical_stop,
            max_loss_stop
        )

        risk = entry - stop_loss

        target1 = entry + (risk * 1.5)
        target2 = entry + (risk * 2.0)
        target3 = entry + (risk * 3.0)

        # ----------------------------------------------------
        # PRE-OPEN / CURRENT PRICE
        # ----------------------------------------------------

        previous_close = float(
            df["Close"].iloc[-1]
        )

        result = {

            "Ticker": ticker.replace(".NS", ""),

            "Price": round(price, 2),

            "Score": round(score, 1),

            "Breakout": round(
                breakout_level,
                2
            ),

            "Breakout Distance %": round(
                breakout_distance * 100,
                2
            ),

            "Entry": round(entry, 2),

            "Stop Loss": round(
                stop_loss,
                2
            ),

            "Target 1": round(
                target1,
                2
            ),

            "Target 2": round(
                target2,
                2
            ),

            "Target 3": round(
                target3,
                2
            ),

            "RSI": round(
                rsi,
                1
            ),

            "Volume Ratio": round(
                volume_ratio,
                2
            ),

            "VCP Score": vcp,

            "Range 5D %": round(
                last["RANGE5"] * 100,
                2
            ),

            "Range 10D %": round(
                last["RANGE10"] * 100,
                2
            ),

            "Range 20D %": round(
                last["RANGE20"] * 100,
                2
            ),

            "Trend OK": condition_1
            and condition_2
            and condition_3,

            "EMA Dip": condition_5,

            "Near Breakout": condition_6,

            "Tight Range": condition_7,
        }

        return result

    except Exception as e:

        return None


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Scanner Settings")

run_scanner = st.sidebar.button(
    "🔎 RUN SCANNER",
    type="primary"
)

show_all = st.sidebar.checkbox(
    "Show all analysed stocks",
    value=False
)

minimum_score = st.sidebar.slider(
    "Minimum Score",
    0,
    100,
    60
)

# ============================================================
# MAIN
# ============================================================

if run_scanner:

    results = []

    progress = st.progress(0)

    status = st.empty()

    total = len(DEFAULT_STOCKS)

    for i, ticker in enumerate(DEFAULT_STOCKS):

        status.write(
            f"Scanning {ticker}..."
        )

        result = analyze_stock(ticker)

        if result is not None:
            results.append(result)

        progress.progress(
            (i + 1) / total
        )

    status.success(
        "Scan completed."
    )

    if not results:

        st.error(
            "No data returned. "
            "Try again in a few seconds."
        )

    else:

        result_df = pd.DataFrame(results)

        result_df = result_df.sort_values(
            "Score",
            ascending=False
        )

        # ----------------------------------------------------
        # TOP PICKS
        # ----------------------------------------------------

        st.subheader(
            "🔥 Top Pre-Breakout Candidates"
        )

        top = result_df[
            result_df["Score"] >= minimum_score
        ].head(10)

        if len(top) == 0:

            st.warning(
                "No stocks currently meet the score threshold."
            )

        else:

            st.dataframe(
                top,
                use_container_width=True,
                hide_index=True
            )

        # ----------------------------------------------------
        # BEST STOCK
        # ----------------------------------------------------

        if len(top) > 0:

            best = top.iloc[0]

            st.subheader(
                f"🏆 #1 Setup — {best['Ticker']}"
            )

            c1, c2, c3, c4, c5 = st.columns(5)

            c1.metric(
                "Score",
                f"{best['Score']}/100"
            )

            c2.metric(
                "Breakout",
                f"₹{best['Breakout']}"
            )

            c3.metric(
                "Stop Loss",
                f"₹{best['Stop Loss']}"
            )

            c4.metric(
                "Target 1",
                f"₹{best['Target 1']}"
            )

            c5.metric(
                "Target 2",
                f"₹{best['Target 2']}"
            )

            st.info(
                f"""
                **Setup:** Pre-breakout candidate

                **Entry/Breakout:** ₹{best['Entry']}

                **Stop Loss:** ₹{best['Stop Loss']}

                **Target 1:** ₹{best['Target 1']}

                **Target 2:** ₹{best['Target 2']}

                **Target 3:** ₹{best['Target 3']}

                **RSI:** {best['RSI']}

                **Volume Ratio:** {best['Volume Ratio']}x

                **VCP Score:** {best['VCP Score']}/4
                """
            )

        # ----------------------------------------------------
        # DOWNLOAD
        # ----------------------------------------------------

        csv = result_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download Results CSV",
            csv,
            "prebreakout_scan.csv",
            "text/csv"
        )

else:

    st.info(
        "Press 🔎 RUN SCANNER to start."
    )

    st.markdown(
        """
        ### Scanner Rules

        ✅ 150 SMA > 220 EMA

        ✅ Close > 50 SMA

        ✅ 50 SMA > 150 SMA

        ✅ Price > 1.25 × 52-week low

        ✅ Dip below 220 EMA within 90 sessions

        ✅ Within 5% of 20-day breakout

        ✅ Tight 5/10-day range

        ✅ VCP-style contraction

        ✅ RSI filter

        ✅ Volume analysis

        ### Risk Management

        Stop-loss uses ATR/technical structure
        with a maximum 15% loss constraint.

        Targets are calculated using risk multiples.
        """
    )

st.caption(
    f"Last page load: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)
