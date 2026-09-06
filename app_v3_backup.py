import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO

# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="NSE Pre-Breakout Scanner V3",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 NSE Pre-Breakout Scanner V3")
st.caption(
    "Advanced VCP + contraction + volume + relative strength "
    "pre-breakout engine"
)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("⚙️ Scanner Settings")

universe_choice = st.sidebar.selectbox(
    "Stock Universe",
    ["NIFTY 500", "NIFTY 250", "NIFTY 100"],
    index=0
)

max_breakout_distance = st.sidebar.slider(
    "Maximum distance from breakout (%)",
    1.0,
    10.0,
    5.0,
    0.5
)

minimum_score = st.sidebar.slider(
    "Minimum setup score",
    40,
    90,
    65,
    5
)

batch_size = st.sidebar.selectbox(
    "Download batch size",
    [50, 75, 100],
    index=1
)

strict_mode = st.sidebar.checkbox(
    "Strict Pre-Breakout Mode",
    value=True
)

st.sidebar.markdown("---")

st.sidebar.info(
    "V3 is designed to find stocks that are tightening "
    "near resistance before a potential breakout."
)

# =========================================================
# NSE UNIVERSE
# =========================================================

@st.cache_data(ttl=86400)
def get_nifty500():

    url = (
        "https://archives.nseindia.com/"
        "content/indices/ind_nifty500list.csv"
    )

    headers = {
        "User-Agent":
            "Mozilla/5.0 (Linux; Android 10; Mobile) "
            "AppleWebKit/537.36 Chrome/120 Safari/537.36"
    }

    try:

        response = requests.get(
            url,
            headers=headers,
            timeout=20
        )

        response.raise_for_status()

        df = pd.read_csv(
            StringIO(response.text)
        )

        if "Symbol" not in df.columns:
            return []

        return (
            df["Symbol"]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )

    except Exception:
        return []


@st.cache_data(ttl=86400)
def get_fallback_universe():

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
        "TATAMOTORS",
        "BEL",
        "HAL",
        "TRENT",
        "SIEMENS",
        "ABB",
        "CIPLA",
        "DRREDDY",
        "DIVISLAB",
        "EICHERMOT",
        "HEROMOTOCO",
        "BAJAJ-AUTO",
        "TVSMOTOR",
        "APOLLOHOSP",
        "CANBK",
        "PNB",
        "BANKBARODA",
        "DLF",
        "PIDILITIND",
        "GRASIM",
        "CUB",
        "UNIONBANK"
    ]


@st.cache_data(ttl=86400)
def get_universe(choice):

    if choice == "NIFTY 500":

        symbols = get_nifty500()

        if len(symbols) >= 300:
            return symbols

        return get_fallback_universe()

    if choice == "NIFTY 250":
        return get_fallback_universe()

    return get_fallback_universe()[:30]


# =========================================================
# INDICATORS
# =========================================================

def add_indicators(df):

    df = df.copy()

    if len(df) < 260:
        return None

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # -----------------------------------------------------
    # TREND
    # -----------------------------------------------------

    df["SMA50"] = close.rolling(50).mean()
    df["SMA150"] = close.rolling(150).mean()
    df["EMA220"] = close.ewm(
        span=220,
        adjust=False
    ).mean()

    # -----------------------------------------------------
    # 52 WEEK
    # -----------------------------------------------------

    df["52W_HIGH"] = high.rolling(252).max()
    df["52W_LOW"] = low.rolling(252).min()

    # -----------------------------------------------------
    # ATR
    # -----------------------------------------------------

    previous_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            abs(high - previous_close),
            abs(low - previous_close)
        ],
        axis=1
    ).max(axis=1)

    df["ATR14"] = tr.rolling(14).mean()

    df["ATR_PCT"] = (
        df["ATR14"] / close * 100
    )

    # -----------------------------------------------------
    # RSI
    # -----------------------------------------------------

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

    df["RSI"] = (
        100 -
        (100 / (1 + rs))
    )

    # -----------------------------------------------------
    # VOLUME
    # -----------------------------------------------------

    df["VOL20"] = volume.rolling(20).mean()
    df["VOL50"] = volume.rolling(50).mean()

    df["VOL_RATIO"] = (
        volume /
        df["VOL20"].replace(0, np.nan)
    )

    # Volume dry-up
    df["VOL5_AVG"] = volume.rolling(5).mean()
    df["VOL20_AVG"] = volume.rolling(20).mean()

    df["VOLUME_DRYUP"] = (
        df["VOL5_AVG"] /
        df["VOL20_AVG"].replace(0, np.nan)
    )

    # -----------------------------------------------------
    # RANGE
    # -----------------------------------------------------

    df["RANGE5"] = (
        (
            high.rolling(5).max() -
            low.rolling(5).min()
        )
        / close
        * 100
    )

    df["RANGE10"] = (
        (
            high.rolling(10).max() -
            low.rolling(10).min()
        )
        / close
        * 100
    )

    df["RANGE20"] = (
        (
            high.rolling(20).max() -
            low.rolling(20).min()
        )
        / close
        * 100
    )

    # -----------------------------------------------------
    # ATR COMPRESSION
    # -----------------------------------------------------

    df["ATR20_AVG"] = (
        df["ATR_PCT"].rolling(20).mean()
    )

    # -----------------------------------------------------
    # RESISTANCE
    # -----------------------------------------------------

    df["BREAKOUT20"] = (
        high.shift(1)
        .rolling(20)
        .max()
    )

    df["BREAKOUT50"] = (
        high.shift(1)
        .rolling(50)
        .max()
    )

    # -----------------------------------------------------
    # EMA220 UNDERCUT
    # -----------------------------------------------------

    df["BELOW220"] = (
        low < df["EMA220"]
    )

    df["DIP90"] = (
        df["BELOW220"]
        .rolling(90)
        .max()
    )

    # -----------------------------------------------------
    # HIGHER LOW STRUCTURE
    # -----------------------------------------------------

    df["LOW5"] = low.rolling(5).min()
    df["LOW10"] = low.rolling(10).min()
    df["LOW20"] = low.rolling(20).min()

    return df


# =========================================================
# DOWNLOAD DATA
# =========================================================

@st.cache_data(
    ttl=1800,
    show_spinner=False
)
def download_stocks(symbols, batch_size):

    results = {}

    tickers = [
        f"{x}.NS"
        for x in symbols
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

                        temp = data[ticker].copy()
                        temp = temp.dropna(
                            how="all"
                        )

                        if len(temp) > 0:

                            symbol = ticker.replace(
                                ".NS",
                                ""
                            )

                            results[
                                symbol
                            ] = temp

                    except Exception:
                        continue

        except Exception:
            continue

    return results


# =========================================================
# NIFTY BENCHMARK
# =========================================================

@st.cache_data(
    ttl=1800,
    show_spinner=False
)
def get_nifty_benchmark():

    try:

        df = yf.download(
            "^NSEI",
            period="2y",
            interval="1d",
            auto_adjust=False,
            progress=False
        )

        if isinstance(
            df.columns,
            pd.MultiIndex
        ):
            df.columns = df.columns.get_level_values(0)

        return df.dropna()

    except Exception:

        return pd.DataFrame()


# =========================================================
# ANALYSIS
# =========================================================

def analyze_stock(
    symbol,
    df,
    nifty_df,
    max_breakout_distance,
    strict_mode
):

    try:

        df = add_indicators(df)

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
        atr_pct = float(row["ATR_PCT"])

        rsi = float(row["RSI"])

        volume_ratio = float(
            row["VOL_RATIO"]
        )

        volume_dryup = float(
            row["VOLUME_DRYUP"]
        )

        range5 = float(row["RANGE5"])
        range10 = float(row["RANGE10"])
        range20 = float(row["RANGE20"])

        breakout = float(
            row["BREAKOUT20"]
        )

        breakout50 = float(
            row["BREAKOUT50"]
        )

        dip90 = bool(
            row["DIP90"]
        )

        low5 = float(row["LOW5"])
        low10 = float(row["LOW10"])
        low20 = float(row["LOW20"])

        values = [
            price,
            sma50,
            sma150,
            ema220,
            high52,
            low52,
            atr,
            atr_pct,
            rsi,
            volume_ratio,
            volume_dryup,
            range5,
            range10,
            range20,
            breakout,
            breakout50,
            low5,
            low10,
            low20
        ]

        if any(
            pd.isna(x)
            for x in values
        ):
            return None

        # =================================================
        # CORE TREND
        # =================================================

        trend1 = (
            sma150 > ema220
        )

        trend2 = (
            price > sma50
        )

        trend3 = (
            sma50 > sma150
        )

        trend_alignment = (
            trend1 and
            trend2 and
            trend3
        )

        # =================================================
        # 52 WEEK POSITION
        # =================================================

        above_25_low = (
            price >
            1.25 * low52
        )

        distance_from_high = (
            (high52 - price)
            / price
            * 100
        )

        near_52_high = (
            distance_from_high <= 20
        )

        # =================================================
        # BREAKOUT
        # =================================================

        breakout_distance = (
            (breakout - price)
            / price
            * 100
        )

        near_breakout = (
            0 <=
            breakout_distance <=
            max_breakout_distance
        )

        already_broken = (
            price > breakout
        )

        # =================================================
        # RANGE CONTRACTION
        # =================================================

        contraction_1 = (
            range5 < range10
        )

        contraction_2 = (
            range10 < range20
        )

        sequential_contraction = (
            contraction_1 and
            contraction_2
        )

        tight_base = (
            range20 <= 18
        )

        # =================================================
        # VOLUME
        # =================================================

        volume_dry = (
            volume_dryup <= 0.90
        )

        volume_strong = (
            volume_ratio >= 1.20
        )

        # =================================================
        # ATR COMPRESSION
        # =================================================

        atr_compressed = (
            atr_pct <= 3.5
        )

        # =================================================
        # HIGHER LOWS
        # =================================================

        higher_lows = (
            low5 >= low10 * 0.985
            and
            low10 >= low20 * 0.985
        )

        # =================================================
        # RESISTANCE QUALITY
        # =================================================

        resistance_close = (
            breakout_distance <= 7
        )

        resistance_quality = (
            breakout <= breakout50 * 1.08
        )

        # =================================================
        # RSI
        # =================================================

        healthy_rsi = (
            45 <= rsi <= 72
        )

        # =================================================
        # RELATIVE STRENGTH
        # =================================================

        relative_strength = np.nan

        try:

            stock_close = df["Close"]

            if len(stock_close) >= 60 and len(nifty_df) >= 60:

                stock_return = (
                    stock_close.iloc[-1] /
                    stock_close.iloc[-61] -
                    1
                ) * 100

                nifty_return = (
                    float(nifty_df["Close"].iloc[-1]) /
                    float(nifty_df["Close"].iloc[-61]) -
                    1
                ) * 100

                relative_strength = (
                    stock_return -
                    nifty_return
                )

        except Exception:
            relative_strength = np.nan

        rs_strong = (
            not pd.isna(relative_strength)
            and
            relative_strength >= 5
        )

        # =================================================
        # VCP SCORE
        # =================================================

        vcp = 0

        if contraction_1:
            vcp += 1

        if contraction_2:
            vcp += 1

        if volume_dry:
            vcp += 1

        if atr_compressed:
            vcp += 1

        if higher_lows:
            vcp += 1

        if near_breakout:
            vcp += 1

        # =================================================
        # HARD FILTER
        # =================================================

        if strict_mode:

            if not trend_alignment:
                return None

            if not above_25_low:
                return None

            if not dip90:
                return None

            if already_broken:
                return None

            if not near_breakout:
                return None

            if not tight_base:
                return None

            if vcp < 3:
                return None

        # =================================================
        # SCORE
        # =================================================

        score = 0

        # Trend - 20
        if trend1:
            score += 7

        if trend2:
            score += 7

        if trend3:
            score += 6

        # Base - 20
        if tight_base:
            score += 8

        if sequential_contraction:
            score += 8

        if higher_lows:
            score += 4

        # Volume - 15
        if volume_dry:
            score += 10

        if volume_strong:
            score += 5

        # Breakout - 15
        if near_breakout:
            score += 10

        if resistance_quality:
            score += 5

        # Momentum - 10
        if healthy_rsi:
            score += 5

        if rs_strong:
            score += 5

        # 52 week - 5
        if near_52_high:
            score += 5

        # ATR - 5
        if atr_compressed:
            score += 5

        # Previous EMA220 dip - 10
        if dip90:
            score += 10

        score = min(
            score,
            100
        )

        # =================================================
        # QUALITY
        # =================================================

        if score >= 90:
            quality = "🔥 EXCEPTIONAL"

        elif score >= 82:
            quality = "🔥 VERY HIGH"

        elif score >= 75:
            quality = "🟢 HIGH"

        elif score >= 68:
            quality = "🟡 GOOD"

        elif score >= 60:
            quality = "⚪ WATCH"

        else:
            quality = "🔴 WEAK"

        # =================================================
        # TRADE LEVELS
        # =================================================

        entry = breakout

        stop_loss = max(
            ema220,
            entry - 2 * atr
        )

        risk = (
            entry -
            stop_loss
        )

        if risk <= 0:
            return None

        target1 = (
            entry +
            risk * 1.5
        )

        target2 = (
            entry +
            risk * 2
        )

        target3 = (
            entry +
            risk * 3
        )

        # =================================================
        # SETUP STATUS
        # =================================================

        if (
            near_breakout
            and
            sequential_contraction
            and
            volume_dry
            and
            vcp >= 4
        ):

            status = "🚀 READY"

        elif (
            near_breakout
            and
            vcp >= 3
        ):

            status = "🟢 NEAR BREAKOUT"

        else:

            status = "🟡 DEVELOPING"

        return {

            "Symbol": symbol,

            "Score": round(
                score,
                1
            ),

            "Quality": quality,

            "Status": status,

            "Close": round(
                price,
                2
            ),

            "Breakout": round(
                breakout,
                2
            ),

            "Distance %": round(
                breakout_distance,
                2
            ),

            "Entry": round(
                entry,
                2
            ),

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

            "VCP": f"{vcp}/6",

            "RSI": round(
                rsi,
                1
            ),

            "Volume Ratio": round(
                volume_ratio,
                2
            ),

            "Volume Dry-up": round(
                volume_dryup,
                2
            ),

            "ATR %": round(
                atr_pct,
                2
            ),

            "Range 5D %": round(
                range5,
                2
            ),

            "Range 10D %": round(
                range10,
                2
            ),

            "Range 20D %": round(
                range20,
                2
            ),

            "Relative Strength": (
                round(
                    relative_strength,
                    2
                )
                if not pd.isna(relative_strength)
                else np.nan
            ),

            "52W High Distance %": round(
                distance_from_high,
                2
            ),

            "EMA220 Dip": (
                "YES"
                if dip90
                else "NO"
            )
        }

    except Exception:
        return None


# =========================================================
# MAIN
# =========================================================

universe = get_universe(
    universe_choice
)

c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "Stocks",
        len(universe)
    )

with c2:
    st.metric(
        "Minimum Score",
        minimum_score
    )

with c3:
    st.metric(
        "Breakout Zone",
        f"{max_breakout_distance}%"
    )

st.markdown("---")

run = st.button(
    "🔎 RUN V3 PRE-BREAKOUT SCANNER",
    type="primary",
    use_container_width=True
)

if run:

    progress = st.progress(0)

    status = st.empty()

    status.info(
        f"Downloading data for "
        f"{len(universe)} stocks..."
    )

    stock_data = download_stocks(
        tuple(universe),
        batch_size
    )

    status.info(
        "Downloading NIFTY benchmark..."
    )

    nifty = get_nifty_benchmark()

    status.info(
        f"Analysing {len(stock_data)} stocks..."
    )

    results = []

    total = len(stock_data)

    for i, (
        symbol,
        df
    ) in enumerate(
        stock_data.items()
    ):

        result = analyze_stock(
            symbol,
            df,
            nifty,
            max_breakout_distance,
            strict_mode
        )

        if result is not None:

            if result["Score"] >= minimum_score:
                results.append(result)

        if total > 0:

            progress.progress(
                int(
                    (i + 1) /
                    total *
                    100
                )
            )

    progress.empty()

    if results:

        result_df = pd.DataFrame(
            results
        )

        result_df = result_df.sort_values(
            [
                "Score",
                "VCP",
                "Distance %"
            ],
            ascending=[
                False,
                False,
                True
            ]
        )

        result_df = result_df.reset_index(
            drop=True
        )

        status.success(
            f"Scan completed — "
            f"{len(result_df)} high-quality "
            f"pre-breakout setups found."
        )

        # =================================================
        # TOP SETUPS
        # =================================================

        st.subheader(
            "🔥 Highest Conviction Pre-Breakout Setups"
        )

        top_columns = [
            "Symbol",
            "Score",
            "Quality",
            "Status",
            "Close",
            "Breakout",
            "Distance %",
            "VCP",
            "RSI",
            "Volume Dry-up",
            "ATR %",
            "Relative Strength"
        ]

        st.dataframe(
            result_df[
                top_columns
            ].head(15),
            use_container_width=True,
            hide_index=True
        )

        # =================================================
        # BEST SETUP
        # =================================================

        best = result_df.iloc[0]

        st.markdown("---")

        st.subheader(
            f"🎯 #1 Pre-Breakout Setup — "
            f"{best['Symbol']}"
        )

        a, b, c, d = st.columns(4)

        with a:
            st.metric(
                "Score",
                f"{best['Score']}/100"
            )

        with b:
            st.metric(
                "Status",
                best["Status"]
            )

        with c:
            st.metric(
                "Breakout",
                f"₹{best['Breakout']:,.2f}"
            )

        with d:
            st.metric(
                "VCP",
                best["VCP"]
            )

        # =================================================
        # TRADE LEVELS
        # =================================================

        st.subheader(
            "📌 Trade Levels"
        )

        a, b, c, d, e = st.columns(5)

        with a:
            st.metric(
                "Entry",
                f"₹{best['Entry']:,.2f}"
            )

        with b:
            st.metric(
                "Stop Loss",
                f"₹{best['Stop Loss']:,.2f}"
            )

        with c:
            st.metric(
                "Target 1",
                f"₹{best['Target 1']:,.2f}"
            )

        with d:
            st.metric(
                "Target 2",
                f"₹{best['Target 2']:,.2f}"
            )

        with e:
            st.metric(
                "Target 3",
                f"₹{best['Target 3']:,.2f}"
            )

        # =================================================
        # DIAGNOSTICS
        # =================================================

        st.subheader(
            "📊 Setup Diagnostics"
        )

        a, b, c, d, e, f = st.columns(6)

        with a:
            st.metric(
                "RSI",
                best["RSI"]
            )

        with b:
            st.metric(
                "Volume",
                f"{best['Volume Ratio']}x"
            )

        with c:
            st.metric(
                "Dry-up",
                best["Volume Dry-up"]
            )

        with d:
            st.metric(
                "ATR",
                f"{best['ATR %']}%"
            )

        with e:
            st.metric(
                "RS",
                f"{best['Relative Strength']}%"
            )

        with f:
            st.metric(
                "52W High",
                f"{best['52W High Distance %']}%"
            )

        # =================================================
        # RANGE
        # =================================================

        st.subheader(
            "📦 Base Contraction"
        )

        r1, r2, r3 = st.columns(3)

        with r1:
            st.metric(
                "5D Range",
                f"{best['Range 5D %']}%"
            )

        with r2:
            st.metric(
                "10D Range",
                f"{best['Range 10D %']}%"
            )

        with r3:
            st.metric(
                "20D Range",
                f"{best['Range 20D %']}%"
            )

        # =================================================
        # ALL RESULTS
        # =================================================

        st.markdown("---")

        st.subheader(
            "📋 All Qualifying Stocks"
        )

        st.dataframe(
            result_df,
            use_container_width=True,
            hide_index=True
        )

        # =================================================
        # CSV
        # =================================================

        csv = result_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download V3 Scanner Results",
            csv,
            "nse_prebreakout_v3.csv",
            "text/csv",
            use_container_width=True
        )

    else:

        status.warning(
            "No stocks currently meet the V3 criteria. "
            "Try reducing Minimum Score or temporarily "
            "turning off Strict Pre-Breakout Mode."
        )

else:

    st.info(
        "Press 🔎 RUN V3 PRE-BREAKOUT SCANNER to begin."
    )
