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
    page_title="NSE Pre-Breakout Scanner V4",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 NSE Pre-Breakout Scanner V4")
st.caption(
    "Advanced VCP + Pre-Breakout + Historical Validation Engine"
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
    "Minimum technical score",
    40,
    90,
    65,
    5
)

strict_mode = st.sidebar.checkbox(
    "Strict Pre-Breakout Mode",
    value=True
)

st.sidebar.markdown("---")

st.sidebar.header("📊 Historical Validation")

backtest_years = st.sidebar.selectbox(
    "Historical period",
    [1, 2, 3],
    index=1
)

forward_days = st.sidebar.selectbox(
    "Forward test period",
    [10, 15, 20, 30],
    index=2
)

minimum_occurrences = st.sidebar.number_input(
    "Minimum historical setups",
    min_value=3,
    max_value=50,
    value=5,
    step=1
)

st.sidebar.info(
    "Historical success means +1R was reached "
    "before -1R within the selected forward period."
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

    # -----------------------------
    # Trend
    # -----------------------------

    df["SMA50"] = close.rolling(50).mean()

    df["SMA150"] = close.rolling(150).mean()

    df["EMA220"] = close.ewm(
        span=220,
        adjust=False
    ).mean()

    # -----------------------------
    # 52 week
    # -----------------------------

    df["52W_HIGH"] = (
        high.rolling(252).max()
    )

    df["52W_LOW"] = (
        low.rolling(252).min()
    )

    # -----------------------------
    # ATR
    # -----------------------------

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
        df["ATR14"] /
        close *
        100
    )

    # -----------------------------
    # RSI
    # -----------------------------

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = (
        avg_gain /
        avg_loss.replace(0, np.nan)
    )

    df["RSI"] = (
        100 -
        (100 / (1 + rs))
    )

    # -----------------------------
    # Volume
    # -----------------------------

    df["VOL20"] = volume.rolling(20).mean()

    df["VOL_RATIO"] = (
        volume /
        df["VOL20"].replace(0, np.nan)
    )

    df["VOL5_AVG"] = (
        volume.rolling(5).mean()
    )

    df["VOLUME_DRYUP"] = (
        df["VOL5_AVG"] /
        df["VOL20"].replace(0, np.nan)
    )

    # -----------------------------
    # Range
    # -----------------------------

    df["RANGE5"] = (
        (
            high.rolling(5).max() -
            low.rolling(5).min()
        )
        / close *
        100
    )

    df["RANGE10"] = (
        (
            high.rolling(10).max() -
            low.rolling(10).min()
        )
        / close *
        100
    )

    df["RANGE20"] = (
        (
            high.rolling(20).max() -
            low.rolling(20).min()
        )
        / close *
        100
    )

    # -----------------------------
    # Breakout resistance
    # -----------------------------

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

    # -----------------------------
    # EMA220 dip
    # -----------------------------

    df["BELOW220"] = (
        low < df["EMA220"]
    )

    df["DIP90"] = (
        df["BELOW220"]
        .rolling(90)
        .max()
    )

    # -----------------------------
    # Base lows
    # -----------------------------

    df["LOW5"] = low.rolling(5).min()

    df["LOW10"] = low.rolling(10).min()

    df["LOW20"] = low.rolling(20).min()

    return df


# =========================================================
# DOWNLOAD MARKET DATA
# =========================================================

@st.cache_data(
    ttl=1800,
    show_spinner=False
)
def download_market_data(
    symbols,
    years=3
):

    results = {}

    tickers = [
        f"{symbol}.NS"
        for symbol in symbols
    ]

    period = f"{years}y"

    # Smaller batches for stability
    batch_size = 50

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
                period=period,
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

                        if len(temp) > 260:

                            symbol = (
                                ticker
                                .replace(".NS", "")
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
def get_nifty():

    try:

        df = yf.download(
            "^NSEI",
            period="3y",
            interval="1d",
            auto_adjust=False,
            progress=False
        )

        if isinstance(
            df.columns,
            pd.MultiIndex
        ):
            df.columns = (
                df.columns
                .get_level_values(0)
            )

        return df.dropna()

    except Exception:

        return pd.DataFrame()


# =========================================================
# V3 TECHNICAL ANALYSIS
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

        trend1 = sma150 > ema220
        trend2 = price > sma50
        trend3 = sma50 > sma150

        trend_alignment = (
            trend1 and
            trend2 and
            trend3
        )

        above_25_low = (
            price >
            1.25 * low52
        )

        distance_from_high = (
            (high52 - price)
            / price *
            100
        )

        near_52_high = (
            distance_from_high <= 20
        )

        breakout_distance = (
            (breakout - price)
            / price *
            100
        )

        near_breakout = (
            0 <=
            breakout_distance <=
            max_breakout_distance
        )

        already_broken = (
            price > breakout
        )

        contraction1 = (
            range5 < range10
        )

        contraction2 = (
            range10 < range20
        )

        sequential_contraction = (
            contraction1 and
            contraction2
        )

        tight_base = (
            range20 <= 18
        )

        volume_dry = (
            volume_dryup <= 0.90
        )

        volume_strong = (
            volume_ratio >= 1.20
        )

        atr_compressed = (
            atr_pct <= 3.5
        )

        higher_lows = (
            low5 >= low10 * 0.985
            and
            low10 >= low20 * 0.985
        )

        resistance_quality = (
            breakout <=
            breakout50 * 1.08
        )

        healthy_rsi = (
            45 <= rsi <= 72
        )

        relative_strength = np.nan

        try:

            if (
                len(df) >= 60
                and
                len(nifty_df) >= 60
            ):

                stock_return = (
                    df["Close"].iloc[-1] /
                    df["Close"].iloc[-61] -
                    1
                ) * 100

                nifty_return = (
                    float(
                        nifty_df["Close"].iloc[-1]
                    ) /
                    float(
                        nifty_df["Close"].iloc[-61]
                    ) -
                    1
                ) * 100

                relative_strength = (
                    stock_return -
                    nifty_return
                )

        except Exception:
            pass

        rs_strong = (
            not pd.isna(relative_strength)
            and
            relative_strength >= 5
        )

        vcp = 0

        if contraction1:
            vcp += 1

        if contraction2:
            vcp += 1

        if volume_dry:
            vcp += 1

        if atr_compressed:
            vcp += 1

        if higher_lows:
            vcp += 1

        if near_breakout:
            vcp += 1

        # Strict filter
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

        # Score
        score = 0

        if trend1:
            score += 7

        if trend2:
            score += 7

        if trend3:
            score += 6

        if tight_base:
            score += 8

        if sequential_contraction:
            score += 8

        if higher_lows:
            score += 4

        if volume_dry:
            score += 10

        if volume_strong:
            score += 5

        if near_breakout:
            score += 10

        if resistance_quality:
            score += 5

        if healthy_rsi:
            score += 5

        if rs_strong:
            score += 5

        if near_52_high:
            score += 5

        if atr_compressed:
            score += 5

        if dip90:
            score += 10

        score = min(
            score,
            100
        )

        if score >= 90:
            quality = "🔥 EXCEPTIONAL"
        elif score >= 82:
            quality = "🔥 VERY HIGH"
        elif score >= 75:
            quality = "🟢 HIGH"
        elif score >= 68:
            quality = "🟡 GOOD"
        else:
            quality = "⚪ WATCH"

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
            "Score": round(score, 1),
            "Quality": quality,
            "Status": status,
            "Close": round(price, 2),
            "Breakout": round(breakout, 2),
            "Distance %": round(
                breakout_distance,
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
            "VCP": f"{vcp}/6",
            "RSI": round(rsi, 1),
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
                if not pd.isna(
                    relative_strength
                )
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
# HISTORICAL SIGNAL
# =========================================================

def historical_signal(
    df,
    index,
    max_breakout_distance,
    strict_mode
):

    try:

        row = df.iloc[index]

        price = float(row["Close"])

        sma50 = float(row["SMA50"])
        sma150 = float(row["SMA150"])
        ema220 = float(row["EMA220"])

        low52 = float(row["52W_LOW"])

        atr = float(row["ATR14"])
        atr_pct = float(row["ATR_PCT"])

        rsi = float(row["RSI"])

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
            low52,
            atr,
            atr_pct,
            rsi,
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

        # -----------------------------
        # Trend
        # -----------------------------

        trend1 = (
            sma150 > ema220
        )

        trend2 = (
            price > sma50
        )

        trend3 = (
            sma50 > sma150
        )

        # -----------------------------
        # Price position
        # -----------------------------

        above_25_low = (
            price >
            1.25 * low52
        )

        # -----------------------------
        # Breakout proximity
        # -----------------------------

        distance = (
            (breakout - price)
            / price *
            100
        )

        near_breakout = (
            0 <=
            distance <=
            max_breakout_distance
        )

        already_broken = (
            price > breakout
        )

        # -----------------------------
        # Contraction
        # -----------------------------

        contraction1 = (
            range5 < range10
        )

        contraction2 = (
            range10 < range20
        )

        sequential_contraction = (
            contraction1 and
            contraction2
        )

        tight_base = (
            range20 <= 18
        )

        # -----------------------------
        # Volume
        # -----------------------------

        volume_dry = (
            volume_dryup <= 0.90
        )

        # -----------------------------
        # ATR
        # -----------------------------

        atr_compressed = (
            atr_pct <= 3.5
        )

        # -----------------------------
        # Higher lows
        # -----------------------------

        higher_lows = (
            low5 >= low10 * 0.985
            and
            low10 >= low20 * 0.985
        )

        # -----------------------------
        # Resistance
        # -----------------------------

        resistance_quality = (
            breakout <=
            breakout50 * 1.08
        )

        # -----------------------------
        # RSI
        # -----------------------------

        healthy_rsi = (
            45 <= rsi <= 72
        )

        # -----------------------------
        # VCP
        # -----------------------------

        vcp = 0

        if contraction1:
            vcp += 1

        if contraction2:
            vcp += 1

        if volume_dry:
            vcp += 1

        if atr_compressed:
            vcp += 1

        if higher_lows:
            vcp += 1

        if near_breakout:
            vcp += 1

        # -----------------------------
        # Strict historical setup
        # -----------------------------

        if strict_mode:

            if not (
                trend1
                and trend2
                and trend3
            ):
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

        # -----------------------------
        # Historical score
        # -----------------------------

        score = 0

        if trend1:
            score += 7

        if trend2:
            score += 7

        if trend3:
            score += 6

        if tight_base:
            score += 8

        if sequential_contraction:
            score += 8

        if higher_lows:
            score += 4

        if volume_dry:
            score += 10

        if near_breakout:
            score += 10

        if resistance_quality:
            score += 5

        if healthy_rsi:
            score += 5

        if atr_compressed:
            score += 5

        if dip90:
            score += 10

        if above_25_low:
            score += 10

        score = min(
            score,
            100
        )

        if score < minimum_score:
            return None

        # -----------------------------
        # Entry / risk
        # -----------------------------

        entry = breakout

        stop = max(
            ema220,
            entry - 2 * atr
        )

        risk = (
            entry -
            stop
        )

        if risk <= 0:
            return None

        target = (
            entry +
            risk
        )

        return {
            "entry": entry,
            "stop": stop,
            "target": target,
            "score": score,
            "vcp": vcp
        }

    except Exception:

        return None


# =========================================================
# BACKTEST ONE STOCK
# =========================================================

def backtest_stock(
    symbol,
    raw_df,
    max_breakout_distance,
    strict_mode,
    years,
    forward_period
):

    try:

        df = add_indicators(
            raw_df
        )

        if df is None:
            return None

        # Only use approximately requested period.
        cutoff = (
            df.index.max()
            - pd.Timedelta(
                days=365 * years
            )
        )

        start_index = max(
            260,
            df.index.searchsorted(
                cutoff
            )
        )

        end_index = (
            len(df) -
            forward_period -
            1
        )

        if end_index <= start_index:
            return None

        trades = []

        for i in range(
            start_index,
            end_index
        ):

            signal = historical_signal(
                df,
                i,
                max_breakout_distance,
                strict_mode
            )

            if signal is None:
                continue

            entry = signal["entry"]
            stop = signal["stop"]
            target = signal["target"]

            future = df.iloc[
                i + 1:
                i + 1 + forward_period
            ]

            if future.empty:
                continue

            outcome = "OPEN"

            exit_return = np.nan

            hit_day = None

            max_gain = (
                (
                    future["High"].max() /
                    entry
                ) - 1
            ) * 100

            max_loss = (
                (
                    future["Low"].min() /
                    entry
                ) - 1
            ) * 100

            # -----------------------------------------
            # Determine which level was reached first
            # -----------------------------------------

            for j in range(
                len(future)
            ):

                day = future.iloc[j]

                day_high = float(
                    day["High"]
                )

                day_low = float(
                    day["Low"]
                )

                target_hit = (
                    day_high >= target
                )

                stop_hit = (
                    day_low <= stop
                )

                # Conservative handling:
                # if both occur on the same candle,
                # count it as failure because daily
                # OHLC cannot tell us intraday order.
                if (
                    target_hit
                    and
                    stop_hit
                ):

                    outcome = "FAIL"

                    exit_return = (
                        (
                            stop /
                            entry
                        ) - 1
                    ) * 100

                    hit_day = j + 1

                    break

                if target_hit:

                    outcome = "SUCCESS"

                    exit_return = (
                        (
                            target /
                            entry
                        ) - 1
                    ) * 100

                    hit_day = j + 1

                    break

                if stop_hit:

                    outcome = "FAIL"

                    exit_return = (
                        (
                            stop /
                            entry
                        ) - 1
                    ) * 100

                    hit_day = j + 1

                    break

            if outcome == "OPEN":

                last_close = float(
                    future["Close"].iloc[-1]
                )

                exit_return = (
                    (
                        last_close /
                        entry
                    ) - 1
                ) * 100

                hit_day = forward_period

            trades.append(
                {
                    "Date":
                        df.index[i],

                    "Score":
                        signal["score"],

                    "VCP":
                        signal["vcp"],

                    "Outcome":
                        outcome,

                    "Return %":
                        exit_return,

                    "Max Gain %":
                        max_gain,

                    "Max Loss %":
                        max_loss,

                    "Days":
                        hit_day
                }
            )

        if not trades:
            return None

        trade_df = pd.DataFrame(
            trades
        )

        total = len(
            trade_df
        )

        success = int(
            (
                trade_df["Outcome"]
                == "SUCCESS"
            ).sum()
        )

        failure = int(
            (
                trade_df["Outcome"]
                == "FAIL"
            ).sum()
        )

        open_trades = int(
            (
                trade_df["Outcome"]
                == "OPEN"
            ).sum()
        )

        success_rate = (
            success /
            total *
            100
        )

        average_return = (
            trade_df["Return %"]
            .mean()
        )

        average_gain = (
            trade_df["Max Gain %"]
            .mean()
        )

        average_loss = (
            trade_df["Max Loss %"]
            .mean()
        )

        average_days = (
            trade_df["Days"]
            .mean()
        )

        # Only completed wins/losses for expectancy
        completed = trade_df[
            trade_df["Outcome"].isin(
                ["SUCCESS", "FAIL"]
            )
        ]

        if not completed.empty:

            win_returns = completed[
                completed["Outcome"]
                == "SUCCESS"
            ]["Return %"]

            loss_returns = completed[
                completed["Outcome"]
                == "FAIL"
            ]["Return %"]

            avg_win = (
                win_returns.mean()
                if not win_returns.empty
                else np.nan
            )

            avg_loss = (
                loss_returns.mean()
                if not loss_returns.empty
                else np.nan
            )

            if (
                not pd.isna(avg_win)
                and
                not pd.isna(avg_loss)
                and
                avg_loss != 0
            ):

                reward_risk = (
                    abs(avg_win / avg_loss)
                )

            else:

                reward_risk = np.nan

        else:

            avg_win = np.nan
            avg_loss = np.nan
            reward_risk = np.nan

        return {
            "Symbol": symbol,
            "Historical Setups": total,
            "Successful": success,
            "Failed": failure,
            "Open": open_trades,
            "Success Rate %":
                round(
                    success_rate,
                    1
                ),
            "Avg Return %":
                round(
                    average_return,
                    2
                ),
            "Avg Max Gain %":
                round(
                    average_gain,
                    2
                ),
            "Avg Max Loss %":
                round(
                    average_loss,
                    2
                ),
            "Avg Win %":
                round(
                    avg_win,
                    2
                )
                if not pd.isna(avg_win)
                else np.nan,
            "Avg Loss %":
                round(
                    avg_loss,
                    2
                )
                if not pd.isna(avg_loss)
                else np.nan,
            "Historical R:R":
                round(
                    reward_risk,
                    2
                )
                if not pd.isna(reward_risk)
                else np.nan,
            "Avg Days":
                round(
                    average_days,
                    1
                )
        }

    except Exception:

        return None


# =========================================================
# UI
# =========================================================

universe = get_universe(
    universe_choice
)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Universe",
        len(universe)
    )

with c2:
    st.metric(
        "Minimum Score",
        minimum_score
    )

with c3:
    st.metric(
        "Historical Years",
        backtest_years
    )

with c4:
    st.metric(
        "Forward Days",
        forward_days
    )

st.markdown("---")

run = st.button(
    "🔎 RUN V4 SCANNER + HISTORICAL VALIDATION",
    type="primary",
    use_container_width=True
)

if run:

    progress = st.progress(0)

    status = st.empty()

    # =====================================================
    # CURRENT SCAN
    # =====================================================

    status.info(
        f"Scanning {len(universe)} NSE stocks..."
    )

    current_data = download_market_data(
        tuple(universe),
        years=3
    )

    status.info(
        "Loading NIFTY benchmark..."
    )

    nifty = get_nifty()

    status.info(
        f"Analysing {len(current_data)} stocks..."
    )

    candidates = []

    total = len(
        current_data
    )

    for i, (
        symbol,
        df
    ) in enumerate(
        current_data.items()
    ):

        result = analyze_stock(
            symbol,
            df,
            nifty,
            max_breakout_distance,
            strict_mode
        )

        if result is not None:

            if (
                result["Score"]
                >=
                minimum_score
            ):

                candidates.append(
                    result
                )

        if total > 0:

            progress.progress(
                int(
                    (i + 1) /
                    total *
                    50
                )
            )

    # =====================================================
    # CURRENT RESULTS
    # =====================================================

    if candidates:

        candidates_df = pd.DataFrame(
            candidates
        )

        candidates_df = (
            candidates_df
            .sort_values(
                "Score",
                ascending=False
            )
            .reset_index(drop=True)
        )

        status.info(
            f"Found {len(candidates_df)} "
            "current qualifying setups. "
            "Running historical validation..."
        )

        # =================================================
        # HISTORICAL TEST
        # =================================================

        historical_results = []

        total_candidates = len(
            candidates_df
        )

        for i, symbol in enumerate(
            candidates_df["Symbol"]
        ):

            if symbol not in current_data:
                continue

            hist = backtest_stock(
                symbol,
                current_data[symbol],
                max_breakout_distance,
                strict_mode,
                backtest_years,
                forward_days
            )

            if hist is not None:

                historical_results.append(
                    hist
                )

            progress.progress(
                50 +
                int(
                    (
                        (i + 1) /
                        total_candidates
                    ) * 50
                )
            )

        progress.empty()

        # =================================================
        # MERGE
        # =================================================

        if historical_results:

            hist_df = pd.DataFrame(
                historical_results
            )

            final_df = candidates_df.merge(
                hist_df,
                on="Symbol",
                how="left"
            )

            # Historical success rate
            final_df[
                "Success Rate %"
            ] = final_df[
                "Success Rate %"
            ].fillna(0)

            # Combined ranking
            final_df[
                "Validated Score"
            ] = (
                final_df["Score"] * 0.60
                +
                final_df["Success Rate %"] * 0.40
            )

            final_df[
                "Validated Score"
            ] = final_df[
                "Validated Score"
            ].round(1)

            final_df = (
                final_df
                .sort_values(
                    "Validated Score",
                    ascending=False
                )
                .reset_index(drop=True)
            )

        else:

            final_df = candidates_df.copy()

        status.success(
            f"V4 completed — "
            f"{len(candidates_df)} current "
            "setups analysed."
        )

        # =================================================
        # CURRENT SETUPS
        # =================================================

        st.subheader(
            "🔥 Current Pre-Breakout Setups"
        )

        display_cols = [
            "Symbol",
            "Score",
            "Quality",
            "Status",
            "Close",
            "Breakout",
            "Distance %",
            "VCP",
            "RSI"
        ]

        st.dataframe(
            candidates_df[
                display_cols
            ],
            use_container_width=True,
            hide_index=True
        )

        # =================================================
        # VALIDATED RANKING
        # =================================================

        if (
            "Success Rate %"
            in final_df.columns
        ):

            st.markdown("---")

            st.subheader(
                "🏆 Historically Validated Ranking"
            )

            st.caption(
                "Validated Score = 60% technical "
                "setup score + 40% historical "
                "success rate. It is a ranking "
                "metric, not a guaranteed probability."
            )

            validation_cols = [
                "Symbol",
                "Score",
                "Success Rate %",
                "Historical Setups",
                "Successful",
                "Failed",
                "Avg Return %",
                "Avg Max Gain %",
                "Avg Max Loss %",
                "Historical R:R",
                "Validated Score"
            ]

            st.dataframe(
                final_df[
                    [
                        x
                        for x in validation_cols
                        if x in final_df.columns
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )

            # =================================================
            # BEST VALIDATED SETUP
            # =================================================

            valid_rows = final_df[
                final_df[
                    "Historical Setups"
                ].fillna(0)
                >=
                minimum_occurrences
            ]

            if not valid_rows.empty:

                best = (
                    valid_rows
                    .iloc[0]
                )

                st.markdown("---")

                st.subheader(
                    f"🎯 #1 Historically Validated Setup — "
                    f"{best['Symbol']}"
                )

                a, b, c, d = st.columns(4)

                with a:

                    st.metric(
                        "Technical Score",
                        f"{best['Score']}/100"
                    )

                with b:

                    st.metric(
                        "Historical Success",
                        f"{best['Success Rate %']}%"
                    )

                with c:

                    st.metric(
                        "Historical Setups",
                        int(
                            best[
                                "Historical Setups"
                            ]
                        )
                    )

                with d:

                    rr = best[
                        "Historical R:R"
                    ]

                    st.metric(
                        "Historical R:R",
                        (
                            f"{rr}"
                            if not pd.isna(rr)
                            else "N/A"
                        )
                    )

                # =============================================
                # TRADE LEVELS
                # =============================================

                st.subheader(
                    "📌 Current Trade Levels"
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

                # =============================================
                # HISTORICAL DETAILS
                # =============================================

                st.subheader(
                    "📊 Historical Performance"
                )

                a, b, c, d, e, f = st.columns(6)

                with a:

                    st.metric(
                        "Setups",
                        int(
                            best[
                                "Historical Setups"
                            ]
                        )
                    )

                with b:

                    st.metric(
                        "Wins",
                        int(
                            best[
                                "Successful"
                            ]
                        )
                    )

                with c:

                    st.metric(
                        "Failures",
                        int(
                            best[
                                "Failed"
                            ]
                        )
                    )

                with d:

                    st.metric(
                        "Avg Return",
                        f"{best['Avg Return %']}%"
                    )

                with e:

                    st.metric(
                        "Avg Max Gain",
                        f"{best['Avg Max Gain %']}%"
                    )

                with f:

                    st.metric(
                        "Avg Max Loss",
                        f"{best['Avg Max Loss %']}%"
                    )

                st.info(
                    "Historical results are based on "
                    "daily OHLC data. When both the "
                    "target and stop are touched on "
                    "the same daily candle, the test "
                    "conservatively counts it as a loss "
                    "because daily data cannot determine "
                    "which level was reached first."
                )

        # =================================================
        # COMPLETE TABLE
        # =================================================

        st.markdown("---")

        st.subheader(
            "📋 Complete V4 Results"
        )

        st.dataframe(
            final_df,
            use_container_width=True,
            hide_index=True
        )

        # =================================================
        # CSV
        # =================================================

        csv = final_df.to_csv(
            index=False
        ).encode(
            "utf-8"
        )

        st.download_button(
            "⬇️ Download V4 Results",
            csv,
            "nse_prebreakout_v4.csv",
            "text/csv",
            use_container_width=True
        )

    else:

        progress.empty()

        status.warning(
            "No current stocks meet the selected "
            "V4 criteria. Try lowering the minimum "
            "technical score or temporarily disabling "
            "Strict Pre-Breakout Mode."
        )

else:

    st.info(
        "Press 🔎 RUN V4 SCANNER + HISTORICAL "
        "VALIDATION to begin."
    )
