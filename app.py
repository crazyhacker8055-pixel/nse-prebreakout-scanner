import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime

# ============================================================
# STOCKPILOT V4.1
# REAL BREAKOUT VALIDATION ENGINE
# ============================================================

st.set_page_config(
    page_title="StockPilot V4.1",
    page_icon="🚀",
    layout="wide"
)

# ------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------

NIFTY500_URL = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def clean_columns(df):
    """Flatten yfinance MultiIndex columns when necessary."""
    if isinstance(df.columns, pd.MultiIndex):
        if len(df.columns.levels) >= 2:
            try:
                df.columns = df.columns.get_level_values(0)
            except Exception:
                df.columns = [x[0] for x in df.columns]

    df.columns = [str(c).strip().title() for c in df.columns]
    return df


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def pct(x):
    if pd.isna(x):
        return "—"
    return f"{x:.1f}%"


def load_nifty500():
    """Load current NIFTY 500 symbols."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "text/csv,*/*"
        }

        r = requests.get(
            NIFTY500_URL,
            headers=headers,
            timeout=20
        )
        r.raise_for_status()

        from io import StringIO
        df = pd.read_csv(StringIO(r.text))

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

        return [s for s in symbols if s]

    except Exception as e:
        st.warning(f"NIFTY 500 list could not be loaded: {e}")
        return []


@st.cache_data(ttl=1800, show_spinner=False)
def download_market_data(symbols, period="2y"):
    """Download market data in batches."""
    tickers = [f"{s}.NS" for s in symbols]

    try:
        data = yf.download(
            tickers=tickers,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=True
        )
        return data
    except Exception:
        return pd.DataFrame()


def extract_symbol(data, symbol):
    """Extract one ticker from batch yfinance result."""
    ticker = f"{symbol}.NS"

    if data is None or data.empty:
        return pd.DataFrame()

    try:
        if isinstance(data.columns, pd.MultiIndex):

            # Normal yfinance layout:
            # Ticker -> OHLCV
            if ticker in data.columns.get_level_values(0):
                df = data[ticker].copy()
            elif symbol in data.columns.get_level_values(0):
                df = data[symbol].copy()
            else:
                # Alternative layout:
                # OHLCV -> Ticker
                if ticker in data.columns.get_level_values(1):
                    df = data.xs(ticker, axis=1, level=1).copy()
                else:
                    return pd.DataFrame()

        else:
            df = data.copy()

        df = clean_columns(df)

        required = ["Open", "High", "Low", "Close", "Volume"]

        if not all(c in df.columns for c in required):
            return pd.DataFrame()

        df = df[required].dropna()
        return df

    except Exception:
        return pd.DataFrame()


def add_indicators(df):
    """Technical indicators used by V4.1."""
    x = df.copy()

    x["SMA20"] = x["Close"].rolling(20).mean()
    x["SMA50"] = x["Close"].rolling(50).mean()
    x["SMA150"] = x["Close"].rolling(150).mean()
    x["EMA220"] = x["Close"].ewm(span=220, adjust=False).mean()

    # ATR
    prev_close = x["Close"].shift(1)

    tr1 = x["High"] - x["Low"]
    tr2 = (x["High"] - prev_close).abs()
    tr3 = (x["Low"] - prev_close).abs()

    x["TR"] = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    x["ATR14"] = x["TR"].rolling(14).mean()

    # RSI
    delta = x["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    x["RSI14"] = 100 - (100 / (1 + rs))

    # Volume
    x["Vol20"] = x["Volume"].rolling(20).mean()
    x["VolRatio"] = x["Volume"] / x["Vol20"]

    # Ranges
    x["Range5"] = (
        x["High"].rolling(5).max()
        - x["Low"].rolling(5).min()
    ) / x["Close"] * 100

    x["Range10"] = (
        x["High"].rolling(10).max()
        - x["Low"].rolling(10).min()
    ) / x["Close"] * 100

    x["Range20"] = (
        x["High"].rolling(20).max()
        - x["Low"].rolling(20).min()
    ) / x["Close"] * 100

    # Prior resistance.
    # IMPORTANT: shift(1) prevents lookahead bias.
    x["Resistance20"] = (
        x["High"]
        .shift(1)
        .rolling(20)
        .max()
    )

    x["High52"] = (
        x["High"]
        .shift(1)
        .rolling(252)
        .max()
    )

    x["Low52"] = (
        x["Low"]
        .shift(1)
        .rolling(252)
        .min()
    )

    # Previous 90 sessions dip below 220 EMA
    below_ema = x["Low"] < x["EMA220"]

    x["Dip90"] = (
        below_ema
        .rolling(90)
        .max()
        .fillna(0)
        .astype(bool)
    )

    # ATR compression
    x["ATR20Avg"] = x["ATR14"].rolling(20).mean()
    x["ATRCompression"] = (
        x["ATR14"] / x["ATR20Avg"]
    )

    return x


def technical_score(df):
    """
    V3-style pre-breakout score.
    Returns score and diagnostic dictionary.
    """

    if len(df) < 260:
        return 0, {}

    r = df.iloc[-1]

    close = safe_float(r["Close"])
    sma20 = safe_float(r["SMA20"])
    sma50 = safe_float(r["SMA50"])
    sma150 = safe_float(r["SMA150"])
    ema220 = safe_float(r["EMA220"])

    resistance = safe_float(r["Resistance20"])
    high52 = safe_float(r["High52"])

    range5 = safe_float(r["Range5"])
    range10 = safe_float(r["Range10"])
    range20 = safe_float(r["Range20"])

    vol_ratio = safe_float(r["VolRatio"])
    atr_compression = safe_float(r["ATRCompression"])
    rsi = safe_float(r["RSI14"])

    score = 0

    # Trend alignment
    if sma150 > ema220:
        score += 10

    if close > sma50:
        score += 10

    if sma50 > sma150:
        score += 10

    # 52-week low distance
    low52 = safe_float(r["Low52"])

    if low52 > 0 and close > 1.25 * low52:
        score += 5

    # Previous dip below EMA220
    if bool(r["Dip90"]):
        score += 10

    # Breakout proximity
    breakout_distance = np.nan

    if resistance > 0:
        breakout_distance = (
            resistance - close
        ) / close * 100

        if breakout_distance <= 2:
            score += 15
        elif breakout_distance <= 5:
            score += 10
        elif breakout_distance <= 8:
            score += 5

    # Range contraction
    if (
        not pd.isna(range5)
        and not pd.isna(range10)
        and not pd.isna(range20)
    ):
        if range5 < range10:
            score += 5

        if range10 < range20:
            score += 5

        if range20 <= 20:
            score += 5

    # Volume dry-up
    if not pd.isna(vol_ratio):
        if vol_ratio <= 0.70:
            score += 5
        elif vol_ratio <= 0.90:
            score += 3

    # ATR compression
    if not pd.isna(atr_compression):
        if atr_compression <= 0.85:
            score += 5
        elif atr_compression <= 1.00:
            score += 3

    # RSI healthy zone
    if not pd.isna(rsi):
        if 45 <= rsi <= 68:
            score += 5

    # Higher-low structure
    recent_low = df["Low"].tail(10).min()
    older_low = df["Low"].iloc[-30:-10].min()

    if recent_low > older_low:
        score += 5

    # 52-week high proximity
    high_distance = np.nan

    if high52 > 0:
        high_distance = (
            high52 - close
        ) / close * 100

        if high_distance <= 5:
            score += 5
        elif high_distance <= 10:
            score += 3

    score = min(score, 100)

    return score, {
        "close": close,
        "sma50": sma50,
        "sma150": sma150,
        "ema220": ema220,
        "resistance": resistance,
        "high52": high52,
        "range5": range5,
        "range10": range10,
        "range20": range20,
        "vol_ratio": vol_ratio,
        "atr_compression": atr_compression,
        "rsi": rsi,
        "breakout_distance": breakout_distance,
        "high_distance": high_distance
    }


def stage_from_score(score, breakout_distance):
    if score >= 90:
        return "EXCEPTIONAL"
    elif score >= 80:
        return "VERY HIGH"
    elif score >= 70:
        return "HIGH"
    elif score >= 60:
        return "WATCH"
    else:
        return "LOW"


def is_prebreakout_setup(row, min_score=65, max_distance=8):
    """Historical setup definition using information available on that day."""

    needed = [
        "Close",
        "SMA50",
        "SMA150",
        "EMA220",
        "Resistance20",
        "Low52",
        "Dip90",
        "Range5",
        "Range10",
        "Range20",
        "VolRatio",
        "ATRCompression"
    ]

    for c in needed:
        if c not in row or pd.isna(row[c]):
            return False

    close = float(row["Close"])
    resistance = float(row["Resistance20"])

    if close <= 0 or resistance <= 0:
        return False

    distance = (
        resistance - close
    ) / close * 100

    # Original core trend rules
    if not (
        row["SMA150"] > row["EMA220"]
        and close > row["SMA50"]
        and row["SMA50"] > row["SMA150"]
    ):
        return False

    if float(row["Low52"]) <= 0:
        return False

    if close <= 1.25 * float(row["Low52"]):
        return False

    if not bool(row["Dip90"]):
        return False

    # Breakout proximity
    if distance > max_distance:
        return False

    # Contraction
    if float(row["Range5"]) > float(row["Range10"]):
        return False

    if float(row["Range10"]) > float(row["Range20"]):
        return False

    # ATR compression
    if float(row["ATRCompression"]) > 1.15:
        return False

    # Calculate a simplified score at this historical point.
    score = 0

    if row["SMA150"] > row["EMA220"]:
        score += 10

    if close > row["SMA50"]:
        score += 10

    if row["SMA50"] > row["SMA150"]:
        score += 10

    if close > 1.25 * row["Low52"]:
        score += 5

    if bool(row["Dip90"]):
        score += 10

    if distance <= 2:
        score += 15
    elif distance <= 5:
        score += 10
    elif distance <= 8:
        score += 5

    if row["Range5"] < row["Range10"]:
        score += 5

    if row["Range10"] < row["Range20"]:
        score += 5

    if row["Range20"] <= 20:
        score += 5

    if row["VolRatio"] <= 0.70:
        score += 5
    elif row["VolRatio"] <= 0.90:
        score += 3

    if row["ATRCompression"] <= 0.85:
        score += 5
    elif row["ATRCompression"] <= 1:
        score += 3

    if score < min_score:
        return False

    return True


# ------------------------------------------------------------
# V4.1 BACKTEST
# ------------------------------------------------------------

def backtest_symbol(
    df,
    min_score=65,
    max_breakout_distance=8,
    breakout_window=5,
    holding_period=20,
    target_r=1.5,
    cooldown=20
):
    """
    Correct historical sequence:

    1. Pre-breakout setup
    2. Confirmed close breakout above prior 20-day resistance
    3. Entry next trading day's OPEN
    4. Stop/target
    5. Maximum holding period
    6. Cooldown prevents duplicate base counting
    """

    if df.empty or len(df) < 400:
        return None

    x = df.copy()

    setups = 0
    breakouts = 0
    entries = 0

    target_hits = 0
    stop_hits = 0
    time_exits = 0
    no_breakout = 0

    trades = []

    i = 260

    last_signal_position = -9999

    while i < len(x) - breakout_window - 2:

        # ----------------------------------------------------
        # COOLDOWN
        # ----------------------------------------------------
        if i - last_signal_position < cooldown:
            i += 1
            continue

        row = x.iloc[i]

        # ----------------------------------------------------
        # PRE-BREAKOUT SETUP
        # ----------------------------------------------------
        if not is_prebreakout_setup(
            row,
            min_score=min_score,
            max_distance=max_breakout_distance
        ):
            i += 1
            continue

        setups += 1

        resistance = safe_float(row["Resistance20"])

        # ----------------------------------------------------
        # SEARCH FOR ACTUAL BREAKOUT
        # ----------------------------------------------------
        breakout_idx = None

        end_idx = min(
            i + breakout_window,
            len(x) - 2
        )

        for j in range(i + 1, end_idx + 1):

            close_j = safe_float(x.iloc[j]["Close"])

            if close_j > resistance:
                breakout_idx = j
                break

        # No actual breakout
        if breakout_idx is None:
            no_breakout += 1
            last_signal_position = i
            i += cooldown
            continue

        breakouts += 1

        # ----------------------------------------------------
        # NEXT-DAY OPEN ENTRY
        # ----------------------------------------------------
        entry_idx = breakout_idx + 1

        if entry_idx >= len(x):
            break

        entry_price = safe_float(
            x.iloc[entry_idx]["Open"]
        )

        if pd.isna(entry_price) or entry_price <= 0:
            i = entry_idx + cooldown
            continue

        entries += 1

        # ----------------------------------------------------
        # STOP
        # ----------------------------------------------------
        atr = safe_float(
            x.iloc[breakout_idx]["ATR14"]
        )

        ema220 = safe_float(
            x.iloc[breakout_idx]["EMA220"]
        )

        if pd.isna(atr) or atr <= 0:
            i = entry_idx + cooldown
            continue

        # ATR-based risk
        atr_stop = entry_price - (2.0 * atr)

        # EMA-based risk
        ema_stop = ema220 * 0.995 if not pd.isna(ema220) else atr_stop

        # Choose the closer protective stop below entry.
        candidates = [
            s for s in [atr_stop, ema_stop]
            if s > 0 and s < entry_price
        ]

        if not candidates:
            stop_price = entry_price - (2.0 * atr)
        else:
            stop_price = max(candidates)

        risk = entry_price - stop_price

        if risk <= 0:
            i = entry_idx + cooldown
            continue

        target_price = entry_price + (
            target_r * risk
        )

        # ----------------------------------------------------
        # FORWARD TRADE SIMULATION
        # ----------------------------------------------------
        outcome = "TIME EXIT"
        exit_price = np.nan
        exit_idx = None

        forward_end = min(
            entry_idx + holding_period,
            len(x) - 1
        )

        for k in range(
            entry_idx,
            forward_end + 1
        ):

            day = x.iloc[k]

            high = safe_float(day["High"])
            low = safe_float(day["Low"])
            close = safe_float(day["Close"])

            target_hit = high >= target_price
            stop_hit = low <= stop_price

            # Conservative daily-bar assumption:
            # if both occurred on same candle,
            # count STOP first.
            if target_hit and stop_hit:
                outcome = "STOP"
                exit_price = stop_price
                exit_idx = k
                break

            if stop_hit:
                outcome = "STOP"
                exit_price = stop_price
                exit_idx = k
                break

            if target_hit:
                outcome = "TARGET"
                exit_price = target_price
                exit_idx = k
                break

            # Last day = time exit
            if k == forward_end:
                outcome = "TIME EXIT"
                exit_price = close
                exit_idx = k

        if pd.isna(exit_price):
            i = entry_idx + cooldown
            continue

        return_pct = (
            exit_price / entry_price - 1
        ) * 100

        r_multiple = (
            exit_price - entry_price
        ) / risk

        if outcome == "TARGET":
            target_hits += 1

        elif outcome == "STOP":
            stop_hits += 1

        else:
            time_exits += 1

        trades.append({
            "setup_date": x.index[i],
            "breakout_date": x.index[breakout_idx],
            "entry_date": x.index[entry_idx],
            "exit_date": x.index[exit_idx],
            "entry": entry_price,
            "stop": stop_price,
            "target": target_price,
            "exit": exit_price,
            "outcome": outcome,
            "return_pct": return_pct,
            "r": r_multiple
        })

        # Prevent overlapping / duplicate signals.
        last_signal_position = exit_idx

        i = exit_idx + 1

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    trade_df = pd.DataFrame(trades)

    if trade_df.empty:
        return {
            "setups": setups,
            "breakouts": breakouts,
            "breakout_rate": (
                breakouts / setups * 100
                if setups else 0
            ),
            "entries": entries,
            "target_hits": 0,
            "stop_hits": 0,
            "time_exits": 0,
            "win_rate": 0,
            "avg_winner": 0,
            "avg_loser": 0,
            "profit_factor": 0,
            "expectancy_r": 0,
            "max_drawdown_r": 0,
            "trades": trade_df
        }

    winners = trade_df[
        trade_df["r"] > 0
    ]["r"]

    losers = trade_df[
        trade_df["r"] < 0
    ]["r"]

    win_rate = (
        len(winners)
        / len(trade_df)
        * 100
    )

    avg_winner = (
        winners.mean()
        if len(winners)
        else 0
    )

    avg_loser = (
        losers.mean()
        if len(losers)
        else 0
    )

    gross_profit = (
        winners.sum()
        if len(winners)
        else 0
    )

    gross_loss = abs(
        losers.sum()
    ) if len(losers) else 0

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else np.inf
    )

    expectancy = trade_df["r"].mean()

    equity = trade_df["r"].cumsum()

    running_max = equity.cummax()

    drawdown = equity - running_max

    max_drawdown = drawdown.min()

    return {
        "setups": setups,
        "breakouts": breakouts,
        "breakout_rate": (
            breakouts / setups * 100
            if setups else 0
        ),
        "entries": entries,
        "target_hits": target_hits,
        "stop_hits": stop_hits,
        "time_exits": time_exits,
        "win_rate": win_rate,
        "avg_winner": avg_winner,
        "avg_loser": avg_loser,
        "profit_factor": profit_factor,
        "expectancy_r": expectancy,
        "max_drawdown_r": max_drawdown,
        "trades": trade_df
    }


# ------------------------------------------------------------
# VALIDATION GRADE
# ------------------------------------------------------------

def validation_grade(stats):
    trades = stats["entries"]
    pf = stats["profit_factor"]
    exp = stats["expectancy_r"]

    if trades < 10:
        return "⚠️ LOW SAMPLE"

    if trades < 30:
        if exp > 0 and pf > 1:
            return "🟡 LIMITED POSITIVE"
        return "🟡 LIMITED"

    if exp > 0 and pf >= 1.5:
        return "🟢 STRONG"

    if exp > 0 and pf > 1:
        return "🟢 POSITIVE"

    if pf > 0.8:
        return "🟠 NEUTRAL"

    return "🔴 WEAK"


# ------------------------------------------------------------
# CURRENT SCANNER
# ------------------------------------------------------------

def scan_current(
    symbols,
    period="2y",
    min_score=65,
    max_breakout_distance=8
):

    data = download_market_data(
        symbols,
        period=period
    )

    rows = []

    for symbol in symbols:

        df = extract_symbol(
            data,
            symbol
        )

        if df.empty or len(df) < 260:
            continue

        df = add_indicators(df)

        score, d = technical_score(df)

        if not d:
            continue

        distance = d["breakout_distance"]

        if pd.isna(distance):
            continue

        if score < min_score:
            continue

        if distance > max_breakout_distance:
            continue

        stage = stage_from_score(
            score,
            distance
        )

        rows.append({
            "Symbol": symbol,
            "Score": score,
            "Stage": stage,
            "Close": d["close"],
            "Breakout": d["resistance"],
            "Distance": distance,
            "RSI": d["rsi"],
            "Volume": d["vol_ratio"],
            "ATR Comp": d["atr_compression"],
            "Range 5D": d["range5"],
            "Range 10D": d["range10"],
            "Range 20D": d["range20"],
            "52W High Dist": d["high_distance"],
            "Data": df
        })

    rows.sort(
        key=lambda x: x["Score"],
        reverse=True
    )

    return rows


# ============================================================
# UI
# ============================================================

st.title("🚀 StockPilot V4.1")
st.caption(
    "Pre-breakout → confirmed breakout → next-day entry → historical validation"
)

st.info(
    "V4.1 is a research/backtesting engine. "
    "Historical results do not guarantee future returns."
)

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------

st.sidebar.header("⚙️ Scanner Settings")

universe = st.sidebar.selectbox(
    "Universe",
    [
        "NIFTY 500",
        "NIFTY 250",
        "NIFTY 100"
    ]
)

min_score = st.sidebar.slider(
    "Minimum Technical Score",
    50,
    95,
    65,
    5
)

max_breakout_distance = st.sidebar.slider(
    "Maximum Breakout Distance %",
    2.0,
    15.0,
    8.0,
    0.5
)

strict_mode = st.sidebar.checkbox(
    "Strict Pre-Breakout Mode",
    True
)

st.sidebar.header("📊 Historical Validation")

history_years = st.sidebar.selectbox(
    "Historical Period",
    [1, 2, 3],
    index=1
)

forward_days = st.sidebar.selectbox(
    "Maximum Holding Period",
    [10, 15, 20, 30],
    index=2
)

breakout_window = st.sidebar.selectbox(
    "Breakout Window",
    [3, 5, 7],
    index=1
)

target_r = st.sidebar.selectbox(
    "Target",
    [1.0, 1.5, 2.0],
    index=1
)

cooldown = st.sidebar.selectbox(
    "Signal Cooldown",
    [10, 15, 20, 30],
    index=2
)

backtest_min_score = st.sidebar.slider(
    "Historical Setup Score",
    50,
    90,
    min_score,
    5
)

run_button = st.sidebar.button(
    "🔍 RUN V4.1",
    type="primary",
    use_container_width=True
)

# ------------------------------------------------------------
# INTRO
# ------------------------------------------------------------

if not run_button:

    st.markdown(
        """
### V4.1 tests the complete trading sequence

**1️⃣ Pre-breakout setup**

Trend + consolidation + contraction + breakout proximity.

**2️⃣ Confirmed breakout**

The stock must actually close above its previous 20-session resistance.

**3️⃣ Entry**

Entry is the **next trading day's OPEN**.

**4️⃣ Risk**

Stop is based on ATR/EMA220 structure.

**5️⃣ Target**

Choose 1R, 1.5R or 2R.

**6️⃣ Outcome**

TARGET / STOP / TIME EXIT.

**7️⃣ Validation**

Profit factor + expectancy + win rate + drawdown + sample size.

---

### ⭐ What changed from V4?

V4 asked:

> "Was this a good historical setup?"

V4.1 asks:

> "Did this setup actually produce a tradable breakout and profitable trade?"
        """
    )

    st.warning(
        "Run the scanner when ready."
    )

    st.stop()


# ------------------------------------------------------------
# LOAD UNIVERSE
# ------------------------------------------------------------

with st.spinner("Loading NSE universe..."):

    all_symbols = load_nifty500()

if not all_symbols:

    st.error(
        "Unable to load NIFTY 500 symbols."
    )
    st.stop()


if universe == "NIFTY 500":
    symbols = all_symbols

elif universe == "NIFTY 250":
    symbols = all_symbols[:250]

else:
    symbols = all_symbols[:100]


# ------------------------------------------------------------
# CURRENT SCAN
# ------------------------------------------------------------

with st.spinner(
    f"Scanning {len(symbols)} stocks..."
):

    current = scan_current(
        symbols,
        period="2y",
        min_score=min_score,
        max_breakout_distance=max_breakout_distance
    )


st.success(
    f"Current scan completed — {len(current)} candidates found."
)

if not current:

    st.warning(
        "No stocks match the current filters. "
        "Try lowering the minimum score or increasing breakout distance."
    )
    st.stop()


# ------------------------------------------------------------
# CURRENT TABLE
# ------------------------------------------------------------

display_rows = []

for item in current:

    display_rows.append({
        "Symbol": item["Symbol"],
        "Score": item["Score"],
        "Stage": item["Stage"],
        "Close": round(item["Close"], 2),
        "Breakout": round(item["Breakout"], 2),
        "Distance": round(item["Distance"], 2),
        "RSI": round(item["RSI"], 1),
        "Vol": round(item["Volume"], 2),
        "ATR Comp": round(item["ATR Comp"], 2),
        "5D": round(item["Range 5D"], 2),
        "10D": round(item["Range 10D"], 2),
        "20D": round(item["Range 20D"], 2)
    })

current_df = pd.DataFrame(display_rows)

st.subheader("🔥 Current Pre-Breakout Candidates")

st.dataframe(
    current_df,
    use_container_width=True,
    hide_index=True
)


# ------------------------------------------------------------
# HISTORICAL VALIDATION
# ------------------------------------------------------------

st.subheader("🧪 V4.1 Historical Breakout Validation")

history_period = (
    f"{history_years + 1}y"
)

# We use one extra year so EMA220 and rolling indicators
# have enough warm-up data.

top_candidates = current[:10]

validation_rows = []

progress = st.progress(0)

for n, item in enumerate(top_candidates):

    symbol = item["Symbol"]

    try:

        hist = yf.download(
            f"{symbol}.NS",
            period=history_period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False
        )

        hist = clean_columns(hist)

        if hist.empty:
            continue

        required = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]

        if not all(
            c in hist.columns
            for c in required
        ):
            continue

        hist = hist[required].dropna()

        hist = add_indicators(hist)

        stats = backtest_symbol(
            hist,
            min_score=backtest_min_score,
            max_breakout_distance=max_breakout_distance,
            breakout_window=breakout_window,
            holding_period=forward_days,
            target_r=target_r,
            cooldown=cooldown
        )

        if stats is None:
            continue

        validation_rows.append({
            "Symbol": symbol,
            "Tech Score": item["Score"],
            "Setups": stats["setups"],
            "Breakouts": stats["breakouts"],
            "Breakout Rate": stats["breakout_rate"],
            "Entries": stats["entries"],
            "Targets": stats["target_hits"],
            "Stops": stats["stop_hits"],
            "Time Exit": stats["time_exits"],
            "Win Rate": stats["win_rate"],
            "Avg Winner R": stats["avg_winner"],
            "Avg Loser R": stats["avg_loser"],
            "Profit Factor": stats["profit_factor"],
            "Expectancy R": stats["expectancy_r"],
            "Max DD R": stats["max_drawdown_r"],
            "Validation": validation_grade(stats)
        })

    except Exception as e:

        st.warning(
            f"{symbol}: historical test skipped."
        )

    progress.progress(
        (n + 1) / len(top_candidates)
    )


progress.empty()


if not validation_rows:

    st.error(
        "No historical validation results were produced."
    )
    st.stop()


validation_df = pd.DataFrame(
    validation_rows
)

# Sort by expectancy first, then profit factor,
# then sample size.

validation_df["_sort_expectancy"] = (
    validation_df["Expectancy R"]
)

validation_df["_sort_pf"] = (
    validation_df["Profit Factor"]
)

validation_df["_sort_entries"] = (
    validation_df["Entries"]
)

validation_df = validation_df.sort_values(
    [
        "_sort_expectancy",
        "_sort_pf",
        "_sort_entries"
    ],
    ascending=False
)

validation_df = validation_df.drop(
    columns=[
        "_sort_expectancy",
        "_sort_pf",
        "_sort_entries"
    ]
)


# ------------------------------------------------------------
# DISPLAY VALIDATION
# ------------------------------------------------------------

st.dataframe(
    validation_df,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "Important: daily OHLC data cannot know the exact intraday order "
    "when both target and stop are touched on the same candle. "
    "V4.1 conservatively counts such cases as STOP."
)


# ------------------------------------------------------------
# BEST VALIDATED STOCK
# ------------------------------------------------------------

best_symbol = validation_df.iloc[0]["Symbol"]

best_item = next(
    x for x in current
    if x["Symbol"] == best_symbol
)

best_validation = validation_df.iloc[0]


st.markdown("---")

st.subheader(
    f"🏆 Best Validated Candidate — {best_symbol}"
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Technical Score",
    f"{best_item['Score']}/100"
)

c2.metric(
    "Win Rate",
    f"{best_validation['Win Rate']:.1f}%"
)

c3.metric(
    "Profit Factor",
    (
        f"{best_validation['Profit Factor']:.2f}"
        if np.isfinite(best_validation["Profit Factor"])
        else "∞"
    )
)

c4.metric(
    "Expectancy",
    f"{best_validation['Expectancy R']:.2f}R"
)


st.write(
    f"**Validation:** {best_validation['Validation']}"
)

st.write(
    f"**Historical setups:** "
    f"{int(best_validation['Setups'])}  |  "
    f"**Breakouts:** "
    f"{int(best_validation['Breakouts'])}  |  "
    f"**Entries:** "
    f"{int(best_validation['Entries'])}"
)

st.write(
    f"**Targets:** "
    f"{int(best_validation['Targets'])}  |  "
    f"**Stops:** "
    f"{int(best_validation['Stops'])}  |  "
    f"**Time exits:** "
    f"{int(best_validation['Time Exit'])}"
)

st.write(
    f"**Average winner:** "
    f"{best_validation['Avg Winner R']:.2f}R  |  "
    f"**Average loser:** "
    f"{best_validation['Avg Loser R']:.2f}R  |  "
    f"**Maximum drawdown:** "
    f"{best_validation['Max DD R']:.2f}R"
)


# ------------------------------------------------------------
# CURRENT TRADE ZONE
# ------------------------------------------------------------

st.subheader(
    f"📍 Current Setup — {best_symbol}"
)

close = best_item["Close"]
breakout = best_item["Breakout"]

current_df_item = best_item["Data"]

latest = current_df_item.iloc[-1]

atr = safe_float(
    latest["ATR14"]
)

ema220 = safe_float(
    latest["EMA220"]
)

# Current theoretical entry is breakout level.
# Actual trading entry remains next-day open
# after confirmed breakout.

entry_zone = breakout

atr_stop = (
    entry_zone - 2 * atr
    if atr > 0
    else np.nan
)

ema_stop = (
    ema220 * 0.995
    if not pd.isna(ema220)
    else np.nan
)

stop_candidates = [
    x for x in [
        atr_stop,
        ema_stop
    ]
    if not pd.isna(x)
    and x > 0
    and x < entry_zone
]

if stop_candidates:
    stop = max(stop_candidates)
else:
    stop = atr_stop

risk = (
    entry_zone - stop
    if not pd.isna(stop)
    else np.nan
)

target1 = (
    entry_zone + risk
    if not pd.isna(risk)
    else np.nan
)

target2 = (
    entry_zone + 1.5 * risk
    if not pd.isna(risk)
    else np.nan
)

target3 = (
    entry_zone + 2 * risk
    if not pd.isna(risk)
    else np.nan
)

a, b, c, d = st.columns(4)

a.metric(
    "Current Price",
    f"₹{close:,.2f}"
)

b.metric(
    "Breakout",
    f"₹{breakout:,.2f}"
)

c.metric(
    "Indicative Stop",
    f"₹{stop:,.2f}"
    if not pd.isna(stop)
    else "—"
)

d.metric(
    "Risk",
    f"₹{risk:,.2f}"
    if not pd.isna(risk)
    else "—"
)

st.write(
    f"**1R:** ₹{target1:,.2f}  |  "
    f"**1.5R:** ₹{target2:,.2f}  |  "
    f"**2R:** ₹{target3:,.2f}"
)

st.warning(
    "These are research levels, not guaranteed trade recommendations. "
    "The actual V4.1 backtest entry is the next day's OPEN after a confirmed breakout."
)


# ------------------------------------------------------------
# EXPLANATION
# ------------------------------------------------------------

with st.expander("🔎 How V4.1 validates a trade"):

    st.markdown(
        """
### Historical sequence

**A. Setup**

The stock must satisfy the pre-breakout rules.

↓

**B. Breakout**

Within the configured breakout window, the closing price must cross above the previous 20-session resistance.

↓

**C. Entry**

The following trading day's OPEN becomes the historical entry.

↓

**D. Stop**

The engine uses an ATR/EMA-based protective stop.

↓

**E. Target**

The target is selected using the configured R multiple.

↓

**F. Exit**

The trade exits when:

- Target is reached
- Stop is reached
- Maximum holding period expires

↓

**G. Statistics**

The engine calculates:

- Breakout rate
- Win rate
- Average winner
- Average loser
- Profit factor
- Expectancy
- Maximum drawdown
- Sample size
        """
    )


st.caption(
    f"StockPilot V4.1 | Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)
