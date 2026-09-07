
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO
from datetime import datetime

# ============================================================
# STOCKPILOT V5
# CONSOLIDATED PRECISION PRE-BREAKOUT RESEARCH ENGINE
# ============================================================

st.set_page_config(
    page_title="StockPilot V5",
    page_icon="🚀",
    layout="wide",
)

NIFTY500_URL = (
    "https://archives.nseindia.com/content/indices/"
    "ind_nifty500list.csv"
)

# -----------------------------
# Basic helpers
# -----------------------------

def num(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def clean_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df.columns = df.columns.get_level_values(0)
        except Exception:
            df.columns = [
                c[0] if isinstance(c, tuple) else c
                for c in df.columns
            ]
    df.columns = [str(c).strip().title() for c in df.columns]
    return df


# -----------------------------
# NIFTY 500 universe
# -----------------------------

@st.cache_data(ttl=1800, show_spinner=False)
def load_universe():
    try:
        r = requests.get(
            NIFTY500_URL,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=20,
        )
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        if "Symbol" not in df.columns:
            return pd.DataFrame()
        df["Symbol"] = (
            df["Symbol"].astype(str).str.strip()
        )
        return df.drop_duplicates("Symbol")
    except Exception:
        return pd.DataFrame()


# -----------------------------
# Market data
# -----------------------------

@st.cache_data(ttl=1800, show_spinner=False)
def download_batch(symbols, period="2y"):
    tickers = [f"{s}.NS" for s in symbols]
    try:
        return yf.download(
            tickers=tickers,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=True,
        )
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=1800, show_spinner=False)
def download_single(symbol, period="5y"):
    try:
        df = yf.download(
            f"{symbol}.NS",
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        df = clean_columns(df)
        required = ["Open", "High", "Low", "Close", "Volume"]
        if df.empty or not all(c in df.columns for c in required):
            return pd.DataFrame()
        return df[required].dropna()
    except Exception:
        return pd.DataFrame()


def extract_symbol(data, symbol):
    ticker = f"{symbol}.NS"
    if data is None or data.empty:
        return pd.DataFrame()
    try:
        if isinstance(data.columns, pd.MultiIndex):
            l0 = data.columns.get_level_values(0)
            l1 = data.columns.get_level_values(1)
            if ticker in l0:
                df = data[ticker].copy()
            elif symbol in l0:
                df = data[symbol].copy()
            elif ticker in l1:
                df = data.xs(ticker, axis=1, level=1).copy()
            else:
                return pd.DataFrame()
        else:
            df = data.copy()
        df = clean_columns(df)
        required = ["Open", "High", "Low", "Close", "Volume"]
        if not all(c in df.columns for c in required):
            return pd.DataFrame()
        return df[required].dropna()
    except Exception:
        return pd.DataFrame()


# -----------------------------
# Indicators
# -----------------------------

def add_indicators(df):
    x = df.copy()

    x["SMA20"] = x["Close"].rolling(20).mean()
    x["SMA50"] = x["Close"].rolling(50).mean()
    x["SMA150"] = x["Close"].rolling(150).mean()
    x["EMA220"] = x["Close"].ewm(
        span=220, adjust=False
    ).mean()

    prev_close = x["Close"].shift(1)
    tr = pd.concat(
        [
            x["High"] - x["Low"],
            (x["High"] - prev_close).abs(),
            (x["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    x["ATR14"] = tr.rolling(14).mean()
    x["ATR20"] = x["ATR14"].rolling(20).mean()
    x["ATRRatio"] = x["ATR14"] / x["ATR20"]

    delta = x["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    x["RSI"] = 100 - (100 / (1 + rs))

    x["Vol20"] = x["Volume"].rolling(20).mean()
    x["VolRatio"] = x["Volume"] / x["Vol20"]

    for n in (5, 10, 20):
        x[f"Range{n}"] = (
            (
                x["High"].rolling(n).max()
                - x["Low"].rolling(n).min()
            )
            / x["Close"]
            * 100
        )

    x["Resistance20"] = (
        x["High"].shift(1).rolling(20).max()
    )

    x["High52"] = (
        x["High"].shift(1).rolling(252).max()
    )
    x["Low52"] = (
        x["Low"].shift(1).rolling(252).min()
    )

    x["Dip90"] = (
        (x["Low"] < x["EMA220"])
        .rolling(90)
        .max()
        .fillna(0)
        .astype(bool)
    )

    x["Low10"] = x["Low"].rolling(10).min()
    x["PrevLow20"] = (
        x["Low"].shift(10).rolling(20).min()
    )

    return x


# -----------------------------
# Technical score
# -----------------------------

def technical_score(df):
    if len(df) < 260:
        return 0, {}

    r = df.iloc[-1]

    close = num(r["Close"])
    sma50 = num(r["SMA50"])
    sma150 = num(r["SMA150"])
    ema220 = num(r["EMA220"])
    resistance = num(r["Resistance20"])
    high52 = num(r["High52"])
    low52 = num(r["Low52"])
    range5 = num(r["Range5"])
    range10 = num(r["Range10"])
    range20 = num(r["Range20"])
    vol = num(r["VolRatio"])
    atr_ratio = num(r["ATRRatio"])
    rsi = num(r["RSI"])

    score = 0

    # Trend: 30
    trend_ok = (
        sma150 > ema220
        and close > sma50
        and sma50 > sma150
    )
    if sma150 > ema220:
        score += 10
    if close > sma50:
        score += 10
    if sma50 > sma150:
        score += 10

    # Location: 10
    if low52 > 0 and close > 1.25 * low52:
        score += 5

    dip90 = bool(r["Dip90"])
    if dip90:
        score += 5

    # Breakout proximity: 15
    distance = np.nan
    if resistance > 0:
        distance = (resistance - close) / close * 100
        if 0 <= distance <= 2:
            score += 15
        elif 2 < distance <= 5:
            score += 10
        elif 5 < distance <= 8:
            score += 5

    # Contraction: 15
    vcp = 0
    if range5 < range10:
        score += 5
        vcp += 1
    if range10 < range20:
        score += 5
        vcp += 1
    if range20 <= 20:
        score += 5
        vcp += 1

    # Volume: 10
    if not pd.isna(vol):
        if vol <= 0.65:
            score += 10
        elif vol <= 0.80:
            score += 7
        elif vol <= 1.00:
            score += 3

    # ATR: 5
    if not pd.isna(atr_ratio):
        if atr_ratio <= 0.80:
            score += 5
        elif atr_ratio <= 0.95:
            score += 3

    # Higher low: 5
    higher_low = (
        not pd.isna(r["Low10"])
        and not pd.isna(r["PrevLow20"])
        and r["Low10"] > r["PrevLow20"]
    )
    if higher_low:
        score += 5

    # RSI: 5
    if not pd.isna(rsi) and 45 <= rsi <= 68:
        score += 5

    # 52W high proximity: 5
    high_distance = np.nan
    if high52 > 0:
        high_distance = (high52 - close) / close * 100
        if high_distance <= 5:
            score += 5
        elif high_distance <= 10:
            score += 3

    return min(int(score), 100), {
        "close": close,
        "sma50": sma50,
        "sma150": sma150,
        "ema220": ema220,
        "resistance": resistance,
        "high52": high52,
        "low52": low52,
        "distance": distance,
        "high_distance": high_distance,
        "range5": range5,
        "range10": range10,
        "range20": range20,
        "vol": vol,
        "atr_ratio": atr_ratio,
        "rsi": rsi,
        "vcp": vcp,
        "higher_low": higher_low,
        "trend_ok": trend_ok,
        "dip90": dip90,
    }


def strict_current_filter(score, d, minimum_score):
    if score < minimum_score:
        return False

    distance = d["distance"]

    if pd.isna(distance) or distance < 0 or distance > 5:
        return False

    if not d["trend_ok"] or not d["dip90"]:
        return False

    if not (d["range5"] < d["range10"] < d["range20"]):
        return False

    if d["vcp"] < 2:
        return False

    if not d["higher_low"]:
        return False

    if not pd.isna(d["vol"]) and d["vol"] > 1.05:
        return False

    if not pd.isna(d["atr_ratio"]) and d["atr_ratio"] > 1.05:
        return False

    if not pd.isna(d["rsi"]) and not (40 <= d["rsi"] <= 72):
        return False

    return True


# -----------------------------
# Current scan
# -----------------------------

def current_scan(symbols, minimum_score):
    data = download_batch(symbols, "2y")
    results = []

    for symbol in symbols:
        df = extract_symbol(data, symbol)
        if df.empty or len(df) < 260:
            continue

        df = add_indicators(df)
        score, d = technical_score(df)

        if not strict_current_filter(
            score, d, minimum_score
        ):
            continue

        stage = (
            "READY"
            if d["distance"] <= 2
            else "FORMING"
        )

        results.append({
            "Symbol": symbol,
            "Tech": score,
            "Stage": stage,
            "Data": df,
            "Details": d,
        })

    results.sort(
        key=lambda x: (
            x["Tech"],
            -x["Details"]["distance"]
            if not pd.isna(x["Details"]["distance"])
            else -999
        ),
        reverse=True,
    )

    return results


# -----------------------------
# Market regime
# -----------------------------

@st.cache_data(ttl=1800, show_spinner=False)
def market_context():
    try:
        df = yf.download(
            "^NSEI",
            period="2y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        df = clean_columns(df)
        if df.empty or "Close" not in df.columns:
            return {
                "regime": "UNKNOWN",
                "score": 50,
                "return60": np.nan,
                "breadth50": np.nan,
                "breadth200": np.nan,
            }

        close = df["Close"]
        ema20 = close.ewm(span=20, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        ema200 = close.ewm(span=200, adjust=False).mean()

        last = float(close.iloc[-1])
        e20 = float(ema20.iloc[-1])
        e50 = float(ema50.iloc[-1])
        e200 = float(ema200.iloc[-1])

        ret60 = (
            (last / float(close.iloc[-61]) - 1) * 100
            if len(close) > 61 else np.nan
        )

        if last > e20 > e50 > e200 and (
            pd.isna(ret60) or ret60 > 0
        ):
            regime, score = "BULLISH", 85
        elif last > e50 > e200:
            regime, score = "BULLISH / SELECTIVE", 75
        elif last > e200 and last > e50:
            regime, score = "NEUTRAL / POSITIVE", 65
        elif last < e50 and last < e200:
            regime, score = "WEAK", 40
        else:
            regime, score = "NEUTRAL", 55

        return {
            "regime": regime,
            "score": score,
            "return60": ret60,
            "nifty_close": last,
        }
    except Exception:
        return {
            "regime": "UNKNOWN",
            "score": 50,
            "return60": np.nan,
            "nifty_close": np.nan,
        }


def breadth_score(current_results, all_symbols, batch):
    above50 = 0
    above200 = 0
    valid = 0

    for symbol in all_symbols:
        df = extract_symbol(batch, symbol)
        if df.empty or len(df) < 220:
            continue

        x = add_indicators(df)
        r = x.iloc[-1]

        if pd.isna(r["SMA50"]) or pd.isna(r["SMA150"]):
            continue

        valid += 1
        if r["Close"] > r["SMA50"]:
            above50 += 1
        if r["Close"] > r["SMA150"]:
            above200 += 1

    if valid == 0:
        return np.nan, np.nan

    return (
        above50 / valid * 100,
        above200 / valid * 100,
    )


# -----------------------------
# Sector strength
# -----------------------------

def sector_strength(universe_df, symbols, batch):
    sector_map = {}

    if "Industry" in universe_df.columns:
        for _, row in universe_df.iterrows():
            sector_map[str(row["Symbol"])] = (
                str(row["Industry"])
            )
    elif "Sector" in universe_df.columns:
        for _, row in universe_df.iterrows():
            sector_map[str(row["Symbol"])] = (
                str(row["Sector"])
            )

    sector_returns = {}

    for symbol in symbols:
        df = extract_symbol(batch, symbol)
        if df.empty or len(df) < 65:
            continue

        industry = sector_map.get(symbol, "Other")
        c = df["Close"]

        try:
            ret20 = (
                c.iloc[-1] / c.iloc[-21] - 1
            ) * 100
            ret60 = (
                c.iloc[-1] / c.iloc[-61] - 1
            ) * 100

            sector_returns.setdefault(
                industry, []
            ).append(
                0.4 * ret20 + 0.6 * ret60
            )
        except Exception:
            continue

    rows = []
    for sector, vals in sector_returns.items():
        if len(vals) < 3:
            continue
        rows.append({
            "Sector": sector,
            "Strength": float(np.median(vals)),
            "Stocks": len(vals),
        })

    if not rows:
        return {}, pd.DataFrame()

    sdf = pd.DataFrame(rows)
    sdf["Rank"] = (
        sdf["Strength"]
        .rank(pct=True)
        * 100
    )

    mapping = dict(
        zip(
            sdf["Sector"],
            sdf["Rank"]
        )
    )

    return mapping, sdf.sort_values(
        "Strength",
        ascending=False
    )


# -----------------------------
# Historical setup definition
# -----------------------------

def historical_setup(row):
    required = [
        "Close", "SMA50", "SMA150", "EMA220",
        "Resistance20", "Low52", "Dip90",
        "Range5", "Range10", "Range20",
        "VolRatio", "ATRRatio", "Low10",
        "PrevLow20", "RSI"
    ]

    if any(
        c not in row or pd.isna(row[c])
        for c in required
    ):
        return False

    close = float(row["Close"])
    resistance = float(row["Resistance20"])

    if close <= 0 or resistance <= 0:
        return False

    distance = (
        (resistance - close)
        / close
        * 100
    )

    if distance < 0 or distance > 5:
        return False

    if not (
        row["SMA150"] > row["EMA220"]
        and row["Close"] > row["SMA50"]
        and row["SMA50"] > row["SMA150"]
    ):
        return False

    if row["Low52"] <= 0 or (
        close <= 1.25 * row["Low52"]
    ):
        return False

    if not bool(row["Dip90"]):
        return False

    if not (
        row["Range5"]
        < row["Range10"]
        < row["Range20"]
    ):
        return False

    if row["ATRRatio"] > 1.05:
        return False

    if row["VolRatio"] > 1.05:
        return False

    if not (
        row["Low10"] > row["PrevLow20"]
    ):
        return False

    if not (
        40 <= row["RSI"] <= 72
    ):
        return False

    return True


# -----------------------------
# Historical replay
# -----------------------------

def historical_replay( df, breakout_window=5, holding_period=20, target_r=1.5, cooldown=20, ):
    if df.empty or len(df) < 400:
        return None

    x = df.copy()

    setups = 0
    breakouts = 0
    entries = 0
    targets = 0
    stops = 0
    time_exits = 0
    trades = []

    i = 260
    last_event = -9999

    while i < len(x) - breakout_window - 2:

        if i - last_event < cooldown:
            i += 1
            continue

        row = x.iloc[i]

        if not historical_setup(row):
            i += 1
            continue

        setups += 1
        resistance = num(row["Resistance20"])

        breakout_idx = None
        end = min(
            i + breakout_window,
            len(x) - 2
        )

        for j in range(i + 1, end + 1):
            # Close-confirmed breakout.
            if num(x.iloc[j]["Close"]) > resistance:
                breakout_idx = j
                break

        if breakout_idx is None:
            last_event = i
            i += 1
            continue

        breakouts += 1

        entry_idx = breakout_idx + 1
        if entry_idx >= len(x):
            break

        entry = num(
            x.iloc[entry_idx]["Open"]
        )

        if pd.isna(entry) or entry <= 0:
            i += 1
            continue

        atr = num(
            x.iloc[breakout_idx]["ATR14"]
        )
        ema = num(
            x.iloc[breakout_idx]["EMA220"]
        )

        if pd.isna(atr) or atr <= 0:
            i += 1
            continue

        atr_stop = entry - 2 * atr
        ema_stop = (
            ema * 0.995
            if not pd.isna(ema)
            else np.nan
        )

        stops_available = [
            s for s in [atr_stop, ema_stop]
            if not pd.isna(s)
            and 0 < s < entry
        ]

        stop = (
            max(stops_available)
            if stops_available
            else atr_stop
        )

        risk = entry - stop
        if risk <= 0:
            i += 1
            continue

        target = entry + target_r * risk
        entries += 1

        outcome = "TIME EXIT"
        exit_price = np.nan
        exit_idx = None

        last = min(
            entry_idx + holding_period,
            len(x) - 1
        )

        for k in range(entry_idx, last + 1):
            high = num(x.iloc[k]["High"])
            low = num(x.iloc[k]["Low"])
            close = num(x.iloc[k]["Close"])

            hit_target = high >= target
            hit_stop = low <= stop

            # Conservative: if both are touched
            # in one daily bar, count STOP.
            if hit_target and hit_stop:
                outcome = "STOP"
                exit_price = stop
                exit_idx = k
                break

            if hit_stop:
                outcome = "STOP"
                exit_price = stop
                exit_idx = k
                break

            if hit_target:
                outcome = "TARGET"
                exit_price = target
                exit_idx = k
                break

            if k == last:
                outcome = "TIME EXIT"
                exit_price = close
                exit_idx = k

        if exit_idx is None or pd.isna(exit_price):
            i += 1
            continue

        r_multiple = (
            (exit_price - entry) / risk
        )

        if outcome == "TARGET":
            targets += 1
        elif outcome == "STOP":
            stops += 1
        else:
            time_exits += 1

        trades.append({
            "Setup": x.index[i],
            "Breakout": x.index[breakout_idx],
            "Entry": x.index[entry_idx],
            "Exit": x.index[exit_idx],
            "R": r_multiple,
            "Outcome": outcome,
        })

        last_event = exit_idx
        i = exit_idx + 1

    tdf = pd.DataFrame(trades)

    if tdf.empty:
        return {
            "setups": setups,
            "breakouts": breakouts,
            "breakout_rate": (
                breakouts / setups * 100
                if setups else 0
            ),
            "entries": 0,
            "targets": 0,
            "stops": 0,
            "time_exits": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "expectancy": 0,
            "max_dd": 0,
            "trades": tdf,
        }

    winners = tdf.loc[
        tdf["R"] > 0, "R"
    ]
    losers = tdf.loc[
        tdf["R"] < 0, "R"
    ]

    gross_profit = (
        winners.sum()
        if len(winners) else 0
    )
    gross_loss = (
        abs(losers.sum())
        if len(losers) else 0
    )

    pf = (
        gross_profit / gross_loss
        if gross_loss > 0
        else np.inf
    )

    equity = tdf["R"].cumsum()
    drawdown = equity - equity.cummax()

    return {
        "setups": setups,
        "breakouts": breakouts,
        "breakout_rate": (
            breakouts / setups * 100
            if setups else 0
        ),
        "entries": len(tdf),
        "targets": targets,
        "stops": stops,
        "time_exits": time_exits,
        "win_rate": (
            len(winners) / len(tdf) * 100
        ),
        "profit_factor": pf,
        "expectancy": float(tdf["R"].mean()),
        "max_dd": float(drawdown.min()),
        "trades": tdf,
    }


# -----------------------------
# Evidence score
# -----------------------------

def wilson_lower_bound(wins, n, z=1.96):
    if n <= 0:
        return 0.0
    p = wins / n
    denom = 1 + z**2 / n
    centre = p + z**2 / (2*n)
    adj = z * np.sqrt(
        (p*(1-p) + z**2/(4*n)) / n
    )
    return max(
        0.0,
        (centre - adj) / denom
    )


def evidence_score(stats):
    n = int(stats["entries"])

    if n <= 0:
        return 0.0

    wins = int(
        round(
            stats["win_rate"]
            / 100
            * n
        )
    )

    # Conservative win-rate component.
    lower_win = (
        wilson_lower_bound(wins, n)
        * 100
    )

    win_component = min(
        lower_win / 60 * 25,
        25
    )

    pf = stats["profit_factor"]
    if np.isfinite(pf):
        pf_component = min(
            max(pf, 0) / 2.0 * 25,
            25
        )
    else:
        # Infinity with very small samples is not
        # allowed to receive a free maximum score.
        pf_component = (
            12 if n < 10 else 22
        )

    exp_component = min(
        max(stats["expectancy"], 0)
        / 1.5
        * 25,
        25
    )

    breakout_component = min(
        max(stats["breakout_rate"], 0)
        / 70
        * 10,
        10
    )

    raw = (
        win_component
        + pf_component
        + exp_component
        + breakout_component
    )

    if n >= 30:
        sample_factor = 1.00
    elif n >= 20:
        sample_factor = 0.92
    elif n >= 10:
        sample_factor = 0.82
    elif n >= 5:
        sample_factor = 0.60
    else:
        sample_factor = 0.35

    return round(
        min(raw * sample_factor, 100),
        1
    )


def evidence_grade(n):
    if n < 5:
        return "🔴 INSUFFICIENT"
    if n < 10:
        return "🟠 LIMITED"
    if n < 20:
        return "🟡 DEVELOPING"
    if n < 30:
        return "🟢 VALIDATED"
    return "🟢🟢 STRONG"


# -----------------------------
# UI
# -----------------------------

st.title("🚀 StockPilot V5")

st.caption(
    "Consolidated NSE pre-breakout research engine — "
    "technical setup + historical evidence + market regime + sector strength"
)

st.info(
    "Daily research data. This is not guaranteed real-time market data "
    "and historical evidence is not a prediction."
)

universe_df = load_universe()

if universe_df.empty:
    st.error("Unable to load the NIFTY 500 universe.")
    st.stop()

# Actual current NIFTY 500 list from NSE.
symbols = universe_df["Symbol"].tolist()

st.sidebar.header("⚙️ Scanner")

st.sidebar.caption(
    "Universe: official current NIFTY 500 constituents"
)

minimum_score = st.sidebar.slider(
    "Minimum Technical Score",
    75, 95, 78
)

st.sidebar.header("🧪 Historical Replay")

holding_period = st.sidebar.selectbox(
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
    "Target R",
    [1.0, 1.5, 2.0],
    index=1
)

cooldown = st.sidebar.selectbox(
    "Signal Cooldown",
    [10, 15, 20, 30],
    index=2
)

run = st.sidebar.button(
    "🚀 RUN STOCKPILOT V5",
    type="primary",
    use_container_width=True
)

if not run:
    st.markdown(
        """ ### V5 pipeline **NIFTY 500** → **Trend** → **EMA220 recovery history** → **VCP / contraction** → **Volume dry-up** → **ATR compression** → **Higher lows** → **Pre-breakout proximity** → **Market regime** → **Sector strength** → **Historical breakout replay** → **Evidence-adjusted ranking** The final ranking does not treat a 1–2 trade sample as strong evidence. """
    )
    st.stop()

# -----------------------------
# Scan
# -----------------------------

with st.spinner(
    f"Scanning {len(symbols)} NSE stocks..."
):
    current = current_scan(
        symbols,
        minimum_score
    )

if not current:
    st.warning(
        "No precision candidates found. "
        "Lower the technical score slightly."
    )
    st.stop()

st.success(
    f"Precision scan complete — "
    f"{len(current)} candidates."
)

# -----------------------------
# Batch used for sector/breadth
# -----------------------------

batch = download_batch(
    symbols,
    "2y"
)

# -----------------------------
# Market context
# -----------------------------

mkt = market_context()

b50, b200 = breadth_score(
    current,
    symbols,
    batch
)

if not pd.isna(b50):
    if b50 >= 65 and mkt["score"] >= 70:
        market_score = 90
    elif b50 >= 50:
        market_score = 70
    elif b50 >= 35:
        market_score = 50
    else:
        market_score = 30
else:
    market_score = mkt["score"]

# -----------------------------
# Sector context
# -----------------------------

sector_map, sector_table = sector_strength(
    universe_df,
    symbols,
    batch
)

# -----------------------------
# Current candidate table
# -----------------------------

rows = []

for item in current:
    d = item["Details"]
    symbol = item["Symbol"]

    industry = "Other"
    if "Industry" in universe_df.columns:
        hit = universe_df.loc[
            universe_df["Symbol"] == symbol,
            "Industry"
        ]
        if not hit.empty:
            industry = str(hit.iloc[0])

    sec = float(
        sector_map.get(industry, 50)
    )

    rows.append({
        "Symbol": symbol,
        "Stage": item["Stage"],
        "Tech": item["Tech"],
        "Close": round(d["close"], 2),
        "Breakout": round(d["resistance"], 2),
        "Distance %": round(d["distance"], 2),
        "RSI": round(d["rsi"], 1),
        "Vol": round(d["vol"], 2),
        "ATR": round(d["atr_ratio"], 2),
        "VCP": d["vcp"],
        "Sector": industry,
        "Sector Strength": round(sec, 1),
    })

current_table = pd.DataFrame(rows)

st.subheader("🔥 Current Pre-Breakout Candidates")
st.dataframe(
    current_table,
    use_container_width=True,
    hide_index=True
)

# -----------------------------
# Market dashboard
# -----------------------------

st.subheader("🌐 Market Regime")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "NIFTY Regime",
    mkt["regime"]
)

c2.metric(
    "NIFTY 60D",
    (
        f"{mkt['return60']:.1f}%"
        if not pd.isna(mkt["return60"])
        else "—"
    )
)

c3.metric(
    "Stocks > 50 SMA",
    (
        f"{b50:.1f}%"
        if not pd.isna(b50)
        else "—"
    )
)

c4.metric(
    "Stocks > 150 SMA",
    (
        f"{b200:.1f}%"
        if not pd.isna(b200)
        else "—"
    )
)

# -----------------------------
# Sector dashboard
# -----------------------------

if not sector_table.empty:
    st.subheader("🏭 Sector Strength")
    st.dataframe(
        sector_table.head(10).round(2),
        use_container_width=True,
        hide_index=True
    )

# -----------------------------
# Historical validation
# -----------------------------

st.subheader(
    "🧪 Historical Breakout Evidence"
)

validation = []
progress = st.progress(0)

# Every current candidate is validated.
for idx, item in enumerate(current):
    symbol = item["Symbol"]

    hist = download_single(
        symbol,
        "5y"
    )

    if hist.empty or len(hist) < 400:
        progress.progress(
            (idx + 1) / len(current)
        )
        continue

    hist = add_indicators(hist)

    stats = historical_replay(
        hist,
        breakout_window=breakout_window,
        holding_period=holding_period,
        target_r=target_r,
        cooldown=cooldown,
    )

    if stats is None:
        progress.progress(
            (idx + 1) / len(current)
        )
        continue

    ev = evidence_score(stats)
    grade = evidence_grade(
        stats["entries"]
    )

    industry = "Other"
    if "Industry" in universe_df.columns:
        hit = universe_df.loc[
            universe_df["Symbol"] == symbol,
            "Industry"
        ]
        if not hit.empty:
            industry = str(hit.iloc[0])

    sec = float(
        sector_map.get(industry, 50)
    )

    # Final opportunity score:
    # Technical 45 + historical evidence 35 +
    # sector 10 + market context 10.
    # Market/sector are context, not trade guarantees.
    final = (
        item["Tech"] * 0.45
        + ev * 0.35
        + sec * 0.10
        + market_score * 0.10
    )

    # Additional evidence penalty.
    # Low sample can remain visible, but cannot dominate.
    n = stats["entries"]
    if n < 5:
        sample_factor = 0.60
    elif n < 10:
        sample_factor = 0.78
    elif n < 20:
        sample_factor = 0.90
    else:
        sample_factor = 1.00

    final *= sample_factor

    validation.append({
        "Symbol": symbol,
        "Stage": item["Stage"],
        "Tech": item["Tech"],
        "Evidence": ev,
        "Sector": round(sec, 1),
        "Market": market_score,
        "Setups": stats["setups"],
        "Breakouts": stats["breakouts"],
        "Breakout %": stats["breakout_rate"],
        "Entries": stats["entries"],
        "Targets": stats["targets"],
        "Stops": stats["stops"],
        "Win %": stats["win_rate"],
        "PF": stats["profit_factor"],
        "Expectancy R": stats["expectancy"],
        "Max DD R": stats["max_dd"],
        "Evidence Grade": grade,
        "Opportunity": round(final, 1),
    })

    progress.progress(
        (idx + 1) / len(current)
    )

progress.empty()

if not validation:
    st.error(
        "Historical validation could not be completed."
    )
    st.stop()

vdf = pd.DataFrame(validation)

# Sort by opportunity, then evidence, then sample.
vdf = vdf.sort_values(
    ["Opportunity", "Evidence", "Entries"],
    ascending=False
)

display_df = vdf.copy()

for c in [
    "Breakout %",
    "Win %",
    "PF",
    "Expectancy R",
    "Max DD R"
]:
    display_df[c] = (
        display_df[c]
        .replace([np.inf, -np.inf], np.nan)
        .round(2)
    )

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)

# -----------------------------
# Best candidate
# -----------------------------

best = vdf.iloc[0]
best_symbol = best["Symbol"]

best_item = next(
    x for x in current
    if x["Symbol"] == best_symbol
)

d = best_item["Details"]

st.markdown("---")
st.subheader(
    f"🏆 #1 Opportunity — {best_symbol}"
)

a, b, c, dcol, e = st.columns(5)

a.metric(
    "Opportunity",
    f"{best['Opportunity']:.1f}/100"
)

b.metric(
    "Technical",
    f"{best['Tech']}/100"
)

c.metric(
    "Evidence",
    f"{best['Evidence']}/100"
)

dcol.metric(
    "Entries",
    int(best["Entries"])
)

pf = best["PF"]
pf_text = (
    "∞"
    if np.isinf(pf)
    else f"{pf:.2f}"
)

e.metric(
    "Profit Factor",
    pf_text
)

st.write(
    f"### Evidence: {best['Evidence Grade']}"
)

st.write(
    f"**Historical setups:** {int(best['Setups'])} | "
    f"**Breakouts:** {int(best['Breakouts'])} | "
    f"**Entries:** {int(best['Entries'])} | "
    f"**Targets:** {int(best['Targets'])} | "
    f"**Stops:** {int(best['Stops'])}"
)

st.write(
    f"**Win rate:** {best['Win %']:.1f}% | "
    f"**Expectancy:** {best['Expectancy R']:.2f}R | "
    f"**Max drawdown:** {best['Max DD R']:.2f}R"
)

# -----------------------------
# Current setup
# -----------------------------

st.subheader(
    f"📍 Current Setup — {best_symbol}"
)

trigger = d["resistance"]
atr = num(
    best_item["Data"].iloc[-1]["ATR14"]
)
ema = num(
    best_item["Data"].iloc[-1]["EMA220"]
)

# This is an indicative risk reference.
# Actual backtest entries use next-day OPEN after confirmation.
proxy_stop_candidates = [
    trigger - 2 * atr
    if not pd.isna(atr) else np.nan,
    ema * 0.995
    if not pd.isna(ema) else np.nan,
]

proxy_stops = [
    x for x in proxy_stop_candidates
    if not pd.isna(x)
    and 0 < x < trigger
]

proxy_stop = (
    max(proxy_stops)
    if proxy_stops
    else np.nan
)

risk = (
    trigger - proxy_stop
    if not pd.isna(proxy_stop)
    else np.nan
)

p1 = (
    trigger + risk
    if not pd.isna(risk)
    else np.nan
)

p15 = (
    trigger + 1.5 * risk
    if not pd.isna(risk)
    else np.nan
)

p2 = (
    trigger + 2 * risk
    if not pd.isna(risk)
    else np.nan
)

a, b, c, dcol = st.columns(4)

a.metric(
    "Current",
    f"₹{d['close']:,.2f}"
)

b.metric(
    "Breakout Trigger",
    f"₹{trigger:,.2f}"
)

c.metric(
    "Indicative SL",
    (
        f"₹{proxy_stop:,.2f}"
        if not pd.isna(proxy_stop)
        else "—"
    )
)

dcol.metric(
    "Distance",
    f"{d['distance']:.2f}%"
)

if not pd.isna(risk):
    st.write(
        f"**Indicative 1R:** ₹{p1:,.2f} | "
        f"**1.5R:** ₹{p15:,.2f} | "
        f"**2R:** ₹{p2:,.2f}"
    )

st.warning(
    "Execution rule: do not treat the current breakout trigger "
    "as an entry. The tested sequence is closing breakout "
    "confirmation followed by next-day OPEN. "
    "The current SL/targets are indicative until that entry exists."
)

# -----------------------------
# Top 5
# -----------------------------

st.markdown("---")
st.subheader("🥇 Top Opportunities")

for rank, (_, row) in enumerate(
    vdf.head(5).iterrows(),
    start=1
):
    st.write(
        f"**{rank}. {row['Symbol']}** — "
        f"Opportunity {row['Opportunity']:.1f} | "
        f"Tech {row['Tech']} | "
        f"Evidence {row['Evidence']} | "
        f"Entries {int(row['Entries'])} | "
        f"{row['Evidence Grade']}"
    )

# -----------------------------
# Methodology
# -----------------------------

with st.expander("🔎 V5 methodology"):
    st.markdown(
        """ ### Current setup The scanner requires the stock to remain genuinely below the previous 20-session resistance. Core conditions include: - 150 SMA > 220 EMA - Close > 50 SMA - 50 SMA > 150 SMA - >25% above previous 52-week low - EMA220 dip during the previous 90 sessions - 0–5% below resistance - 5/10/20 range contraction - volume contraction - ATR compression - higher-low structure - healthy RSI - VCP-style contraction ### Historical replay For every current candidate: **Historical setup** → confirmed closing breakout → next trading day OPEN → ATR/EMA stop → target → time exit The same-day target + stop case is conservatively treated as a stop. ### Evidence The historical score uses a conservative win-rate estimate, profit factor, expectancy, breakout rate and sample-size penalty. A tiny sample is deliberately prevented from dominating the ranking. ### Market context NIFTY trend and breadth are displayed as context. ### Sector context Sector strength is calculated from the current NIFTY universe's industry grouping and recent stock performance. Sector and market context influence ranking modestly; they do not override the technical setup or historical evidence. """
    )

st.caption(
    "StockPilot V5 | "
    + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
)
