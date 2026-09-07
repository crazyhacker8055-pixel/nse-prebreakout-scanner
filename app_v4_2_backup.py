import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO
from datetime import datetime

# ============================================================
# STOCKPILOT V4.2
# PRECISION PRE-BREAKOUT + HISTORICAL VALIDATION ENGINE
# ============================================================

st.set_page_config(
    page_title="StockPilot V4.2",
    page_icon="🚀",
    layout="wide"
)

NIFTY500_URL = (
    "https://archives.nseindia.com/content/indices/"
    "ind_nifty500list.csv"
)


# ============================================================
# HELPERS
# ============================================================

def sf(x, default=np.nan):
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

    df.columns = [
        str(c).strip().title()
        for c in df.columns
    ]

    return df


# ============================================================
# NIFTY 500
# ============================================================

@st.cache_data(ttl=1800, show_spinner=False)
def load_nifty500():

    try:

        r = requests.get(
            NIFTY500_URL,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "text/csv,*/*"
            },
            timeout=20
        )

        r.raise_for_status()

        df = pd.read_csv(
            StringIO(r.text)
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

    except Exception as e:

        st.error(
            f"NIFTY 500 loading failed: {e}"
        )

        return []


# ============================================================
# DATA
# ============================================================

@st.cache_data(ttl=1800, show_spinner=False)
def download_data(symbols, period="2y"):

    tickers = [
        f"{s}.NS"
        for s in symbols
    ]

    try:

        return yf.download(
            tickers=tickers,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=True
        )

    except Exception:

        return pd.DataFrame()


def extract_symbol(data, symbol):

    ticker = f"{symbol}.NS"

    if data is None or data.empty:
        return pd.DataFrame()

    try:

        if isinstance(
            data.columns,
            pd.MultiIndex
        ):

            l0 = data.columns.get_level_values(0)
            l1 = data.columns.get_level_values(1)

            if ticker in l0:
                df = data[ticker].copy()

            elif symbol in l0:
                df = data[symbol].copy()

            elif ticker in l1:
                df = data.xs(
                    ticker,
                    axis=1,
                    level=1
                ).copy()

            else:
                return pd.DataFrame()

        else:

            df = data.copy()

        df = clean_columns(df)

        required = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]

        if not all(
            c in df.columns
            for c in required
        ):
            return pd.DataFrame()

        return (
            df[required]
            .dropna()
            .copy()
        )

    except Exception:

        return pd.DataFrame()


# ============================================================
# INDICATORS
# ============================================================

def indicators(df):

    x = df.copy()

    # Moving averages
    x["SMA20"] = x["Close"].rolling(20).mean()
    x["SMA50"] = x["Close"].rolling(50).mean()
    x["SMA150"] = x["Close"].rolling(150).mean()
    x["EMA220"] = x["Close"].ewm(
        span=220,
        adjust=False
    ).mean()

    # ATR
    pc = x["Close"].shift(1)

    tr = pd.concat(
        [
            x["High"] - x["Low"],
            (x["High"] - pc).abs(),
            (x["Low"] - pc).abs()
        ],
        axis=1
    ).max(axis=1)

    x["TR"] = tr
    x["ATR14"] = tr.rolling(14).mean()
    x["ATR20"] = x["ATR14"].rolling(20).mean()
    x["ATRRatio"] = (
        x["ATR14"] /
        x["ATR20"]
    )

    # RSI
    delta = x["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    ag = gain.rolling(14).mean()
    al = loss.rolling(14).mean()

    rs = ag / al.replace(0, np.nan)

    x["RSI"] = (
        100 -
        100 / (1 + rs)
    )

    # Volume
    x["Vol20"] = (
        x["Volume"]
        .rolling(20)
        .mean()
    )

    x["VolRatio"] = (
        x["Volume"] /
        x["Vol20"]
    )

    # Ranges
    for n in [5, 10, 20]:

        x[f"Range{n}"] = (
            (
                x["High"].rolling(n).max()
                -
                x["Low"].rolling(n).min()
            )
            /
            x["Close"]
            *
            100
        )

    # Previous resistance
    x["Resistance20"] = (
        x["High"]
        .shift(1)
        .rolling(20)
        .max()
    )

    # Previous 52W levels
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

    # EMA220 dip within previous 90 sessions
    dipped = (
        x["Low"] <
        x["EMA220"]
    )

    x["Dip90"] = (
        dipped
        .rolling(90)
        .max()
        .fillna(0)
        .astype(bool)
    )

    # Higher-low structure
    x["Low10"] = (
        x["Low"]
        .rolling(10)
        .min()
    )

    x["PrevLow20"] = (
        x["Low"]
        .shift(10)
        .rolling(20)
        .min()
    )

    return x


# ============================================================
# VCP / TECHNICAL SCORE
# ============================================================

def calculate_score(df):

    if len(df) < 260:
        return 0, {}

    r = df.iloc[-1]

    close = sf(r["Close"])
    sma50 = sf(r["SMA50"])
    sma150 = sf(r["SMA150"])
    ema220 = sf(r["EMA220"])

    resistance = sf(r["Resistance20"])
    high52 = sf(r["High52"])
    low52 = sf(r["Low52"])

    range5 = sf(r["Range5"])
    range10 = sf(r["Range10"])
    range20 = sf(r["Range20"])

    vol = sf(r["VolRatio"])
    atr_ratio = sf(r["ATRRatio"])
    rsi = sf(r["RSI"])

    score = 0

    # --------------------------------------------------------
    # TREND = 30
    # --------------------------------------------------------

    trend = 0

    if sma150 > ema220:
        score += 10
        trend += 1

    if close > sma50:
        score += 10
        trend += 1

    if sma50 > sma150:
        score += 10
        trend += 1

    # --------------------------------------------------------
    # LOCATION = 15
    # --------------------------------------------------------

    if (
        low52 > 0
        and close > 1.25 * low52
    ):
        score += 5

    dip = bool(r["Dip90"])

    if dip:
        score += 5

    high_distance = np.nan

    if high52 > 0:

        high_distance = (
            high52 - close
        ) / close * 100

        if high_distance <= 5:
            score += 5

        elif high_distance <= 10:
            score += 3

    # --------------------------------------------------------
    # BREAKOUT PROXIMITY = 15
    # --------------------------------------------------------

    distance = np.nan

    if resistance > 0:

        distance = (
            resistance - close
        ) / close * 100

        if 0 <= distance <= 2:
            score += 15

        elif 2 < distance <= 5:
            score += 10

        elif 5 < distance <= 8:
            score += 5

    # Already broken = zero breakout points
    if distance < 0:
        distance = -abs(distance)

    # --------------------------------------------------------
    # VCP / RANGE CONTRACTION = 15
    # --------------------------------------------------------

    vcp = 0

    if (
        not pd.isna(range5)
        and
        not pd.isna(range10)
        and
        not pd.isna(range20)
    ):

        if range5 < range10:
            vcp += 1
            score += 5

        if range10 < range20:
            vcp += 1
            score += 5

        if range20 <= 20:
            vcp += 1
            score += 5

    # --------------------------------------------------------
    # VOLUME DRY-UP = 10
    # --------------------------------------------------------

    volume_score = 0

    if not pd.isna(vol):

        if vol <= 0.65:
            score += 10
            volume_score = 10

        elif vol <= 0.80:
            score += 7
            volume_score = 7

        elif vol <= 1.00:
            score += 3
            volume_score = 3

    # --------------------------------------------------------
    # ATR COMPRESSION = 5
    # --------------------------------------------------------

    if not pd.isna(atr_ratio):

        if atr_ratio <= 0.80:
            score += 5

        elif atr_ratio <= 0.95:
            score += 3

    # --------------------------------------------------------
    # HIGHER LOW = 5
    # --------------------------------------------------------

    higher_low = False

    if (
        not pd.isna(r["Low10"])
        and
        not pd.isna(r["PrevLow20"])
    ):

        if (
            r["Low10"] >
            r["PrevLow20"]
        ):

            score += 5
            higher_low = True

    # --------------------------------------------------------
    # RSI = 5
    # --------------------------------------------------------

    if (
        not pd.isna(rsi)
        and
        45 <= rsi <= 68
    ):
        score += 5

    score = min(
        int(score),
        100
    )

    return score, {

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
        "volume_score": volume_score,
        "higher_low": higher_low,
        "trend": trend
    }


# ============================================================
# STRICT CURRENT FILTER
# ============================================================

def qualifies_current(
    score,
    d,
    strict=True
):

    distance = d["distance"]

    # Must still be below resistance
    if (
        pd.isna(distance)
        or distance < 0
    ):
        return False

    # Strict proximity
    max_distance = (
        5
        if strict
        else 8
    )

    if distance > max_distance:
        return False

    # Minimum score
    minimum = (
        78
        if strict
        else 65
    )

    if score < minimum:
        return False

    # Trend
    if d["trend"] < 3:
        return False

    # Compression
    if (
        d["range5"] >=
        d["range10"]
    ):
        return False

    if (
        d["range10"] >=
        d["range20"]
    ):
        return False

    # ATR
    if (
        not pd.isna(
            d["atr_ratio"]
        )
        and
        d["atr_ratio"] > 1.05
    ):
        return False

    # Volume
    if (
        not pd.isna(d["vol"])
        and
        d["vol"] > 1.05
    ):
        return False

    # Higher low
    if strict and not d["higher_low"]:
        return False

    return True


# ============================================================
# HISTORICAL SETUP
# ============================================================

def historical_setup(
    row,
    minimum_score,
    max_distance
):

    required = [
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
        "ATRRatio",
        "Low10",
        "PrevLow20"
    ]

    for c in required:

        if (
            c not in row
            or
            pd.isna(row[c])
        ):
            return False

    close = float(
        row["Close"]
    )

    resistance = float(
        row["Resistance20"]
    )

    if (
        close <= 0
        or resistance <= 0
    ):
        return False

    distance = (
        resistance -
        close
    ) / close * 100

    # Critical: still PRE-breakout
    if (
        distance < 0
        or
        distance > max_distance
    ):
        return False

    # Core trend
    if not (
        row["SMA150"] >
        row["EMA220"]
        and
        close >
        row["SMA50"]
        and
        row["SMA50"] >
        row["SMA150"]
    ):
        return False

    # 52W low
    if (
        row["Low52"] <= 0
        or
        close <=
        1.25 *
        row["Low52"]
    ):
        return False

    # EMA220 dip
    if not bool(
        row["Dip90"]
    ):
        return False

    # Contraction
    if not (
        row["Range5"] <
        row["Range10"] <
        row["Range20"]
    ):
        return False

    # ATR compression
    if (
        row["ATRRatio"] >
        1.05
    ):
        return False

    # Volume
    if (
        row["VolRatio"] >
        1.05
    ):
        return False

    # Higher low
    if not (
        row["Low10"] >
        row["PrevLow20"]
    ):
        return False

    # Calculate historical score
    s = 0

    s += 10
    s += 10
    s += 10
    s += 5
    s += 5

    if distance <= 2:
        s += 15

    elif distance <= 5:
        s += 10

    else:
        s += 5

    s += 5
    s += 5
    s += 5

    if row["VolRatio"] <= 0.65:
        s += 10

    elif row["VolRatio"] <= 0.80:
        s += 7

    elif row["VolRatio"] <= 1.0:
        s += 3

    if row["ATRRatio"] <= 0.80:
        s += 5

    elif row["ATRRatio"] <= 0.95:
        s += 3

    s += 5

    if (
        45 <=
        row["RSI"] <= 68
    ):
        s += 5

    return s >= minimum_score


# ============================================================
# BACKTEST
# ============================================================

def backtest(
    df,
    setup_score=65,
    max_distance=5,
    breakout_window=5,
    holding_period=20,
    target_r=1.5,
    cooldown=20
):

    if (
        df.empty
        or len(df) < 400
    ):
        return None

    x = df.copy()

    setups = 0
    breakouts = 0
    entries = 0
    targets = 0
    stops = 0
    time_exits = 0
    no_breakouts = 0

    trades = []

    i = 260

    last_event = -9999

    while (
        i <
        len(x)
        -
        breakout_window
        -
        2
    ):

        if (
            i -
            last_event
            < cooldown
        ):

            i += 1
            continue

        row = x.iloc[i]

        if not historical_setup(
            row,
            setup_score,
            max_distance
        ):

            i += 1
            continue

        setups += 1

        resistance = sf(
            row["Resistance20"]
        )

        breakout_idx = None

        end = min(
            i + breakout_window,
            len(x) - 2
        )

        # ----------------------------------------------------
        # ACTUAL BREAKOUT
        # ----------------------------------------------------

        for j in range(
            i + 1,
            end + 1
        ):

            if (
                sf(x.iloc[j]["Close"])
                >
                resistance
            ):

                breakout_idx = j
                break

        if breakout_idx is None:

            no_breakouts += 1

            last_event = i

            i += cooldown

            continue

        breakouts += 1

        # ----------------------------------------------------
        # NEXT DAY OPEN
        # ----------------------------------------------------

        entry_idx = (
            breakout_idx + 1
        )

        if entry_idx >= len(x):
            break

        entry = sf(
            x.iloc[entry_idx]["Open"]
        )

        if (
            pd.isna(entry)
            or
            entry <= 0
        ):

            i += cooldown
            continue

        entries += 1

        atr = sf(
            x.iloc[breakout_idx]["ATR14"]
        )

        ema = sf(
            x.iloc[breakout_idx]["EMA220"]
        )

        if (
            pd.isna(atr)
            or
            atr <= 0
        ):

            i += cooldown
            continue

        atr_stop = (
            entry -
            2 * atr
        )

        ema_stop = np.nan

        if not pd.isna(ema):

            ema_stop = (
                ema *
                0.995
            )

        stops_available = [
            s
            for s in [
                atr_stop,
                ema_stop
            ]
            if (
                not pd.isna(s)
                and
                0 < s < entry
            )
        ]

        if stops_available:

            stop = max(
                stops_available
            )

        else:

            stop = atr_stop

        risk = (
            entry -
            stop
        )

        if risk <= 0:

            i += cooldown
            continue

        target = (
            entry +
            target_r *
            risk
        )

        outcome = "TIME EXIT"

        exit_price = np.nan
        exit_idx = None

        last = min(
            entry_idx +
            holding_period,
            len(x) - 1
        )

        for k in range(
            entry_idx,
            last + 1
        ):

            high = sf(
                x.iloc[k]["High"]
            )

            low = sf(
                x.iloc[k]["Low"]
            )

            close = sf(
                x.iloc[k]["Close"]
            )

            hit_target = (
                high >= target
            )

            hit_stop = (
                low <= stop
            )

            # Conservative
            if (
                hit_target
                and
                hit_stop
            ):

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

        if (
            exit_idx is None
            or
            pd.isna(exit_price)
        ):

            i += cooldown
            continue

        r = (
            exit_price -
            entry
        ) / risk

        ret = (
            exit_price /
            entry -
            1
        ) * 100

        if outcome == "TARGET":
            targets += 1

        elif outcome == "STOP":
            stops += 1

        else:
            time_exits += 1

        trades.append({
            "Setup": x.index[i],
            "Breakout":
                x.index[breakout_idx],
            "Entry":
                x.index[entry_idx],
            "Exit":
                x.index[exit_idx],
            "R": r,
            "Return": ret,
            "Outcome": outcome
        })

        last_event = exit_idx

        i = (
            exit_idx +
            1
        )

    tdf = pd.DataFrame(
        trades
    )

    if tdf.empty:

        return {
            "setups": setups,
            "breakouts": breakouts,
            "breakout_rate":
                (
                    breakouts /
                    setups *
                    100
                    if setups else 0
                ),
            "entries": 0,
            "targets": 0,
            "stops": 0,
            "time_exits": 0,
            "win_rate": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "profit_factor": 0,
            "expectancy": 0,
            "max_dd": 0,
            "trades": tdf
        }

    winners = tdf[
        tdf["R"] > 0
    ]["R"]

    losers = tdf[
        tdf["R"] < 0
    ]["R"]

    win_rate = (
        len(winners) /
        len(tdf) *
        100
    )

    avg_win = (
        winners.mean()
        if len(winners)
        else 0
    )

    avg_loss = (
        losers.mean()
        if len(losers)
        else 0
    )

    gross_profit = (
        winners.sum()
        if len(winners)
        else 0
    )

    gross_loss = (
        abs(losers.sum())
        if len(losers)
        else 0
    )

    if gross_loss > 0:

        pf = (
            gross_profit /
            gross_loss
        )

    else:

        pf = np.inf

    expectancy = (
        tdf["R"].mean()
    )

    equity = (
        tdf["R"]
        .cumsum()
    )

    dd = (
        equity -
        equity.cummax()
    )

    max_dd = dd.min()

    return {
        "setups": setups,
        "breakouts": breakouts,
        "breakout_rate":
            (
                breakouts /
                setups *
                100
                if setups else 0
            ),
        "entries": len(tdf),
        "targets": targets,
        "stops": stops,
        "time_exits": time_exits,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": pf,
        "expectancy": expectancy,
        "max_dd": max_dd,
        "trades": tdf
    }


# ============================================================
# HISTORICAL VALIDATION SCORE
# ============================================================

def validation_score(stats):

    n = stats["entries"]

    if n == 0:
        return 0

    pf = stats["profit_factor"]

    if not np.isfinite(pf):
        pf_component = 20
    else:
        pf_component = min(
            max(pf, 0) / 2 * 20,
            20
        )

    win_component = min(
        stats["win_rate"] / 100 * 20,
        20
    )

    breakout_component = min(
        stats["breakout_rate"] /
        100 *
        15,
        15
    )

    expectancy_component = min(
        max(
            stats["expectancy"],
            0
        ) /
        1.5 *
        20,
        20
    )

    # Sample quality
    if n >= 30:
        sample_factor = 1.00

    elif n >= 20:
        sample_factor = 0.90

    elif n >= 10:
        sample_factor = 0.75

    elif n >= 5:
        sample_factor = 0.50

    else:
        sample_factor = 0.25

    raw = (
        pf_component +
        win_component +
        breakout_component +
        expectancy_component
    )

    score = (
        raw *
        sample_factor
    )

    return round(
        min(score, 100),
        1
    )


def validation_label(
    stats,
    score
):

    n = stats["entries"]
    pf = stats["profit_factor"]
    exp = stats["expectancy"]

    if n < 5:
        return "⚠️ VERY LOW SAMPLE"

    if n < 10:
        return "⚠️ LOW SAMPLE"

    if (
        n >= 30
        and
        pf > 1.5
        and
        exp > 0
    ):
        return "🟢 STRONG"

    if (
        n >= 10
        and
        pf > 1
        and
        exp > 0
    ):
        return "🟢 POSITIVE"

    if (
        pf > 0.8
    ):
        return "🟠 NEUTRAL"

    return "🔴 WEAK"


# ============================================================
# CURRENT SCANNER
# ============================================================

def scan(
    symbols,
    minimum_score,
    strict
):

    data = download_data(
        symbols,
        "2y"
    )

    results = []

    for symbol in symbols:

        df = extract_symbol(
            data,
            symbol
        )

        if (
            df.empty
            or
            len(df) < 260
        ):
            continue

        df = indicators(
            df
        )

        score, d = (
            calculate_score(df)
        )

        if not qualifies_current(
            score,
            d,
            strict
        ):
            continue

        # User selected minimum
        if score < minimum_score:
            continue

        results.append({
            "Symbol": symbol,
            "Score": score,
            "Data": df,
            "Details": d
        })

    results.sort(
        key=lambda x:
        x["Score"],
        reverse=True
    )

    return results


# ============================================================
# UI
# ============================================================

st.title(
    "🚀 StockPilot V4.2"
)

st.caption(
    "Precision pre-breakout scanner + evidence-based historical validation"
)

st.info(
    "V4.2 is a research engine. "
    "Historical performance is not a guarantee of future results."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "⚙️ Scanner Settings"
)

universe = (
    st.sidebar.selectbox(
        "Universe",
        [
            "NIFTY 500",
            "NIFTY 250",
            "NIFTY 100"
        ]
    )
)

minimum_score = (
    st.sidebar.slider(
        "Minimum Technical Score",
        60,
        95,
        78,
        1
    )
)

strict = (
    st.sidebar.checkbox(
        "🔥 Precision / Strict Mode",
        True
    )
)

st.sidebar.header(
    "🧪 Historical Validation"
)

history_years = (
    st.sidebar.selectbox(
        "Historical Period",
        [1, 2, 3],
        index=1
    )
)

holding_period = (
    st.sidebar.selectbox(
        "Maximum Holding Period",
        [10, 15, 20, 30],
        index=2
    )
)

breakout_window = (
    st.sidebar.selectbox(
        "Breakout Window",
        [3, 5, 7],
        index=1
    )
)

target_r = (
    st.sidebar.selectbox(
        "Target R",
        [1.0, 1.5, 2.0],
        index=1
    )
)

cooldown = (
    st.sidebar.selectbox(
        "Signal Cooldown",
        [10, 15, 20, 30],
        index=2
    )
)

historical_score = (
    st.sidebar.slider(
        "Historical Setup Score",
        60,
        90,
        70,
        1
    )
)

run = (
    st.sidebar.button(
        "🚀 RUN V4.2",
        type="primary",
        use_container_width=True
    )
)


if not run:

    st.markdown(
        """
# 🎯 V4.2 Precision Engine

The scanner now follows:

**NIFTY 500**

↓

**Trend alignment**

↓

**EMA220 recovery / dip history**

↓

**VCP-style contraction**

↓

**Volume dry-up**

↓

**ATR compression**

↓

**Higher lows**

↓

**Close just below resistance**

↓

**Historical breakout evidence**

↓

# ⭐ Final Opportunity Score

The historical ranking includes a **sample-size penalty**, so one lucky trade cannot automatically become the #1 stock.
        """
    )

    st.stop()


# ============================================================
# UNIVERSE
# ============================================================

symbols = load_nifty500()

if not symbols:
    st.stop()

if universe == "NIFTY 250":
    symbols = symbols[:250]

elif universe == "NIFTY 100":
    symbols = symbols[:100]


# ============================================================
# CURRENT SCAN
# ============================================================

with st.spinner(
    f"Scanning {len(symbols)} stocks..."
):

    current = scan(
        symbols,
        minimum_score,
        strict
    )


st.success(
    f"Precision scan complete — "
    f"{len(current)} qualified candidates."
)


if not current:

    st.warning(
        "No precision candidates found. "
        "Try lowering Minimum Technical Score "
        "or turn Precision Mode OFF."
    )

    st.stop()


# ============================================================
# TOP CURRENT CANDIDATES
# ============================================================

st.subheader(
    "🔥 Top Pre-Breakout Candidates"
)

rows = []

for item in current[:15]:

    d = item["Details"]

    rows.append({

        "Symbol":
            item["Symbol"],

        "Tech Score":
            item["Score"],

        "Close":
            round(d["close"], 2),

        "Breakout":
            round(
                d["resistance"],
                2
            ),

        "Distance %":
            round(
                d["distance"],
                2
            ),

        "RSI":
            round(
                d["rsi"],
                1
            ),

        "Vol":
            round(
                d["vol"],
                2
            ),

        "ATR":
            round(
                d["atr_ratio"],
                2
            ),

        "VCP":
            f"{d['range5']:.1f}/"
            f"{d['range10']:.1f}/"
            f"{d['range20']:.1f}"
    })


st.dataframe(
    pd.DataFrame(rows),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# HISTORICAL VALIDATION
# ============================================================

st.subheader(
    "🧪 Historical Breakout Validation"
)

# Validate only top 8
# This keeps free Community Cloud reasonably fast.

candidates = current[:8]

validation = []

progress = st.progress(0)

history_period = (
    f"{history_years + 1}y"
)

for number, item in enumerate(
    candidates
):

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

        hist = clean_columns(
            hist
        )

        required = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]

        if (
            hist.empty
            or
            not all(
                c in hist.columns
                for c in required
            )
        ):
            continue

        hist = (
            hist[required]
            .dropna()
        )

        hist = indicators(
            hist
        )

        stats = backtest(

            hist,

            setup_score=
                historical_score,

            max_distance=
                5 if strict else 8,

            breakout_window=
                breakout_window,

            holding_period=
                holding_period,

            target_r=
                target_r,

            cooldown=
                cooldown
        )

        if stats is None:
            continue

        vscore = (
            validation_score(
                stats
            )
        )

        label = (
            validation_label(
                stats,
                vscore
            )
        )

        validation.append({

            "Symbol":
                symbol,

            "Tech":
                item["Score"],

            "Hist Score":
                vscore,

            "Setups":
                stats["setups"],

            "Breakouts":
                stats["breakouts"],

            "Breakout %":
                stats[
                    "breakout_rate"
                ],

            "Entries":
                stats["entries"],

            "Targets":
                stats["targets"],

            "Stops":
                stats["stops"],

            "Win %":
                stats["win_rate"],

            "PF":
                stats["profit_factor"],

            "Expectancy R":
                stats["expectancy"],

            "Max DD R":
                stats["max_dd"],

            "Validation":
                label
        })

    except Exception:

        pass

    progress.progress(
        (number + 1) /
        len(candidates)
    )

progress.empty()


if not validation:

    st.error(
        "No historical validation results."
    )

    st.stop()


vdf = pd.DataFrame(
    validation
)


# ============================================================
# FINAL OPPORTUNITY SCORE
# ============================================================

def final_score(row):

    tech = float(
        row["Tech"]
    )

    hist = float(
        row["Hist Score"]
    )

    entries = int(
        row["Entries"]
    )

    # Technical 55%
    # Historical 45%
    base = (
        tech * 0.55
        +
        hist * 0.45
    )

    # Additional sample penalty
    if entries >= 30:
        factor = 1.00

    elif entries >= 20:
        factor = 0.97

    elif entries >= 10:
        factor = 0.92

    elif entries >= 5:
        factor = 0.80

    else:
        factor = 0.60

    return round(
        base * factor,
        1
    )


vdf["Opportunity"] = (
    vdf.apply(
        final_score,
        axis=1
    )
)


# Sort by final score
vdf = (
    vdf
    .sort_values(
        [
            "Opportunity",
            "Hist Score",
            "Entries"
        ],
        ascending=False
    )
)


# Round display
for c in [
    "Breakout %",
    "Win %",
    "PF",
    "Expectancy R",
    "Max DD R"
]:

    vdf[c] = (
        vdf[c]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .round(2)
    )


st.dataframe(
    vdf,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# BEST CANDIDATE
# ============================================================

best = vdf.iloc[0]

best_symbol = (
    best["Symbol"]
)

best_item = next(
    x for x in current
    if x["Symbol"] ==
    best_symbol
)

best_details = (
    best_item["Details"]
)


st.markdown("---")

st.subheader(
    f"🏆 #1 Precision Candidate — "
    f"{best_symbol}"
)


a, b, c, d, e = (
    st.columns(5)
)

a.metric(
    "Opportunity",
    f"{best['Opportunity']}/100"
)

b.metric(
    "Technical",
    f"{best['Tech']}/100"
)

c.metric(
    "Historical",
    f"{best['Hist Score']}/100"
)

d.metric(
    "Win Rate",
    f"{best['Win %']:.1f}%"
)

pf = best["PF"]

if pd.isna(pf):
    pf_text = "—"

elif pf >= 999:
    pf_text = "∞"

else:
    pf_text = f"{pf:.2f}"

e.metric(
    "Profit Factor",
    pf_text
)


st.write(
    f"### Validation: "
    f"{best['Validation']}"
)

st.write(
    f"**Historical setups:** "
    f"{int(best['Setups'])}"
    f" | "
    f"**Breakouts:** "
    f"{int(best['Breakouts'])}"
    f" | "
    f"**Entries:** "
    f"{int(best['Entries'])}"
)

st.write(
    f"**Targets:** "
    f"{int(best['Targets'])}"
    f" | "
    f"**Stops:** "
    f"{int(best['Stops'])}"
    f" | "
    f"**Expectancy:** "
    f"{best['Expectancy R']:.2f}R"
    f" | "
    f"**Max DD:** "
    f"{best['Max DD R']:.2f}R"
)


# ============================================================
# CURRENT TRADE ZONE
# ============================================================

st.subheader(
    f"📍 Current Setup — "
    f"{best_symbol}"
)

close = best_details[
    "close"
]

breakout = best_details[
    "resistance"
]

atr = sf(
    best_item["Data"]
    .iloc[-1]["ATR14"]
)

ema = sf(
    best_item["Data"]
    .iloc[-1]["EMA220"]
)

atr_stop = (
    breakout -
    2 * atr
)

ema_stop = (
    ema * 0.995
)

possible_stops = [
    x
    for x in [
        atr_stop,
        ema_stop
    ]
    if (
        not pd.isna(x)
        and
        x > 0
        and
        x < breakout
    )
]

if possible_stops:

    stop = max(
        possible_stops
    )

else:

    stop = atr_stop

risk = (
    breakout -
    stop
)

t1 = (
    breakout +
    risk
)

t15 = (
    breakout +
    1.5 * risk
)

t2 = (
    breakout +
    2 * risk
)


a, b, c, d = (
    st.columns(4)
)

a.metric(
    "Current",
    f"₹{close:,.2f}"
)

b.metric(
    "Breakout",
    f"₹{breakout:,.2f}"
)

c.metric(
    "Indicative SL",
    f"₹{stop:,.2f}"
)

d.metric(
    "Risk",
    f"₹{risk:,.2f}"
)


st.write(
    f"**1R:** ₹{t1:,.2f}"
    f" | "
    f"**1.5R:** ₹{t15:,.2f}"
    f" | "
    f"**2R:** ₹{t2:,.2f}"
)


st.warning(
    "Research only. The scanner does not guarantee "
    "breakouts or returns. Historical statistics are "
    "evidence, not prediction."
)


# ============================================================
# TOP 3
# ============================================================

st.markdown("---")

st.subheader(
    "🥇🥈🥉 Top 3 Precision Opportunities"
)

top3 = vdf.head(3)

for rank, (_, row) in enumerate(
    top3.iterrows(),
    start=1
):

    st.write(
        f"**{rank}. {row['Symbol']}** — "
        f"Opportunity **{row['Opportunity']}/100** | "
        f"Tech **{row['Tech']}** | "
        f"Hist **{row['Hist Score']}** | "
        f"Win **{row['Win %']:.1f}%** | "
        f"PF **{row['PF'] if pd.notna(row['PF']) else '—'}** | "
        f"{row['Validation']}"
    )


# ============================================================
# EXPLANATION
# ============================================================

with st.expander(
    "🔎 How V4.2 ranks stocks"
):

    st.markdown(
        """
### Technical Score — 55%

Measures the current setup:

- 150 SMA > 220 EMA
- Close > 50 SMA
- 50 SMA > 150 SMA
- >25% above 52-week low
- Previous EMA220 dip
- Breakout proximity
- 5/10/20-day contraction
- Volume dry-up
- ATR compression
- Higher lows
- RSI

### Historical Score — 45%

Measures:

- Historical setup count
- Actual breakout count
- Breakout rate
- Valid next-day entries
- Win rate
- Profit factor
- Expectancy
- Sample size

### Sample-size penalty

A stock with:

**1 trade + 1 winner**

cannot beat a stock with:

**30 trades + consistently positive expectancy**

simply because its win rate is 100%.

### Actual historical sequence

**Pre-breakout**

↓

**Confirmed closing breakout**

↓

**Next-day OPEN**

↓

**Stop / Target / Time exit**

↓

**Historical statistics**

↓

**Final Opportunity Score**
        """
    )


st.caption(
    "StockPilot V4.2 | "
    +
    datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )
)
