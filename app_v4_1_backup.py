import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime
from io import StringIO

# ============================================================
# STOCKPILOT V4.1
# REAL PRE-BREAKOUT + BREAKOUT VALIDATION ENGINE
# ============================================================

st.set_page_config(
    page_title="StockPilot V4.1",
    page_icon="🚀",
    layout="wide"
)

NIFTY500_URL = (
    "https://archives.nseindia.com/content/indices/"
    "ind_nifty500list.csv"
)


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_float(value, default=np.nan):
    try:
        return float(value)
    except Exception:
        return default


def clean_columns(df):
    """
    Normalise yfinance columns.
    Handles both normal and MultiIndex data.
    """

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
# LOAD NIFTY 500
# ============================================================

@st.cache_data(ttl=1800, show_spinner=False)
def load_nifty500():

    try:

        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "text/csv,*/*"
        }

        response = requests.get(
            NIFTY500_URL,
            headers=headers,
            timeout=20
        )

        response.raise_for_status()

        df = pd.read_csv(
            StringIO(response.text)
        )

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

    except Exception as e:

        st.error(
            f"Unable to load NIFTY 500 universe: {e}"
        )

        return []


# ============================================================
# MARKET DATA
# ============================================================

@st.cache_data(
    ttl=1800,
    show_spinner=False
)
def download_market_data(
    symbols,
    period="2y"
):

    tickers = [
        f"{symbol}.NS"
        for symbol in symbols
    ]

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


def extract_symbol(
    data,
    symbol
):

    ticker = f"{symbol}.NS"

    if data is None or data.empty:
        return pd.DataFrame()

    try:

        if isinstance(
            data.columns,
            pd.MultiIndex
        ):

            level0 = (
                data.columns
                .get_level_values(0)
            )

            level1 = (
                data.columns
                .get_level_values(1)
            )

            if ticker in level0:

                df = data[ticker].copy()

            elif symbol in level0:

                df = data[symbol].copy()

            elif ticker in level1:

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
            col in df.columns
            for col in required
        ):
            return pd.DataFrame()

        df = (
            df[required]
            .dropna()
            .copy()
        )

        return df

    except Exception:

        return pd.DataFrame()


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(df):

    x = df.copy()

    # --------------------------------------------------------
    # MOVING AVERAGES
    # --------------------------------------------------------

    x["SMA20"] = (
        x["Close"]
        .rolling(20)
        .mean()
    )

    x["SMA50"] = (
        x["Close"]
        .rolling(50)
        .mean()
    )

    x["SMA150"] = (
        x["Close"]
        .rolling(150)
        .mean()
    )

    x["EMA220"] = (
        x["Close"]
        .ewm(
            span=220,
            adjust=False
        )
        .mean()
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    previous_close = (
        x["Close"].shift(1)
    )

    tr1 = (
        x["High"] -
        x["Low"]
    )

    tr2 = (
        x["High"] -
        previous_close
    ).abs()

    tr3 = (
        x["Low"] -
        previous_close
    ).abs()

    x["TR"] = pd.concat(
        [
            tr1,
            tr2,
            tr3
        ],
        axis=1
    ).max(axis=1)

    x["ATR14"] = (
        x["TR"]
        .rolling(14)
        .mean()
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    delta = x["Close"].diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = (
        gain
        .rolling(14)
        .mean()
    )

    avg_loss = (
        loss
        .rolling(14)
        .mean()
    )

    rs = (
        avg_gain /
        avg_loss.replace(
            0,
            np.nan
        )
    )

    x["RSI14"] = (
        100 -
        (
            100 /
            (1 + rs)
        )
    )

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    x["Vol20"] = (
        x["Volume"]
        .rolling(20)
        .mean()
    )

    x["VolRatio"] = (
        x["Volume"] /
        x["Vol20"]
    )

    # --------------------------------------------------------
    # CONSOLIDATION RANGES
    # --------------------------------------------------------

    x["Range5"] = (
        (
            x["High"]
            .rolling(5)
            .max()
            -
            x["Low"]
            .rolling(5)
            .min()
        )
        /
        x["Close"]
        *
        100
    )

    x["Range10"] = (
        (
            x["High"]
            .rolling(10)
            .max()
            -
            x["Low"]
            .rolling(10)
            .min()
        )
        /
        x["Close"]
        *
        100
    )

    x["Range20"] = (
        (
            x["High"]
            .rolling(20)
            .max()
            -
            x["Low"]
            .rolling(20)
            .min()
        )
        /
        x["Close"]
        *
        100
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # ALL BREAKOUT LEVELS USE PREVIOUS DATA ONLY
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # PREVIOUS 90 SESSION EMA220 DIP
    # --------------------------------------------------------

    below_ema = (
        x["Low"] <
        x["EMA220"]
    )

    x["Dip90"] = (
        below_ema
        .rolling(90)
        .max()
        .fillna(0)
        .astype(bool)
    )

    # --------------------------------------------------------
    # ATR COMPRESSION
    # --------------------------------------------------------

    x["ATR20Avg"] = (
        x["ATR14"]
        .rolling(20)
        .mean()
    )

    x["ATRCompression"] = (
        x["ATR14"] /
        x["ATR20Avg"]
    )

    # --------------------------------------------------------
    # HIGHER LOW
    # --------------------------------------------------------

    x["RecentLow10"] = (
        x["Low"]
        .rolling(10)
        .min()
    )

    x["OlderLow"] = (
        x["Low"]
        .shift(10)
        .rolling(20)
        .min()
    )

    return x


# ============================================================
# CURRENT TECHNICAL SCORE
# ============================================================

def technical_score(df):

    if len(df) < 260:
        return 0, {}

    r = df.iloc[-1]

    close = safe_float(
        r["Close"]
    )

    sma50 = safe_float(
        r["SMA50"]
    )

    sma150 = safe_float(
        r["SMA150"]
    )

    ema220 = safe_float(
        r["EMA220"]
    )

    resistance = safe_float(
        r["Resistance20"]
    )

    high52 = safe_float(
        r["High52"]
    )

    low52 = safe_float(
        r["Low52"]
    )

    range5 = safe_float(
        r["Range5"]
    )

    range10 = safe_float(
        r["Range10"]
    )

    range20 = safe_float(
        r["Range20"]
    )

    volume_ratio = safe_float(
        r["VolRatio"]
    )

    atr_compression = safe_float(
        r["ATRCompression"]
    )

    rsi = safe_float(
        r["RSI14"]
    )

    score = 0

    # --------------------------------------------------------
    # TREND
    # --------------------------------------------------------

    if sma150 > ema220:
        score += 10

    if close > sma50:
        score += 10

    if sma50 > sma150:
        score += 10

    # --------------------------------------------------------
    # 52-WEEK LOW
    # --------------------------------------------------------

    if (
        not pd.isna(low52)
        and low52 > 0
        and close > 1.25 * low52
    ):
        score += 5

    # --------------------------------------------------------
    # EMA220 DIP
    # --------------------------------------------------------

    if bool(r["Dip90"]):
        score += 10

    # --------------------------------------------------------
    # BREAKOUT PROXIMITY
    # --------------------------------------------------------

    breakout_distance = np.nan

    if (
        not pd.isna(resistance)
        and resistance > 0
    ):

        breakout_distance = (
            resistance -
            close
        ) / close * 100

        # IMPORTANT:
        # negative = already broken out
        if (
            breakout_distance >= 0
            and breakout_distance <= 2
        ):
            score += 15

        elif (
            breakout_distance > 2
            and breakout_distance <= 5
        ):
            score += 10

        elif (
            breakout_distance > 5
            and breakout_distance <= 8
        ):
            score += 5

    # --------------------------------------------------------
    # RANGE CONTRACTION
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # VOLUME DRY-UP
    # --------------------------------------------------------

    if not pd.isna(volume_ratio):

        if volume_ratio <= 0.70:
            score += 5

        elif volume_ratio <= 0.90:
            score += 3

    # --------------------------------------------------------
    # ATR COMPRESSION
    # --------------------------------------------------------

    if not pd.isna(
        atr_compression
    ):

        if atr_compression <= 0.85:
            score += 5

        elif atr_compression <= 1:
            score += 3

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    if not pd.isna(rsi):

        if 45 <= rsi <= 68:
            score += 5

    # --------------------------------------------------------
    # HIGHER LOW
    # --------------------------------------------------------

    recent_low = (
        df["Low"]
        .tail(10)
        .min()
    )

    older_low = (
        df["Low"]
        .iloc[-30:-10]
        .min()
    )

    if recent_low > older_low:
        score += 5

    # --------------------------------------------------------
    # 52W HIGH PROXIMITY
    # --------------------------------------------------------

    high_distance = np.nan

    if (
        not pd.isna(high52)
        and high52 > 0
    ):

        high_distance = (
            high52 -
            close
        ) / close * 100

        if high_distance <= 5:
            score += 5

        elif high_distance <= 10:
            score += 3

    score = min(
        score,
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

        "range5": range5,

        "range10": range10,

        "range20": range20,

        "vol_ratio": volume_ratio,

        "atr_compression":
            atr_compression,

        "rsi": rsi,

        "breakout_distance":
            breakout_distance,

        "high_distance":
            high_distance
    }


# ============================================================
# HISTORICAL PRE-BREAKOUT SETUP
# ============================================================

def is_prebreakout_setup(
    row,
    min_score=65,
    max_distance=8
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
        "ATRCompression"
    ]

    for column in required:

        if (
            column not in row
            or pd.isna(row[column])
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

    # ========================================================
    # CRITICAL V4.1 FIX
    # A PRE-BREAKOUT STOCK MUST STILL BE BELOW RESISTANCE.
    # ========================================================

    if distance < 0:
        return False

    if distance > max_distance:
        return False

    # --------------------------------------------------------
    # ORIGINAL TREND CONDITIONS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 52W LOW
    # --------------------------------------------------------

    low52 = float(
        row["Low52"]
    )

    if low52 <= 0:
        return False

    if close <= 1.25 * low52:
        return False

    # --------------------------------------------------------
    # EMA220 DIP
    # --------------------------------------------------------

    if not bool(
        row["Dip90"]
    ):
        return False

    # --------------------------------------------------------
    # RANGE CONTRACTION
    # --------------------------------------------------------

    if (
        row["Range5"] >
        row["Range10"]
    ):
        return False

    if (
        row["Range10"] >
        row["Range20"]
    ):
        return False

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    if (
        row["ATRCompression"]
        > 1.15
    ):
        return False

    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------

    score = 0

    if (
        row["SMA150"] >
        row["EMA220"]
    ):
        score += 10

    if (
        close >
        row["SMA50"]
    ):
        score += 10

    if (
        row["SMA50"] >
        row["SMA150"]
    ):
        score += 10

    if (
        close >
        1.25 * low52
    ):
        score += 5

    if bool(
        row["Dip90"]
    ):
        score += 10

    if distance <= 2:
        score += 15

    elif distance <= 5:
        score += 10

    elif distance <= 8:
        score += 5

    if (
        row["Range5"] <
        row["Range10"]
    ):
        score += 5

    if (
        row["Range10"] <
        row["Range20"]
    ):
        score += 5

    if (
        row["Range20"] <= 20
    ):
        score += 5

    if (
        row["VolRatio"] <= 0.70
    ):
        score += 5

    elif (
        row["VolRatio"] <= 0.90
    ):
        score += 3

    if (
        row["ATRCompression"]
        <= 0.85
    ):
        score += 5

    elif (
        row["ATRCompression"]
        <= 1
    ):
        score += 3

    if score < min_score:
        return False

    return True


# ============================================================
# VALIDATION ENGINE
# ============================================================

def backtest_symbol(
    df,
    min_score=65,
    max_breakout_distance=8,
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

    target_hits = 0
    stop_hits = 0
    time_exits = 0
    no_breakout = 0

    trades = []

    # Enough warm-up for EMA220 + 52W calculations
    i = 260

    last_signal_position = -9999

    while (
        i <
        len(x)
        -
        breakout_window
        -
        2
    ):

        # ----------------------------------------------------
        # COOLDOWN
        # ----------------------------------------------------

        if (
            i -
            last_signal_position
            < cooldown
        ):

            i += 1
            continue

        setup_row = x.iloc[i]

        # ----------------------------------------------------
        # PRE-BREAKOUT SETUP
        # ----------------------------------------------------

        if not is_prebreakout_setup(
            setup_row,
            min_score=min_score,
            max_distance=max_breakout_distance
        ):

            i += 1
            continue

        setups += 1

        resistance = safe_float(
            setup_row["Resistance20"]
        )

        # ----------------------------------------------------
        # ACTUAL BREAKOUT SEARCH
        # ----------------------------------------------------

        breakout_idx = None

        end_idx = min(
            i + breakout_window,
            len(x) - 2
        )

        for j in range(
            i + 1,
            end_idx + 1
        ):

            close_j = safe_float(
                x.iloc[j]["Close"]
            )

            # ACTUAL CONFIRMED BREAKOUT
            if close_j > resistance:

                breakout_idx = j

                break

        # ----------------------------------------------------
        # NO BREAKOUT
        # ----------------------------------------------------

        if breakout_idx is None:

            no_breakout += 1

            last_signal_position = i

            i += cooldown

            continue

        breakouts += 1

        # ----------------------------------------------------
        # NEXT DAY OPEN ENTRY
        # ----------------------------------------------------

        entry_idx = (
            breakout_idx + 1
        )

        if (
            entry_idx >= len(x)
        ):
            break

        entry_price = safe_float(
            x.iloc[entry_idx]["Open"]
        )

        if (
            pd.isna(entry_price)
            or entry_price <= 0
        ):

            i = (
                entry_idx +
                cooldown
            )

            continue

        entries += 1

        # ----------------------------------------------------
        # RISK
        # ----------------------------------------------------

        atr = safe_float(
            x.iloc[breakout_idx]["ATR14"]
        )

        ema220 = safe_float(
            x.iloc[breakout_idx]["EMA220"]
        )

        if (
            pd.isna(atr)
            or atr <= 0
        ):

            i = (
                entry_idx +
                cooldown
            )

            continue

        atr_stop = (
            entry_price -
            2.0 * atr
        )

        ema_stop = np.nan

        if not pd.isna(
            ema220
        ):

            ema_stop = (
                ema220 *
                0.995
            )

        candidates = [
            value
            for value in [
                atr_stop,
                ema_stop
            ]
            if (
                not pd.isna(value)
                and
                value > 0
                and
                value < entry_price
            )
        ]

        if candidates:

            stop_price = max(
                candidates
            )

        else:

            stop_price = (
                entry_price -
                2.0 * atr
            )

        risk = (
            entry_price -
            stop_price
        )

        if risk <= 0:

            i = (
                entry_idx +
                cooldown
            )

            continue

        # ----------------------------------------------------
        # TARGET
        # ----------------------------------------------------

        target_price = (
            entry_price +
            target_r * risk
        )

        # ----------------------------------------------------
        # TRADE SIMULATION
        # ----------------------------------------------------

        outcome = "TIME EXIT"

        exit_price = np.nan

        exit_idx = None

        forward_end = min(
            entry_idx +
            holding_period,
            len(x) - 1
        )

        for k in range(
            entry_idx,
            forward_end + 1
        ):

            day = x.iloc[k]

            high = safe_float(
                day["High"]
            )

            low = safe_float(
                day["Low"]
            )

            close = safe_float(
                day["Close"]
            )

            target_hit = (
                high >= target_price
            )

            stop_hit = (
                low <= stop_price
            )

            # ------------------------------------------------
            # CONSERVATIVE RULE
            # If both occur on same daily candle,
            # assume STOP happened first.
            # ------------------------------------------------

            if (
                target_hit
                and stop_hit
            ):

                outcome = "STOP"

                exit_price = (
                    stop_price
                )

                exit_idx = k

                break

            if stop_hit:

                outcome = "STOP"

                exit_price = (
                    stop_price
                )

                exit_idx = k

                break

            if target_hit:

                outcome = "TARGET"

                exit_price = (
                    target_price
                )

                exit_idx = k

                break

            if (
                k ==
                forward_end
            ):

                outcome = "TIME EXIT"

                exit_price = close

                exit_idx = k

        if (
            pd.isna(exit_price)
        ):

            i = (
                entry_idx +
                cooldown
            )

            continue

        return_pct = (
            exit_price /
            entry_price -
            1
        ) * 100

        r_multiple = (
            exit_price -
            entry_price
        ) / risk

        if outcome == "TARGET":

            target_hits += 1

        elif outcome == "STOP":

            stop_hits += 1

        else:

            time_exits += 1

        trades.append({

            "Setup Date":
                x.index[i],

            "Breakout Date":
                x.index[breakout_idx],

            "Entry Date":
                x.index[entry_idx],

            "Exit Date":
                x.index[exit_idx],

            "Entry":
                entry_price,

            "Stop":
                stop_price,

            "Target":
                target_price,

            "Exit":
                exit_price,

            "Outcome":
                outcome,

            "Return %":
                return_pct,

            "R":
                r_multiple
        })

        # ----------------------------------------------------
        # PREVENT OVERLAPPING SETUPS
        # ----------------------------------------------------

        last_signal_position = (
            exit_idx
        )

        i = (
            exit_idx + 1
        )

    # ========================================================
    # STATISTICS
    # ========================================================

    trade_df = pd.DataFrame(
        trades
    )

    if trade_df.empty:

        return {

            "setups":
                setups,

            "breakouts":
                breakouts,

            "breakout_rate":
                (
                    breakouts /
                    setups *
                    100
                    if setups
                    else 0
                ),

            "entries":
                entries,

            "target_hits":
                0,

            "stop_hits":
                0,

            "time_exits":
                0,

            "win_rate":
                0,

            "avg_winner":
                0,

            "avg_loser":
                0,

            "profit_factor":
                0,

            "expectancy_r":
                0,

            "max_drawdown_r":
                0,

            "trades":
                trade_df
        }

    # --------------------------------------------------------
    # WINNERS / LOSERS
    # --------------------------------------------------------

    winners = (
        trade_df[
            trade_df["R"] > 0
        ]["R"]
    )

    losers = (
        trade_df[
            trade_df["R"] < 0
        ]["R"]
    )

    win_rate = (
        len(winners) /
        len(trade_df) *
        100
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

    gross_loss = (
        abs(losers.sum())
        if len(losers)
        else 0
    )

    if gross_loss > 0:

        profit_factor = (
            gross_profit /
            gross_loss
        )

    else:

        profit_factor = np.inf

    # --------------------------------------------------------
    # EXPECTANCY
    # --------------------------------------------------------

    expectancy = (
        trade_df["R"]
        .mean()
    )

    # --------------------------------------------------------
    # MAX DRAWDOWN
    # --------------------------------------------------------

    equity = (
        trade_df["R"]
        .cumsum()
    )

    running_max = (
        equity
        .cummax()
    )

    drawdown = (
        equity -
        running_max
    )

    max_drawdown = (
        drawdown.min()
    )

    return {

        "setups":
            setups,

        "breakouts":
            breakouts,

        "breakout_rate":
            (
                breakouts /
                setups *
                100
                if setups
                else 0
            ),

        "entries":
            entries,

        "target_hits":
            target_hits,

        "stop_hits":
            stop_hits,

        "time_exits":
            time_exits,

        "no_breakout":
            no_breakout,

        "win_rate":
            win_rate,

        "avg_winner":
            avg_winner,

        "avg_loser":
            avg_loser,

        "profit_factor":
            profit_factor,

        "expectancy_r":
            expectancy,

        "max_drawdown_r":
            max_drawdown,

        "trades":
            trade_df
    }


# ============================================================
# VALIDATION GRADE
# ============================================================

def validation_grade(stats):

    trades = stats["entries"]

    pf = stats["profit_factor"]

    expectancy = (
        stats["expectancy_r"]
    )

    if trades < 10:

        return "⚠️ LOW SAMPLE"

    if trades < 30:

        if (
            expectancy > 0
            and pf > 1
        ):
            return "🟡 LIMITED POSITIVE"

        return "🟡 LIMITED"

    if (
        expectancy > 0
        and pf >= 1.5
    ):

        return "🟢 STRONG"

    if (
        expectancy > 0
        and pf > 1
    ):

        return "🟢 POSITIVE"

    if pf > 0.8:

        return "🟠 NEUTRAL"

    return "🔴 WEAK"


# ============================================================
# CURRENT SCANNER
# ============================================================

def scan_current(
    symbols,
    min_score=65,
    max_breakout_distance=8
):

    data = download_market_data(
        symbols,
        period="2y"
    )

    results = []

    for symbol in symbols:

        df = extract_symbol(
            data,
            symbol
        )

        if (
            df.empty
            or len(df) < 260
        ):
            continue

        df = add_indicators(
            df
        )

        score, details = (
            technical_score(df)
        )

        if not details:
            continue

        distance = (
            details[
                "breakout_distance"
            ]
        )

        # ----------------------------------------------------
        # CRITICAL FIX:
        # Already broken stocks are NOT pre-breakout.
        # ----------------------------------------------------

        if pd.isna(distance):
            continue

        if distance < 0:
            continue

        if (
            distance >
            max_breakout_distance
        ):
            continue

        if score < min_score:
            continue

        if score >= 90:

            stage = (
                "EXCEPTIONAL"
            )

        elif score >= 80:

            stage = (
                "VERY HIGH"
            )

        elif score >= 70:

            stage = "HIGH"

        else:

            stage = "WATCH"

        results.append({

            "Symbol":
                symbol,

            "Score":
                score,

            "Stage":
                stage,

            "Close":
                details["close"],

            "Breakout":
                details["resistance"],

            "Distance":
                distance,

            "RSI":
                details["rsi"],

            "Volume":
                details["vol_ratio"],

            "ATR Comp":
                details[
                    "atr_compression"
                ],

            "Range 5D":
                details["range5"],

            "Range 10D":
                details["range10"],

            "Range 20D":
                details["range20"],

            "52W High Dist":
                details[
                    "high_distance"
                ],

            "Data":
                df
        })

    results.sort(
        key=lambda x:
        x["Score"],
        reverse=True
    )

    return results


# ============================================================
# USER INTERFACE
# ============================================================

st.title(
    "🚀 StockPilot V4.1"
)

st.caption(
    "Pre-breakout → confirmed breakout → next-day entry → historical validation"
)

st.info(
    "V4.1 is a research/backtesting engine. "
    "Historical results do not guarantee future returns."
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

min_score = (
    st.sidebar.slider(
        "Minimum Technical Score",
        50,
        95,
        65,
        5
    )
)

max_breakout_distance = (
    st.sidebar.slider(
        "Maximum Breakout Distance %",
        2.0,
        15.0,
        8.0,
        0.5
    )
)

strict_mode = (
    st.sidebar.checkbox(
        "Strict Pre-Breakout Mode",
        True
    )
)

st.sidebar.header(
    "📊 Historical Validation"
)

history_years = (
    st.sidebar.selectbox(
        "Historical Period",
        [1, 2, 3],
        index=1
    )
)

forward_days = (
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
        "Target",
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

backtest_min_score = (
    st.sidebar.slider(
        "Historical Setup Score",
        50,
        90,
        min_score,
        5
    )
)

run_button = (
    st.sidebar.button(
        "🔍 RUN V4.1",
        type="primary",
        use_container_width=True
    )
)


# ============================================================
# START SCREEN
# ============================================================

if not run_button:

    st.markdown(
        """
## 🎯 What V4.1 tests

### 1️⃣ Pre-Breakout Setup

Trend + consolidation + contraction + proximity to resistance.

↓

### 2️⃣ Confirmed Breakout

The stock must **close above the previous 20-session resistance**.

↓

### 3️⃣ Entry

The historical entry is the **next trading day's OPEN**.

↓

### 4️⃣ Risk

ATR/EMA220 based protective stop.

↓

### 5️⃣ Target

Choose:

**1R / 1.5R / 2R**

↓

### 6️⃣ Exit

Target / Stop / Time Exit.

↓

### 7️⃣ Validation

Win rate + Profit Factor + Expectancy + Drawdown + Sample Size.
        """
    )

    st.warning(
        "Press RUN V4.1 to start."
    )

    st.stop()


# ============================================================
# LOAD UNIVERSE
# ============================================================

with st.spinner(
    "Loading NIFTY 500..."
):

    all_symbols = (
        load_nifty500()
    )

if not all_symbols:

    st.stop()


# NOTE:
# The current free implementation uses the NIFTY500 universe.
# NIFTY100/NIFTY250 options currently use the first 100/250
# rows from the downloaded NIFTY500 list.
#
# We can replace these with the official NIFTY100/NIFTY250
# constituent lists later.

if universe == "NIFTY 500":

    symbols = all_symbols

elif universe == "NIFTY 250":

    symbols = all_symbols[:250]

else:

    symbols = all_symbols[:100]


# ============================================================
# CURRENT SCAN
# ============================================================

with st.spinner(
    f"Scanning {len(symbols)} stocks..."
):

    current = scan_current(
        symbols,
        min_score=min_score,
        max_breakout_distance=
            max_breakout_distance
    )


st.success(
    f"Current scan completed — "
    f"{len(current)} genuine pre-breakout candidates found."
)


if not current:

    st.warning(
        "No stocks match the current filters. "
        "Try lowering Minimum Technical Score "
        "or increasing Maximum Breakout Distance."
    )

    st.stop()


# ============================================================
# CURRENT TOP CANDIDATES
# ============================================================

st.subheader(
    "🔥 Top Pre-Breakout Candidates"
)

display_rows = []

for item in current[:20]:

    display_rows.append({

        "Symbol":
            item["Symbol"],

        "Score":
            item["Score"],

        "Stage":
            item["Stage"],

        "Close":
            round(
                item["Close"],
                2
            ),

        "Breakout":
            round(
                item["Breakout"],
                2
            ),

        "Distance %":
            round(
                item["Distance"],
                2
            ),

        "RSI":
            round(
                item["RSI"],
                1
            ),

        "Vol":
            round(
                item["Volume"],
                2
            ),

        "ATR":
            round(
                item["ATR Comp"],
                2
            ),

        "5D":
            round(
                item["Range 5D"],
                2
            ),

        "10D":
            round(
                item["Range 10D"],
                2
            ),

        "20D":
            round(
                item["Range 20D"],
                2
            )
    })


current_df = pd.DataFrame(
    display_rows
)

st.dataframe(
    current_df,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "Only the top 20 current candidates are displayed. "
    "Historical validation is performed on the top 10."
)


# ============================================================
# HISTORICAL VALIDATION
# ============================================================

st.subheader(
    "🧪 V4.1 Historical Breakout Validation"
)

history_period = (
    f"{history_years + 1}y"
)

# Validate top 10 only.
top_candidates = current[:10]

validation_rows = []

progress = st.progress(0)

for n, item in enumerate(
    top_candidates
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
            col in hist.columns
            for col in required
        ):
            continue

        hist = (
            hist[
                required
            ]
            .dropna()
        )

        hist = add_indicators(
            hist
        )

        stats = backtest_symbol(

            hist,

            min_score=
                backtest_min_score,

            max_breakout_distance=
                max_breakout_distance,

            breakout_window=
                breakout_window,

            holding_period=
                forward_days,

            target_r=
                target_r,

            cooldown=
                cooldown
        )

        if stats is None:
            continue

        validation_rows.append({

            "Symbol":
                symbol,

            "Tech Score":
                item["Score"],

            "Setups":
                stats["setups"],

            "Breakouts":
                stats["breakouts"],

            "Breakout Rate":
                stats["breakout_rate"],

            "Entries":
                stats["entries"],

            "Targets":
                stats["target_hits"],

            "Stops":
                stats["stop_hits"],

            "Time Exit":
                stats["time_exits"],

            "Win Rate":
                stats["win_rate"],

            "Avg Winner R":
                stats["avg_winner"],

            "Avg Loser R":
                stats["avg_loser"],

            "Profit Factor":
                stats["profit_factor"],

            "Expectancy R":
                stats["expectancy_r"],

            "Max DD R":
                stats["max_drawdown_r"],

            "Validation":
                validation_grade(stats)
        })

    except Exception:

        pass

    progress.progress(
        (n + 1) /
        len(top_candidates)
    )

progress.empty()


# ============================================================
# VALIDATION TABLE
# ============================================================

if not validation_rows:

    st.error(
        "No historical validation results were produced."
    )

    st.stop()


validation_df = pd.DataFrame(
    validation_rows
)


# ------------------------------------------------------------
# RANKING
# ------------------------------------------------------------

validation_df["_expectancy"] = (
    validation_df[
        "Expectancy R"
    ]
)

validation_df["_profit_factor"] = (
    validation_df[
        "Profit Factor"
    ]
    .replace(
        np.inf,
        999
    )
)

validation_df["_entries"] = (
    validation_df[
        "Entries"
    ]
)

validation_df = (
    validation_df
    .sort_values(
        [
            "_expectancy",
            "_profit_factor",
            "_entries"
        ],
        ascending=False
    )
)

validation_df = (
    validation_df
    .drop(
        columns=[
            "_expectancy",
            "_profit_factor",
            "_entries"
        ]
    )
)


# ------------------------------------------------------------
# ROUND DISPLAY NUMBERS
# ------------------------------------------------------------

for col in [
    "Breakout Rate",
    "Win Rate",
    "Avg Winner R",
    "Avg Loser R",
    "Profit Factor",
    "Expectancy R",
    "Max DD R"
]:

    if col in validation_df.columns:

        validation_df[col] = (
            validation_df[col]
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .round(2)
        )


st.dataframe(
    validation_df,
    use_container_width=True,
    hide_index=True
)


st.caption(
    "Daily OHLC data cannot determine the exact intraday "
    "order when both target and stop are touched on the "
    "same candle. V4.1 conservatively counts such cases "
    "as STOP."
)


# ============================================================
# BEST VALIDATED CANDIDATE
# ============================================================

best_symbol = (
    validation_df.iloc[0]["Symbol"]
)

best_item = next(
    item
    for item in current
    if item["Symbol"] ==
    best_symbol
)

best_validation = (
    validation_df.iloc[0]
)


st.markdown("---")

st.subheader(
    f"🏆 Best Validated Candidate — "
    f"{best_symbol}"
)


c1, c2, c3, c4 = (
    st.columns(4)
)


c1.metric(
    "Technical Score",
    f"{best_item['Score']}/100"
)

c2.metric(
    "Win Rate",
    f"{best_validation['Win Rate']:.1f}%"
)

pf = best_validation[
    "Profit Factor"
]

if pd.isna(pf):

    pf_text = "—"

else:

    pf_text = (
        "∞"
        if pf >= 999
        else f"{pf:.2f}"
    )

c3.metric(
    "Profit Factor",
    pf_text
)

c4.metric(
    "Expectancy",
    f"{best_validation['Expectancy R']:.2f}R"
)


st.write(
    f"**Validation:** "
    f"{best_validation['Validation']}"
)


st.write(
    f"**Historical setups:** "
    f"{int(best_validation['Setups'])}"
    f"  |  "
    f"**Breakouts:** "
    f"{int(best_validation['Breakouts'])}"
    f"  |  "
    f"**Entries:** "
    f"{int(best_validation['Entries'])}"
)


st.write(
    f"**Targets:** "
    f"{int(best_validation['Targets'])}"
    f"  |  "
    f"**Stops:** "
    f"{int(best_validation['Stops'])}"
    f"  |  "
    f"**Time exits:** "
    f"{int(best_validation['Time Exit'])}"
)


st.write(
    f"**Average winner:** "
    f"{best_validation['Avg Winner R']:.2f}R"
    f"  |  "
    f"**Average loser:** "
    f"{best_validation['Avg Loser R']:.2f}R"
)


st.write(
    f"**Maximum drawdown:** "
    f"{best_validation['Max DD R']:.2f}R"
)


# ============================================================
# CURRENT SETUP
# ============================================================

st.subheader(
    f"📍 Current Setup — "
    f"{best_symbol}"
)

latest = (
    best_item["Data"]
    .iloc[-1]
)

close = safe_float(
    latest["Close"]
)

breakout = safe_float(
    latest["Resistance20"]
)

atr = safe_float(
    latest["ATR14"]
)

ema220 = safe_float(
    latest["EMA220"]
)


# ------------------------------------------------------------
# INDICATIVE STOP
# ------------------------------------------------------------

atr_stop = np.nan

if (
    not pd.isna(atr)
    and atr > 0
):

    atr_stop = (
        breakout -
        2 * atr
    )


ema_stop = np.nan

if (
    not pd.isna(ema220)
    and ema220 > 0
):

    ema_stop = (
        ema220 *
        0.995
    )


stop_candidates = [
    value
    for value in [
        atr_stop,
        ema_stop
    ]
    if (
        not pd.isna(value)
        and value > 0
        and value < breakout
    )
]


if stop_candidates:

    stop = max(
        stop_candidates
    )

else:

    stop = atr_stop


risk = np.nan

if (
    not pd.isna(stop)
    and breakout > stop
):

    risk = (
        breakout -
        stop
    )


target1 = np.nan
target15 = np.nan
target2 = np.nan

if not pd.isna(risk):

    target1 = (
        breakout +
        risk
    )

    target15 = (
        breakout +
        1.5 * risk
    )

    target2 = (
        breakout +
        2 * risk
    )


a, b, c, d = (
    st.columns(4)
)


a.metric(
    "Current Price",
    f"₹{close:,.2f}"
)

b.metric(
    "Breakout Level",
    f"₹{breakout:,.2f}"
)

c.metric(
    "Indicative Stop",
    (
        f"₹{stop:,.2f}"
        if not pd.isna(stop)
        else "—"
    )
)

d.metric(
    "Risk",
    (
        f"₹{risk:,.2f}"
        if not pd.isna(risk)
        else "—"
    )
)


if not pd.isna(risk):

    st.write(
        f"**1R:** ₹{target1:,.2f}"
        f"  |  "
        f"**1.5R:** ₹{target15:,.2f}"
        f"  |  "
        f"**2R:** ₹{target2:,.2f}"
    )


st.warning(
    "These are research levels. "
    "The historical engine enters at the next day's OPEN "
    "only after a confirmed closing breakout."
)


# ============================================================
# EXPLANATION
# ============================================================

with st.expander(
    "🔎 How V4.1 works"
):

    st.markdown(
        """
### Historical trade sequence

**A. PRE-BREAKOUT**

The stock must be below its previous 20-day resistance.

**B. BREAKOUT**

Within the configured breakout window, the stock must CLOSE above that resistance.

**C. ENTRY**

Entry = next trading day's OPEN.

**D. STOP**

ATR / EMA220 based stop.

**E. TARGET**

1R / 1.5R / 2R.

**F. EXIT**

Target / Stop / Time Exit.

**G. VALIDATION**

The system calculates:

- Setup count
- Breakout count
- Breakout rate
- Entry count
- Target hits
- Stop hits
- Time exits
- Win rate
- Average winner
- Average loser
- Profit factor
- Expectancy
- Maximum drawdown
- Sample size
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "StockPilot V4.1 | "
    f"Updated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)
