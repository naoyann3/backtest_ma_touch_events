from __future__ import annotations

from datetime import datetime
import gc
from pathlib import Path
import time

import pandas as pd
import yfinance as yf

from config import LONG_TERM_SCREEN_VERSION, LONG_TERM_WATCHLISTS_DIR, ensure_results_dirs
from market_data_utils import prepare_price_history, select_latest_completed_row
from output_format import format_long_term_gc_output, format_long_term_latest_output, format_long_term_output

TICKERS_CSV = "tickers.csv"
OUTPUT_CSV = "long_term_watchlist.csv"
GC_OUTPUT_CSV = "long_term_gc_watchlist.csv"
UNIVERSE_OFFSET_TXT = "universe_offset.txt"
GC_WATCHLISTS_DIRNAME = "long_term_gc_watchlists"

MAX_TICKERS = 250
SLEEP_SEC = 0.8
TOP_N_OUTPUT = 50
TOP_N_GC_OUTPUT = 20

MIN_TURNOVER = 100_000_000
MIN_MARKET_CAP = 30_000_000_000
MIN_REVENUE_GROWTH_PCT = 5.0
MIN_PROFIT_MARGIN_PCT = 5.0
MIN_ROE_PCT = 8.0
MAX_52W_HIGH_GAP_PCT = 20.0
MAX_CHANGE_20D_PCT = 25.0
MAX_CHANGE_60D_PCT = 80.0
RECENT_CROSS_LOOKBACK = 10
PERFECT_ORDER_LOOKBACK = 5
BEARISH_ORDER_LOOKBACK = 60
REVERSAL_LOOKBACK = 10
GC_MIN_MA25_SLOPE_PCT = 0.25
GC_MIN_MA75_SLOPE_PCT = 0.25


def _ticker_path() -> Path:
    return Path(__file__).resolve().parent / TICKERS_CSV


def _latest_output_path() -> Path:
    return Path(__file__).resolve().parent / OUTPUT_CSV


def _latest_gc_output_path() -> Path:
    return Path(__file__).resolve().parent / GC_OUTPUT_CSV


def _offset_path() -> Path:
    return Path(__file__).resolve().parent / UNIVERSE_OFFSET_TXT


def _gc_watchlists_dir() -> Path:
    return Path(__file__).resolve().parent / "results" / GC_WATCHLISTS_DIRNAME


def load_universe_offset(total_count: int) -> int:
    if total_count <= 0:
        return 0

    path = _offset_path()
    if not path.exists():
        return 0

    try:
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            return 0
        return int(raw) % total_count
    except Exception:
        return 0


def save_universe_offset(next_offset: int) -> None:
    _offset_path().write_text(str(next_offset), encoding="utf-8")


def load_tickers() -> pd.DataFrame:
    df = pd.read_csv(_ticker_path())
    df = df.dropna(subset=["ticker"])
    df["ticker"] = df["ticker"].astype(str).str.strip()
    if "name" not in df.columns:
        df["name"] = df["ticker"]

    df = df.reset_index(drop=True)
    total_count = len(df)
    if total_count == 0:
        return df

    offset = load_universe_offset(total_count)
    rotated = pd.concat([df.iloc[offset:], df.iloc[:offset]], ignore_index=True)
    selected = rotated.head(MAX_TICKERS).reset_index(drop=True)
    next_offset = (offset + MAX_TICKERS) % total_count
    save_universe_offset(next_offset)
    return selected


def close_ticker_session(ticker_obj) -> None:
    if ticker_obj is None:
        return
    session = getattr(getattr(ticker_obj, "_data", None), "session", None)
    close = getattr(session, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass
    gc.collect()


def fetch_price_history(ticker_obj, ticker: str) -> pd.DataFrame | None:
    try:
        df = ticker_obj.history(
            period="18mo",
            interval="1d",
            auto_adjust=False,
            actions=True,
        )
    except Exception as exc:
        print(f"fetch_price_history error: {ticker} {exc}")
        return None

    if df is None or df.empty or len(df) < 120:
        return None

    return prepare_price_history(df)


def fetch_fundamentals(ticker_obj, ticker: str) -> dict | None:
    try:
        info = ticker_obj.info
    except Exception as exc:
        print(f"fetch_fundamentals error: {ticker} {exc}")
        return None

    if not info:
        return None

    market_cap = info.get("marketCap")
    roe = info.get("returnOnEquity")
    profit_margin = info.get("profitMargins")
    revenue_growth = info.get("revenueGrowth")
    current_ratio = info.get("currentRatio")
    debt_to_equity = info.get("debtToEquity")
    sector = info.get("sector")
    industry = info.get("industry")

    return {
        "market_cap": float(market_cap) if market_cap is not None else None,
        "roe_pct": float(roe) * 100 if roe is not None else None,
        "profit_margin_pct": float(profit_margin) * 100 if profit_margin is not None else None,
        "revenue_growth_pct": float(revenue_growth) * 100 if revenue_growth is not None else None,
        "current_ratio": float(current_ratio) if current_ratio is not None else None,
        "debt_to_equity": float(debt_to_equity) if debt_to_equity is not None else None,
        "sector": sector,
        "industry": industry,
    }


def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ma25"] = df["Close"].rolling(25).mean()
    df["ma75"] = df["Close"].rolling(75).mean()
    df["ma200"] = df["Close"].rolling(200).mean()
    df["ma25_above_ma200"] = df["ma25"] > df["ma200"]
    df["ma75_above_ma200"] = df["ma75"] > df["ma200"]
    df["ma25_cross_200_today"] = df["ma25_above_ma200"] & (~df["ma25_above_ma200"].shift(1).fillna(False))
    df["ma75_cross_200_today"] = df["ma75_above_ma200"] & (~df["ma75_above_ma200"].shift(1).fillna(False))
    df["ma25_cross_200_recent"] = (
        df["ma25_cross_200_today"].rolling(RECENT_CROSS_LOOKBACK, min_periods=1).max().fillna(0).astype(bool)
    )
    df["ma75_cross_200_recent"] = (
        df["ma75_cross_200_today"].rolling(RECENT_CROSS_LOOKBACK, min_periods=1).max().fillna(0).astype(bool)
    )
    df["ma25_above_ma75"] = df["ma25"] > df["ma75"]
    df["ma75_above_ma200"] = df["ma75"] > df["ma200"]
    df["perfect_order"] = df["ma25_above_ma75"] & df["ma75_above_ma200"]
    df["bearish_stack"] = (df["ma200"] > df["ma75"]) & (df["ma75"] > df["ma25"])
    df["bearish_perfect_order"] = df["bearish_stack"]
    df["ma25_cross_75_today"] = df["ma25_above_ma75"] & (~df["ma25_above_ma75"].shift(1).fillna(False))
    df["ma75_cross_200_recent_tight"] = (
        df["ma75_cross_200_today"].rolling(PERFECT_ORDER_LOOKBACK, min_periods=1).max().fillna(0).astype(bool)
    )
    df["ma25_cross_200_recent_tight"] = (
        df["ma25_cross_200_today"].rolling(PERFECT_ORDER_LOOKBACK, min_periods=1).max().fillna(0).astype(bool)
    )
    df["ma25_cross_75_recent_tight"] = (
        df["ma25_cross_75_today"].rolling(PERFECT_ORDER_LOOKBACK, min_periods=1).max().fillna(0).astype(bool)
    )
    df["perfect_order_recent"] = (
        df["perfect_order"].rolling(PERFECT_ORDER_LOOKBACK, min_periods=1).max().fillna(0).astype(bool)
    )
    df["perfect_order_today"] = df["perfect_order"] & (~df["perfect_order"].shift(1).fillna(False))
    df["perfect_order_recent_tight"] = (
        df["perfect_order_today"].rolling(REVERSAL_LOOKBACK, min_periods=1).max().fillna(0).astype(bool)
    )
    df["bearish_stack_recent"] = (
        df["bearish_stack"].rolling(BEARISH_ORDER_LOOKBACK, min_periods=1).max().fillna(0).astype(bool)
    )
    df["bearish_perfect_order_recent"] = (
        df["bearish_perfect_order"].shift(1).rolling(BEARISH_ORDER_LOOKBACK, min_periods=1).max().fillna(0).astype(bool)
    )
    df["ma25_slope_pct"] = (df["ma25"] - df["ma25"].shift(5)) / df["ma25"].shift(5) * 100
    df["ma75_slope_pct"] = (df["ma75"] - df["ma75"].shift(5)) / df["ma75"].shift(5) * 100
    df["ma200_slope_pct"] = (df["ma200"] - df["ma200"].shift(5)) / df["ma200"].shift(5) * 100
    df["close_vs_ma25_pct"] = (df["Close"] - df["ma25"]) / df["ma25"] * 100
    df["close_vs_ma75_pct"] = (df["Close"] - df["ma75"]) / df["ma75"] * 100
    df["close_vs_ma200_pct"] = (df["Close"] - df["ma200"]) / df["ma200"] * 100
    df["touch_ma25_intraday"] = df["Low"] <= df["ma25"]
    df["touch_ma75_intraday"] = df["Low"] <= df["ma75"]
    df["reclaim_ma25_close"] = df["touch_ma25_intraday"] & (df["Close"] >= df["ma25"])
    df["reclaim_ma75_close"] = df["touch_ma75_intraday"] & (df["Close"] >= df["ma75"])
    df["initial_trend_signal"] = (
        (df["Close"] >= df["ma25"])
        & df["perfect_order"]
        & (df["ma25_slope_pct"] > 0)
        & (df["ma75_slope_pct"] > 0)
        & df["ma75_cross_200_recent_tight"]
    )
    df["reversal_from_bearish_po"] = (
        df["bearish_perfect_order_recent"]
        & df["perfect_order"]
        & df["perfect_order_recent_tight"]
        & df["ma25_cross_75_recent_tight"]
        & df["ma75_cross_200_recent_tight"]
        & (df["Close"] >= df["ma25"])
        & (df["close_vs_ma25_pct"].between(-2.0, 10.0))
        & (df["ma25_slope_pct"] > 0)
        & (df["ma75_slope_pct"] > 0)
    )
    df["early_reversal_setup"] = (
        df["bearish_perfect_order_recent"]
        & (df["Close"] >= df["ma75"])
        & df["ma25_cross_75_recent_tight"]
        & (df["ma75_cross_200_recent_tight"] | df["ma25_cross_200_recent_tight"])
        & (df["ma25_slope_pct"] > 0)
        & (df["ma75_slope_pct"] > 0)
        & (df["close_vs_ma25_pct"].between(-3.0, 8.0))
    )
    df["vol_avg20"] = df["Volume"].rolling(20).mean()
    df["turnover"] = df["raw_close"] * df["Volume"]
    df["turnover_million"] = df["turnover"] / 1_000_000
    df["change_20d_pct"] = (df["Close"] - df["Close"].shift(20)) / df["Close"].shift(20) * 100
    df["change_60d_pct"] = (df["Close"] - df["Close"].shift(60)) / df["Close"].shift(60) * 100
    df["change_120d_pct"] = (df["Close"] - df["Close"].shift(120)) / df["Close"].shift(120) * 100
    df["high_60"] = df["High"].rolling(60).max()
    df["drawdown_from_60d_high_pct"] = (df["Close"] - df["high_60"]) / df["high_60"] * 100
    df["recent_high_252"] = df["High"].rolling(252, min_periods=120).max()
    df["gap_to_52w_high_pct"] = (df["recent_high_252"] - df["Close"]) / df["Close"] * 100
    df["vol_avg10"] = df["Volume"].rolling(10).mean()
    df["volume_ratio_20"] = df["Volume"] / df["vol_avg20"]
    df["volume_ratio_10"] = df["Volume"] / df["vol_avg10"]
    df["volume_ratio_20_mean_5"] = df["volume_ratio_20"].rolling(5).mean()
    df["volume_ratio_20_mean_10"] = df["volume_ratio_20"].rolling(10).mean()
    df["down_day"] = df["Close"] < df["Open"]
    df["down_volume_spike"] = df["down_day"] & (df["volume_ratio_20"] >= 1.5)
    price_range = (df["High"] - df["Low"]).replace(0, pd.NA)
    upper_shadow = (df["High"] - df[["Open", "Close"]].max(axis=1)).clip(lower=0)
    lower_shadow = (df[["Open", "Close"]].min(axis=1) - df["Low"]).clip(lower=0)
    df["upper_shadow_pct"] = (upper_shadow / price_range) * 100
    df["lower_shadow_pct"] = (lower_shadow / price_range) * 100
    df["distribution_warning"] = (
        (df["upper_shadow_pct"] >= 45)
        & (df["volume_ratio_20"] >= 1.5)
        & (df["gap_to_52w_high_pct"] <= 8)
    )
    cross_positions = pd.Series(range(len(df)), index=df.index).where(df["ma75_cross_200_today"])
    df["days_since_75gc200"] = pd.Series(range(len(df)), index=df.index) - cross_positions.ffill()
    df.loc[cross_positions.ffill().isna(), "days_since_75gc200"] = pd.NA
    po_positions = pd.Series(range(len(df)), index=df.index).where(df["perfect_order_today"])
    df["days_since_perfect_order"] = pd.Series(range(len(df)), index=df.index) - po_positions.ffill()
    df.loc[po_positions.ffill().isna(), "days_since_perfect_order"] = pd.NA
    pullback_trend_ok = (
        (df["Close"] >= df["ma200"])
        & (df["ma200_slope_pct"] > 0)
        & (df["ma75_slope_pct"] > 0)
        & df["ma25_above_ma75"]
    )
    df["failed_ma25_reclaim"] = df["touch_ma25_intraday"] & (~df["reclaim_ma25_close"])
    df["failed_ma75_reclaim"] = df["touch_ma75_intraday"] & (~df["reclaim_ma75_close"])
    df["support_reaction_ok"] = (
        (df["reclaim_ma25_close"] | df["reclaim_ma75_close"])
        & (~df["down_volume_spike"])
        & (df["drawdown_from_60d_high_pct"] > -15)
    )
    df["ma25_pullback_candidate"] = (
        pullback_trend_ok
        & df["touch_ma25_intraday"]
        & df["reclaim_ma25_close"]
        & (df["drawdown_from_60d_high_pct"] > -10)
    )
    df["ma75_pullback_candidate"] = (
        pullback_trend_ok
        & df["touch_ma75_intraday"]
        & df["reclaim_ma75_close"]
        & (df["drawdown_from_60d_high_pct"] > -15)
    )
    pullback_score = pd.Series(0.0, index=df.index)
    pullback_score += (df["Close"] >= df["ma200"]).fillna(False).astype(float) * 2.0
    pullback_score += (df["ma200_slope_pct"] > 0).fillna(False).astype(float) * 2.0
    pullback_score += (df["ma75_slope_pct"] > 0).fillna(False).astype(float) * 1.5
    pullback_score += df["ma25_above_ma75"].fillna(False).astype(float) * 1.0
    pullback_score += df["reclaim_ma25_close"].fillna(False).astype(float) * 1.5
    pullback_score += df["reclaim_ma75_close"].fillna(False).astype(float) * 2.5
    pullback_score += df["support_reaction_ok"].fillna(False).astype(float) * 1.0
    pullback_score += df["ma25_pullback_candidate"].fillna(False).astype(float) * 0.5
    pullback_score += df["ma75_pullback_candidate"].fillna(False).astype(float) * 1.0
    pullback_score += df["close_vs_ma25_pct"].between(-1.0, 3.0).fillna(False).astype(float) * 1.0
    pullback_score += df["close_vs_ma75_pct"].between(-1.0, 5.0).fillna(False).astype(float) * 1.0
    pullback_score -= df["down_volume_spike"].fillna(False).astype(float) * 2.5
    pullback_score -= df["failed_ma25_reclaim"].fillna(False).astype(float) * 1.5
    pullback_score -= df["failed_ma75_reclaim"].fillna(False).astype(float) * 2.0
    pullback_score -= (df["Close"] < df["ma75"]).fillna(False).astype(float) * 2.0
    pullback_score -= (df["change_20d_pct"] < -8).fillna(False).astype(float) * 1.0
    pullback_score -= (df["drawdown_from_60d_high_pct"] <= -15).fillna(False).astype(float) * 1.0
    df["pullback_score"] = pullback_score.round(2)
    df["pullback_candidate"] = (
        pullback_trend_ok
        & df["support_reaction_ok"]
        & (df["ma25_pullback_candidate"] | df["ma75_pullback_candidate"])
        & (df["pullback_score"] >= 6.0)
    )
    stealth_score = pd.Series(0.0, index=df.index)
    stealth_score += (df["volume_ratio_20_mean_5"] >= 1.15).fillna(False).astype(float) * 1.5
    stealth_score += (df["volume_ratio_20_mean_10"] >= 1.05).fillna(False).astype(float) * 1.0
    stealth_score += (df["Close"] >= df["ma75"]).fillna(False).astype(float) * 1.0
    stealth_score += (df["Close"] >= df["ma200"]).fillna(False).astype(float) * 0.5
    stealth_score += (df["ma200_slope_pct"] > 0).fillna(False).astype(float) * 1.0
    stealth_score += (df["ma75_slope_pct"] > 0).fillna(False).astype(float) * 1.0
    stealth_score += df["drawdown_from_60d_high_pct"].between(-12, 2).fillna(False).astype(float) * 1.0
    stealth_score += (df["lower_shadow_pct"] >= 35).fillna(False).astype(float) * 1.0
    stealth_score += df["reclaim_ma25_close"].fillna(False).astype(float) * 1.0
    stealth_score += df["reclaim_ma75_close"].fillna(False).astype(float) * 1.0
    stealth_score -= df["down_volume_spike"].fillna(False).astype(float) * 1.5
    stealth_score -= df["distribution_warning"].fillna(False).astype(float) * 1.5
    stealth_score -= (df["Close"] < df["ma75"]).fillna(False).astype(float) * 1.0
    stealth_score -= (df["change_20d_pct"] > 20).fillna(False).astype(float) * 0.5
    df["stealth_accumulation_score"] = stealth_score.round(2)
    df["stealth_accumulation_candidate"] = (
        (df["volume_ratio_20_mean_5"] >= 1.10)
        & (df["Close"] >= df["ma75"])
        & (df["ma200_slope_pct"] > 0)
        & (~df["down_volume_spike"])
        & (~df["distribution_warning"])
        & (df["drawdown_from_60d_high_pct"] > -15)
        & (df["stealth_accumulation_score"] >= 5.5)
    )
    return df


def passes_long_term_filter(latest: pd.Series, fundamentals: dict) -> bool:
    if latest["turnover"] < MIN_TURNOVER:
        return False

    if latest["Close"] < latest["ma75"]:
        return False

    if latest["ma25"] < latest["ma75"]:
        return False

    if pd.notna(latest["ma200"]) and latest["Close"] < latest["ma200"]:
        return False

    if latest["change_60d_pct"] < 0:
        return False

    if latest["gap_to_52w_high_pct"] > MAX_52W_HIGH_GAP_PCT:
        return False

    if latest["change_20d_pct"] > MAX_CHANGE_20D_PCT:
        return False

    if latest["change_60d_pct"] > MAX_CHANGE_60D_PCT:
        return False

    market_cap = fundamentals.get("market_cap")
    if market_cap is None or market_cap < MIN_MARKET_CAP:
        return False

    revenue_growth = fundamentals.get("revenue_growth_pct")
    if revenue_growth is None or revenue_growth < MIN_REVENUE_GROWTH_PCT:
        return False

    profit_margin = fundamentals.get("profit_margin_pct")
    if profit_margin is None or profit_margin < MIN_PROFIT_MARGIN_PCT:
        return False

    roe = fundamentals.get("roe_pct")
    if roe is None or roe < MIN_ROE_PCT:
        return False

    return True


def score_row(latest: pd.Series, fundamentals: dict) -> tuple[float, float, float, float]:
    trend_score = 0.0
    quality_score = 0.0
    strength_score = 0.0
    risk_penalty = 0.0

    if latest["Close"] >= latest["ma25"]:
        trend_score += 2.0
    if latest["Close"] >= latest["ma75"]:
        trend_score += 2.5
    if pd.notna(latest["ma200"]) and latest["Close"] >= latest["ma200"]:
        trend_score += 1.5
    if latest["ma25_slope_pct"] > 0:
        trend_score += 1.5
    if latest["ma75_slope_pct"] > 0:
        trend_score += 1.8
    if pd.notna(latest["ma200_slope_pct"]) and latest["ma200_slope_pct"] > 0:
        trend_score += 1.0

    revenue_growth = fundamentals.get("revenue_growth_pct") or 0.0
    profit_margin = fundamentals.get("profit_margin_pct") or 0.0
    roe = fundamentals.get("roe_pct") or 0.0
    current_ratio = fundamentals.get("current_ratio") or 0.0
    debt_to_equity = fundamentals.get("debt_to_equity")
    quality_score += min(revenue_growth, 30.0) * 0.11
    quality_score += min(profit_margin, 25.0) * 0.18
    quality_score += min(roe, 25.0) * 0.16
    if current_ratio >= 1.5:
        quality_score += 0.8

    strength_score += min(max(latest["change_20d_pct"], 0.0), 30.0) * 0.08
    strength_score += min(max(latest["change_60d_pct"], 0.0), 50.0) * 0.08
    gap_score = min(max(16.0 - latest["gap_to_52w_high_pct"], 0.0), 16.0) * 0.12
    if latest["gap_to_52w_high_pct"] < 2.0:
        gap_score -= 0.8
    strength_score += gap_score
    strength_score += min(max(latest["volume_ratio_20"], 0.0), 3.0) * 0.5

    if debt_to_equity is not None and debt_to_equity > 150:
        risk_penalty -= 1.5
    elif debt_to_equity is not None and debt_to_equity > 100:
        risk_penalty -= 0.7
    if latest["change_20d_pct"] > 18:
        risk_penalty -= (latest["change_20d_pct"] - 18) * 0.18
    if latest["change_60d_pct"] > 45:
        risk_penalty -= (latest["change_60d_pct"] - 45) * 0.06
    if latest["volume_ratio_20"] > 4:
        risk_penalty -= 0.8
    if latest["ma75_cross_200_recent_tight"]:
        strength_score += 1.0
    if latest["ma25_cross_75_recent_tight"]:
        strength_score += 0.6
    if latest["initial_trend_signal"]:
        strength_score += 3.0
    if latest["early_reversal_setup"]:
        strength_score += 2.5
    if latest["reversal_from_bearish_po"]:
        strength_score += 4.0
    if latest["pullback_candidate"]:
        strength_score += 2.0
    strength_score += min(max(latest["pullback_score"], 0.0), 10.0) * 0.15

    total = trend_score + quality_score + strength_score + risk_penalty
    return round(total, 2), round(trend_score, 2), round(quality_score, 2), round(strength_score + risk_penalty, 2)


def run():
    ensure_results_dirs()
    tickers = load_tickers()
    rows: list[dict] = []
    run_started_at = datetime.now()
    generated_at = run_started_at.isoformat(timespec="seconds")
    run_stamp = run_started_at.strftime("%Y%m%d_%H%M%S")
    screen_date = None

    for idx, row in tickers.iterrows():
        ticker = row["ticker"]
        name = row["name"]
        print(f"{idx + 1}/{len(tickers)} {ticker}")

        ticker_obj = None
        try:
            ticker_obj = yf.Ticker(ticker)
            hist = fetch_price_history(ticker_obj, ticker)
            if hist is None:
                continue

            fundamentals = fetch_fundamentals(ticker_obj, ticker)
            if fundamentals is None:
                continue

            hist = calc_indicators(hist)
            latest, latest_date = select_latest_completed_row(hist)

            if not passes_long_term_filter(latest, fundamentals):
                time.sleep(SLEEP_SEC)
                continue

            screen_date = latest_date if screen_date is None else max(screen_date, latest_date)
            score, trend_score, quality_score, strength_score = score_row(latest, fundamentals)

            rows.append(
                {
                    "run_date": latest_date.isoformat(),
                    "screen_version": LONG_TERM_SCREEN_VERSION,
                    "generated_at": generated_at,
                    "ticker": ticker,
                    "name": name,
                    "score": score,
                    "trend_score": trend_score,
                    "quality_score": quality_score,
                    "strength_score": strength_score,
                    "close": round(float(latest["raw_close"]), 3),
                    "turnover_million": round(float(latest["turnover_million"]), 3),
                    "market_cap_billion": round((fundamentals["market_cap"] or 0.0) / 1_000_000_000, 3),
                    "revenue_growth_pct": round(fundamentals["revenue_growth_pct"], 3),
                    "profit_margin_pct": round(fundamentals["profit_margin_pct"], 3),
                    "roe_pct": round(fundamentals["roe_pct"], 3),
                    "current_ratio": round(fundamentals["current_ratio"], 3) if fundamentals["current_ratio"] is not None else None,
                    "debt_to_equity": round(fundamentals["debt_to_equity"], 3) if fundamentals["debt_to_equity"] is not None else None,
                    "change_20d_pct": round(float(latest["change_20d_pct"]), 3),
                    "change_60d_pct": round(float(latest["change_60d_pct"]), 3),
                    "change_120d_pct": round(float(latest["change_120d_pct"]), 3),
                    "gap_to_52w_high_pct": round(float(latest["gap_to_52w_high_pct"]), 3),
                    "volume_ratio_20": round(float(latest["volume_ratio_20"]), 3),
                    "ma25_slope_pct": round(float(latest["ma25_slope_pct"]), 3),
                    "ma75_slope_pct": round(float(latest["ma75_slope_pct"]), 3),
                    "ma200_slope_pct": round(float(latest["ma200_slope_pct"]), 3) if pd.notna(latest["ma200_slope_pct"]) else None,
                    "close_vs_ma25_pct": round(float(latest["close_vs_ma25_pct"]), 3) if pd.notna(latest["close_vs_ma25_pct"]) else None,
                    "close_vs_ma75_pct": round(float(latest["close_vs_ma75_pct"]), 3) if pd.notna(latest["close_vs_ma75_pct"]) else None,
                    "close_vs_ma200_pct": round(float(latest["close_vs_ma200_pct"]), 3) if pd.notna(latest["close_vs_ma200_pct"]) else None,
                    "days_since_75gc200": int(latest["days_since_75gc200"]) if pd.notna(latest["days_since_75gc200"]) else None,
                    "days_since_perfect_order": int(latest["days_since_perfect_order"]) if pd.notna(latest["days_since_perfect_order"]) else None,
                    "touch_ma25_intraday": bool(latest["touch_ma25_intraday"]),
                    "touch_ma75_intraday": bool(latest["touch_ma75_intraday"]),
                    "reclaim_ma25_close": bool(latest["reclaim_ma25_close"]),
                    "reclaim_ma75_close": bool(latest["reclaim_ma75_close"]),
                    "failed_ma25_reclaim": bool(latest["failed_ma25_reclaim"]),
                    "failed_ma75_reclaim": bool(latest["failed_ma75_reclaim"]),
                    "support_reaction_ok": bool(latest["support_reaction_ok"]),
                    "ma25_pullback_candidate": bool(latest["ma25_pullback_candidate"]),
                    "ma75_pullback_candidate": bool(latest["ma75_pullback_candidate"]),
                    "down_volume_spike": bool(latest["down_volume_spike"]),
                    "pullback_score": round(float(latest["pullback_score"]), 3) if pd.notna(latest["pullback_score"]) else None,
                    "pullback_candidate": bool(latest["pullback_candidate"]),
                    "ma25_above_ma200": bool(latest["ma25_above_ma200"]),
                    "ma75_above_ma200": bool(latest["ma75_above_ma200"]),
                    "ma25_above_ma75": bool(latest["ma25_above_ma75"]),
                    "perfect_order": bool(latest["perfect_order"]),
                    "bearish_stack_recent": bool(latest["bearish_stack_recent"]),
                    "bearish_perfect_order_recent": bool(latest["bearish_perfect_order_recent"]),
                    "ma25_cross_200_today": bool(latest["ma25_cross_200_today"]),
                    "ma75_cross_200_today": bool(latest["ma75_cross_200_today"]),
                    "ma25_cross_200_recent": bool(latest["ma25_cross_200_recent"]),
                    "ma75_cross_200_recent": bool(latest["ma75_cross_200_recent"]),
                    "ma25_cross_75_today": bool(latest["ma25_cross_75_today"]),
                    "ma25_cross_75_recent_tight": bool(latest["ma25_cross_75_recent_tight"]),
                    "ma25_cross_200_recent_tight": bool(latest["ma25_cross_200_recent_tight"]),
                    "ma75_cross_200_recent_tight": bool(latest["ma75_cross_200_recent_tight"]),
                    "perfect_order_recent": bool(latest["perfect_order_recent"]),
                    "perfect_order_today": bool(latest["perfect_order_today"]),
                    "perfect_order_recent_tight": bool(latest["perfect_order_recent_tight"]),
                    "initial_trend_signal": bool(latest["initial_trend_signal"]),
                    "early_reversal_setup": bool(latest["early_reversal_setup"]),
                    "reversal_from_bearish_po": bool(latest["reversal_from_bearish_po"]),
                    "sector": fundamentals["sector"],
                    "industry": fundamentals["industry"],
                }
            )
        finally:
            close_ticker_session(ticker_obj)
        if (idx + 1) % 25 == 0:
            gc.collect()
        time.sleep(SLEEP_SEC)

    df = pd.DataFrame(rows)
    if df.empty:
        print("No long-term candidates")
        return

    df = df.sort_values(["score", "quality_score", "trend_score"], ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    output_df = df[
        [
            "run_date",
            "screen_version",
            "generated_at",
            "rank",
            "ticker",
            "name",
            "score",
            "trend_score",
            "quality_score",
            "strength_score",
            "close",
            "turnover_million",
            "market_cap_billion",
            "revenue_growth_pct",
            "profit_margin_pct",
            "roe_pct",
            "current_ratio",
            "debt_to_equity",
            "change_20d_pct",
            "change_60d_pct",
            "change_120d_pct",
            "gap_to_52w_high_pct",
            "volume_ratio_20",
            "ma25_slope_pct",
            "ma75_slope_pct",
            "ma200_slope_pct",
            "close_vs_ma25_pct",
            "close_vs_ma75_pct",
            "close_vs_ma200_pct",
            "days_since_75gc200",
            "days_since_perfect_order",
            "touch_ma25_intraday",
            "touch_ma75_intraday",
            "reclaim_ma25_close",
            "reclaim_ma75_close",
            "failed_ma25_reclaim",
            "failed_ma75_reclaim",
            "support_reaction_ok",
            "ma25_pullback_candidate",
            "ma75_pullback_candidate",
            "down_volume_spike",
            "pullback_score",
            "pullback_candidate",
            "ma25_above_ma200",
            "ma75_above_ma200",
            "ma25_above_ma75",
            "perfect_order",
            "bearish_stack_recent",
            "bearish_perfect_order_recent",
            "ma25_cross_200_today",
            "ma75_cross_200_today",
            "ma25_cross_200_recent",
            "ma75_cross_200_recent",
            "ma25_cross_75_today",
            "ma25_cross_75_recent_tight",
            "ma25_cross_200_recent_tight",
            "ma75_cross_200_recent_tight",
            "perfect_order_recent",
            "perfect_order_today",
            "perfect_order_recent_tight",
            "initial_trend_signal",
            "early_reversal_setup",
            "reversal_from_bearish_po",
            "sector",
            "industry",
        ]
    ].head(TOP_N_OUTPUT)
    latest_export_df = format_long_term_latest_output(output_df)
    history_export_df = format_long_term_output(output_df)
    gc_df = df[
        (
            df["reversal_from_bearish_po"]
            | (
                df["early_reversal_setup"]
                & df["days_since_75gc200"].notna()
                & (df["days_since_75gc200"] <= PERFECT_ORDER_LOOKBACK)
            )
            | df["initial_trend_signal"]
        )
        & df["close_vs_ma25_pct"].notna()
        & (df["close_vs_ma25_pct"] >= -1.0)
        & (df["close_vs_ma25_pct"] <= 8.0)
        & (~df["down_volume_spike"])
        & (df["ma25_slope_pct"] >= GC_MIN_MA25_SLOPE_PCT)
        & (df["ma75_slope_pct"] >= GC_MIN_MA75_SLOPE_PCT)
    ].copy()
    gc_df = gc_df.sort_values(
        ["reversal_from_bearish_po", "early_reversal_setup", "days_since_perfect_order", "days_since_75gc200", "score"],
        ascending=[False, False, True, True, False],
    ).reset_index(drop=True)
    gc_df["rank"] = gc_df.index + 1
    gc_output_df = gc_df[
        [
            "run_date",
            "screen_version",
            "generated_at",
            "rank",
            "ticker",
            "name",
            "reversal_from_bearish_po",
            "early_reversal_setup",
            "initial_trend_signal",
            "days_since_perfect_order",
            "days_since_75gc200",
            "close_vs_ma25_pct",
            "close_vs_ma75_pct",
            "bearish_perfect_order_recent",
            "perfect_order_today",
            "perfect_order_recent_tight",
            "perfect_order_recent",
            "ma25_cross_75_recent_tight",
            "ma25_cross_200_recent_tight",
            "ma75_cross_200_recent_tight",
            "score",
            "trend_score",
            "quality_score",
            "strength_score",
            "close",
            "turnover_million",
            "market_cap_billion",
            "revenue_growth_pct",
            "profit_margin_pct",
            "roe_pct",
            "change_20d_pct",
            "change_60d_pct",
            "gap_to_52w_high_pct",
            "volume_ratio_20",
            "ma25_slope_pct",
            "ma75_slope_pct",
            "ma25_above_ma75",
            "ma25_above_ma200",
            "ma75_above_ma200",
            "perfect_order",
            "sector",
            "industry",
        ]
    ].head(TOP_N_GC_OUTPUT)
    gc_export_df = format_long_term_gc_output(gc_output_df)

    if screen_date is None:
        screen_date = pd.Timestamp(output_df["run_date"].max()).date()

    latest_output_path = _latest_output_path()
    latest_gc_output_path = _latest_gc_output_path()
    dated_output_path = LONG_TERM_WATCHLISTS_DIR / f"{screen_date.isoformat()}_{LONG_TERM_SCREEN_VERSION}_{run_stamp}.csv"
    gc_watchlists_dir = _gc_watchlists_dir()
    gc_watchlists_dir.mkdir(parents=True, exist_ok=True)
    dated_gc_output_path = gc_watchlists_dir / f"{screen_date.isoformat()}_{LONG_TERM_SCREEN_VERSION}_{run_stamp}.csv"
    latest_export_df.to_csv(latest_output_path, index=False, encoding="utf-8-sig")
    history_export_df.to_csv(dated_output_path, index=False, encoding="utf-8-sig")
    gc_export_df.to_csv(latest_gc_output_path, index=False, encoding="utf-8-sig")
    gc_export_df.to_csv(dated_gc_output_path, index=False, encoding="utf-8-sig")

    print("\n==== Long Term Watchlist ====")
    print(latest_export_df.to_string(index=False))
    print(f"\nCSV出力完了: {latest_output_path.name}")
    print(f"履歴保存完了: {dated_output_path}")
    print(f"GC専用出力完了: {latest_gc_output_path.name}")
    print(f"GC専用履歴保存完了: {dated_gc_output_path}")


if __name__ == "__main__":
    run()
