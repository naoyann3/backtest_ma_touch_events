from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import pandas as pd


JST = ZoneInfo("Asia/Tokyo")
MARKET_CLOSE_CUTOFF = time(16, 0)
PRICE_COLUMNS = ["Open", "High", "Low", "Close"]
REQUIRED_COLUMNS = PRICE_COLUMNS + ["Volume"]


def prepare_price_history(df: pd.DataFrame) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        return None

    hist = df.copy()
    if "Adj Close" not in hist.columns:
        hist["Adj Close"] = hist["Close"]
    if "Stock Splits" not in hist.columns:
        hist["Stock Splits"] = 0.0

    hist = hist[REQUIRED_COLUMNS + ["Adj Close", "Stock Splits"]].dropna(subset=REQUIRED_COLUMNS).copy()
    raw_close = hist["Close"].replace(0, pd.NA)
    hist["adjustment_factor"] = (hist["Adj Close"] / raw_close).replace([float("inf"), float("-inf")], pd.NA).fillna(1.0)

    for col in PRICE_COLUMNS:
        hist[f"raw_{col.lower()}"] = hist[col]
        hist[col] = hist[col] * hist["adjustment_factor"]

    hist["Stock Splits"] = hist["Stock Splits"].fillna(0.0)
    return hist


def select_latest_completed_row(df: pd.DataFrame, now: datetime | None = None) -> tuple[pd.Series, datetime.date]:
    if df.empty:
        raise ValueError("price history is empty")

    if now is None:
        now_jst = datetime.now(JST)
    elif now.tzinfo is None:
        now_jst = now.replace(tzinfo=JST)
    else:
        now_jst = now.astimezone(JST)
    latest_pos = len(df) - 1
    latest_ts = pd.Timestamp(df.index[latest_pos])
    latest_date = latest_ts.date()

    if latest_date >= now_jst.date() and now_jst.time() < MARKET_CLOSE_CUTOFF and len(df) >= 2:
        latest_pos -= 1
        latest_ts = pd.Timestamp(df.index[latest_pos])
        latest_date = latest_ts.date()

    return df.iloc[latest_pos], latest_date


def _normalized_index(index: pd.Index) -> pd.DatetimeIndex:
    normalized_index = pd.DatetimeIndex(index)
    if normalized_index.tz is not None:
        normalized_index = normalized_index.tz_localize(None)
    return normalized_index.normalize()


def _row_on_or_before(hist: pd.DataFrame, target_date: date) -> pd.Series | None:
    normalized_index = _normalized_index(hist.index)
    mask = normalized_index <= pd.Timestamp(target_date)
    if not mask.any():
        return None
    last_pos = mask.nonzero()[0][-1]
    return hist.iloc[last_pos]


def adjusted_entry_price(
    entry_price: float | None,
    entry_date: str,
    hist: pd.DataFrame,
    latest_row: pd.Series | None = None,
) -> float | None:
    if entry_price in (None, 0) or not entry_date:
        return entry_price

    try:
        entry_day = pd.Timestamp(entry_date).date()
    except Exception:
        return entry_price

    entry_row = _row_on_or_before(hist, entry_day)
    if entry_row is None:
        return entry_price

    current_row = latest_row if latest_row is not None else hist.iloc[-1]

    entry_factor = float(entry_row.get("adjustment_factor", 1.0) or 1.0)
    current_factor = float(current_row.get("adjustment_factor", 1.0) or 1.0)

    normalized_index = _normalized_index(hist.index)
    splits = hist.loc[normalized_index > pd.Timestamp(entry_day), "Stock Splits"]
    split_factor = 1.0
    for ratio in splits:
        try:
            ratio_float = float(ratio)
        except Exception:
            continue
        if ratio_float > 0:
            split_factor *= ratio_float

    if split_factor > 0 and abs(split_factor - 1.0) >= 0.01:
        return entry_price / split_factor

    if current_factor == 0:
        return entry_price

    implied_factor = entry_factor / current_factor
    if implied_factor > 0 and abs(implied_factor - 1.0) >= 0.05:
        return entry_price * implied_factor

    entry_close_basis = entry_row.get("raw_close", entry_row.get("Close"))
    if pd.notna(entry_close_basis) and float(entry_close_basis) > 0:
        basis_ratio = entry_price / float(entry_close_basis)
        if basis_ratio >= 1.8 or basis_ratio <= 0.55:
            return float(entry_close_basis)

    return entry_price


def detect_price_data_issue(latest: pd.Series, hist: pd.DataFrame) -> str:
    adjustment_range = hist["adjustment_factor"].tail(90)
    factor_jump = 0.0
    if not adjustment_range.empty:
        factor_min = float(adjustment_range.min())
        factor_max = float(adjustment_range.max())
        if factor_min > 0:
            factor_jump = factor_max / factor_min - 1.0

    close_vs_ma25_pct = latest.get("close_vs_ma25_pct")
    if close_vs_ma25_pct is None and latest.get("ma25") not in (None, 0) and pd.notna(latest.get("ma25")):
        close_vs_ma25_pct = (float(latest["Close"]) - float(latest["ma25"])) / float(latest["ma25"]) * 100

    extreme_drop = (
        close_vs_ma25_pct is not None
        and pd.notna(close_vs_ma25_pct)
        and float(close_vs_ma25_pct) <= -40
        and float(latest["change_20d_pct"]) <= -40
        and float(latest["drawdown_from_60d_high_pct"]) <= -40
    )
    if factor_jump >= 0.2:
        return "価格補正係数が大きく変化"
    if extreme_drop:
        return "価格データ要確認"
    return ""
