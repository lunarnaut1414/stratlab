"""opus-1 mutation of gen10_sp500_dual_quality_voltarget (IS 0.91).

Parent: stratlab/strategies/arena/gen_10/sp500_dual_quality_voltarget.py

Hypothesis (opus-1, gen_10):
    Parent uses TWO per-stock quality filters (RSI(14)>=40 AND price>own
    200d SMA) before ranking by 126d momentum, with portfolio 14pct vol-
    targeting and IEF defensive.  This variant adds a THIRD filter — ADX(14)
    >= 22 — to require directional trend strength on top of "not oversold"
    (RSI) and "above intermediate trend" (200d SMA).  ADX captures the
    magnitude of directional movement regardless of direction; combined
    with the existing two filters, a stock must be (a) not deeply oversold,
    (b) above its own long trend, AND (c) in a trend with measurable
    directional strength.  Stocks that pass all three are structurally
    higher-quality momentum candidates.

    Defensive sleeve changed from IEF (intermediate Treasuries) to SHY
    (1-3y, near-cash, near-zero duration) per gen_8 OOS lesson — break the
    bond-duration correlation cluster in defensive sleeves.

    Risk-on: top-15 (still), inverse-vol weighted, 14pct vol-target (50-97pct).

Diversification rationale:
    - The 3rd ADX filter changes WHICH stocks qualify on calm-trend days
      (filtering out stocks in slow drift), which should naturally diverge
      from parent's ranking on a meaningful number of days.
    - Defensive sleeve is SHY (no leaderboard strategy uses SHY-only
      defensive — most use IEF, TLT, or blends).
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10
MOM_LOOKBACK = 63        # shortened from parent's 126d for different ranking
RSI_WINDOW = 14
RSI_FLOOR = 45.0         # tightened from parent's 40
ADX_WINDOW = 14
ADX_FLOOR = 25.0         # standard ADX trend threshold
STOCK_TREND_WINDOW = 100 # 100d trend filter (shorter than parent's 200d SMA)
VOL_WINDOW_INDIV = 21
SPY_TREND_WINDOW = 200
TOP_K = 10               # narrower than parent's 15
VOL_TARGET = 0.12        # tighter than parent's 14
PORT_VOL_WINDOW = 30
EXPOSURE_MIN = 0.50
EXPOSURE_MAX = 0.97
ANNUALIZATION = 252


def _compute_rsi(prices: np.ndarray, window: int) -> float:
    if len(prices) < window + 1:
        return float("nan")
    deltas = np.diff(prices[-(window + 1):])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = float(np.mean(gains))
    avg_loss = float(np.mean(losses))
    if avg_loss < 1e-10:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _compute_adx_from_closes(prices: np.ndarray, window: int) -> float:
    """Approximate ADX from close-only series (no OHLC available in ctx.closes_window).

    Use absolute-day-return-direction as proxy for +DM/-DM:
        +DM = abs(ret) if ret>0 else 0
        -DM = abs(ret) if ret<0 else 0
        TR  = abs(ret)
    Then ADX = Wilder's-smooth(|+DI - -DI| / (+DI + -DI)) where DI = DM/TR EMA.

    This is a close-only proxy. Not the textbook ADX (which uses high/low/close)
    but captures the same trend-strength concept: ratio of net directional
    movement to total movement.
    """
    n = window * 2 + 5
    if len(prices) < n:
        return float("nan")
    p = prices[-n:]
    rets = np.diff(p) / p[:-1]
    if len(rets) < window * 2:
        return float("nan")

    plus_dm = np.where(rets > 0, np.abs(rets), 0.0)
    minus_dm = np.where(rets < 0, np.abs(rets), 0.0)
    tr = np.abs(rets) + 1e-12

    # Wilder smoothing approximated via EMA of length=window
    def _wilder_ema(x: np.ndarray, w: int) -> np.ndarray:
        out = np.zeros_like(x)
        out[0] = x[0]
        alpha = 1.0 / w
        for i in range(1, len(x)):
            out[i] = (1 - alpha) * out[i - 1] + alpha * x[i]
        return out

    plus_di = _wilder_ema(plus_dm, window) / _wilder_ema(tr, window) * 100.0
    minus_di = _wilder_ema(minus_dm, window) / _wilder_ema(tr, window) * 100.0
    denom = plus_di + minus_di
    denom = np.where(denom < 1e-9, 1e-9, denom)
    dx = np.abs(plus_di - minus_di) / denom * 100.0
    adx = _wilder_ema(dx, window)
    return float(adx[-1])


class Opus1TripleQualityShyDefensive(Strategy):
    """SP500 126d momentum with TRIPLE per-stock quality (RSI>=40 AND price>200d-SMA
    AND ADX>=22); inverse-vol; portfolio 14pct vol-target; SHY defensive sleeve.
    """

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = MOM_LOOKBACK + max(STOCK_TREND_WINDOW, ADX_WINDOW * 3) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % REBALANCE_EVERY != 0:
            return []

        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < SPY_TREND_WINDOW + 2:
            return []
        spy_sma = float(spy_close.iloc[-SPY_TREND_WINDOW:].mean())
        spy_bull = float(spy_close.iloc[-1]) > spy_sma

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if p > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        def _go_defensive() -> None:
            if "SHY" in live:
                target["SHY"] = EXPOSURE_MAX

        if not spy_bull:
            _go_defensive()
        else:
            need = max(MOM_LOOKBACK, STOCK_TREND_WINDOW) + RSI_WINDOW + ADX_WINDOW * 3 + 5
            prices = ctx.closes_window(need)
            if len(prices) < need - 10:
                return []

            scores: dict[str, float] = {}
            inv_vols: dict[str, float] = {}

            for sym in prices.columns:
                col = prices[sym].dropna()
                if len(col) < MOM_LOOKBACK + 2:
                    continue

                # Filter 1: RSI >= 40
                rsi_val = _compute_rsi(col.values, RSI_WINDOW)
                if not np.isfinite(rsi_val) or rsi_val < RSI_FLOOR:
                    continue

                # Filter 2: price above own 200d SMA
                if len(col) < STOCK_TREND_WINDOW:
                    continue
                stock_sma = float(col.iloc[-STOCK_TREND_WINDOW:].mean())
                stock_price = float(col.iloc[-1])
                if stock_price < stock_sma:
                    continue

                # Filter 3: ADX(14) >= 22 (close-only proxy)
                adx_val = _compute_adx_from_closes(col.values, ADX_WINDOW)
                if not np.isfinite(adx_val) or adx_val < ADX_FLOOR:
                    continue

                # 126d momentum score
                p_end = float(col.iloc[-1])
                p_start = float(col.iloc[-MOM_LOOKBACK])
                if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                    continue
                ret = p_end / p_start - 1.0
                if not np.isfinite(ret):
                    continue

                tail = col.values[-(VOL_WINDOW_INDIV + 1):]
                if len(tail) < VOL_WINDOW_INDIV + 1:
                    continue
                logr = np.log(tail[1:] / tail[:-1])
                rv = float(np.std(logr))
                if rv <= 1e-6 or not np.isfinite(rv):
                    continue

                scores[sym] = ret
                inv_vols[sym] = 1.0 / rv

            if len(scores) < 5:
                _go_defensive()
            else:
                k = min(TOP_K, len(scores))
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                iv_sum = sum(inv_vols[s] for s in ranked)
                if iv_sum <= 0:
                    return []

                vol_prices = ctx.closes_window(PORT_VOL_WINDOW + 5)
                port_rets = []
                n_rows = len(vol_prices)
                for row_idx in range(1, n_rows):
                    row_ret = 0.0
                    count = 0
                    for sym in ranked:
                        if sym not in vol_prices.columns:
                            continue
                        p_now = vol_prices[sym].iloc[row_idx]
                        p_prev = vol_prices[sym].iloc[row_idx - 1]
                        if np.isfinite(p_now) and np.isfinite(p_prev) and p_prev > 0:
                            row_ret += np.log(float(p_now) / float(p_prev))
                            count += 1
                    if count > 0:
                        port_rets.append(row_ret / count)

                if len(port_rets) >= 10:
                    daily_vol = float(np.std(port_rets))
                    annual_vol = daily_vol * np.sqrt(ANNUALIZATION)
                    scale = VOL_TARGET / annual_vol if annual_vol > 1e-6 else 1.0
                    exposure = float(np.clip(scale, EXPOSURE_MIN, EXPOSURE_MAX))
                else:
                    exposure = EXPOSURE_MAX

                for sym in ranked:
                    target[sym] = exposure * inv_vols[sym] / iv_sum

        orders: list[Order] = []
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        for sym, weight in target.items():
            price = live.get(sym)
            if not price or price <= 0:
                continue
            tgt_shares = int(equity * weight / price)
            cur = int(ctx.position(sym).size)
            delta = tgt_shares - cur
            if abs(delta) < 1:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            orders.append(Order(side=side, size=abs(delta), symbol=sym))

        return orders


def _universe() -> list[str]:
    from stratlab.data.universe import sp500_tickers
    return sp500_tickers() + ["SPY", "SHY"]


UNIVERSE = _universe

NAME = "opus1_triple_quality_shy_defensive"
HYPOTHESIS = (
    "opus-1 mutation of gen10_sp500_dual_quality_voltarget (IS 0.91): triple-axis mutation — "
    "(1) add THIRD filter ADX(14)>=25 on top of RSI(14)>=45 (tighter) AND price>100d SMA "
    "(shorter than parent's 200d); (2) ranking horizon 63d not 126d (different stock subset); "
    "(3) defensive sleeve SHY (near-cash) instead of IEF; top-10 inverse-vol, 12pct vol-target; "
    "SPY 200d bear -> SHY — co-mutation of selectivity AND defensive AND momentum-horizon "
    "breaks correlation cluster while preserving regime-invariant mechanism"
)

STRATEGY = Opus1TripleQualityShyDefensive()
