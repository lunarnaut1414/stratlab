"""opus-1 mutation of gen10_sp500_infr_momentum (IS 1.33, h1=1.02/h2=1.71).

Parent: stratlab/strategies/arena/gen_10/sp500_infr_momentum.py

Hypothesis (opus-1, gen_10):
    Parent SP500 information-ratio momentum routes to IEF (7-10y Treasuries)
    when SPY 200d-bear OR no qualifying stocks. IEF is a pure intermediate-
    duration bet whose calm-VIX-IS performance may not extend to OOS regimes
    with elevated rate volatility.

    This variant replaces the IEF defensive with a TLT 60pct + GLD 37pct
    blend.  TLT supplies long-duration tail hedge (different from IEF's
    intermediate-duration) and GLD supplies a non-correlated real-asset
    hedge.  The combination breaks the bond-duration correlation cluster
    in defensive sleeves on the leaderboard (most of which use IEF/TLT
    alone) and gives a defensive sleeve whose loss-mode-corr to other
    strategies' bear branches should be materially lower.

    Risk-on branch unchanged: IR(63d) >= 0.5 filter, top-15 by 126d momentum,
    inverse-vol weighted, portfolio 13pct vol-target (50-97pct), biweekly
    rebalance.

    Per gen_8 OOS lesson: keep the alpha signal, change the defensive sleeve
    for OOS diversification. The h2-up profile of parent (1.71 in second
    half of IS) is the kind of mechanism that suggests genuine improving
    edge, so preserving its alpha selection is high-value.

Diversification rationale:
    - Parent corr 0.841 to top-5. Defensive engagements differ in 2010-2018
      because IEF +TLT correlated strongly during rate cycles but GLD often
      decorrelates from both.  Daily-return divergence on defensive days
      pulls corr down without changing risk-on alpha.
    - TLT+GLD blend is unused on the leaderboard as a defensive — the closest
      existing strategy is the IR-momentum parent itself (IEF) and the
      RSP-breadth strategies (SPY+TLT or TLT).  Combining 60pct TLT +
      37pct GLD yields a different sleeve than any of them.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10
MOM_LOOKBACK = 126
IR_WINDOW = 63
IR_THRESHOLD = 0.8       # tightened from parent's 0.5 (stricter alpha bar)
VOL_WINDOW_INDIV = 21
SPY_TREND_WINDOW = 200
TOP_K = 10               # narrowed from parent's 15 (more concentrated)
VOL_TARGET = 0.13
PORT_VOL_WINDOW = 21
EXPOSURE_MIN = 0.50
EXPOSURE_MAX = 0.97
ANNUALIZATION = 252
# Defensive sleeve weights
DEFENSIVE_TLT_W = 0.60
DEFENSIVE_GLD_W = 0.37


def _compute_ir(stock_prices: np.ndarray, spy_prices: np.ndarray, window: int) -> float:
    """Information ratio for stock relative to SPY over last `window` bars.

    IR = idiosyncratic cumulative return / (residual std * sqrt(window)).
    """
    n = min(len(stock_prices), len(spy_prices))
    need = window + 1
    if n < need:
        return float("nan")

    s = stock_prices[-need:]
    m = spy_prices[-need:]

    s_ret = np.log(s[1:] / s[:-1])
    m_ret = np.log(m[1:] / m[:-1])

    if len(s_ret) < window or len(m_ret) < window:
        return float("nan")

    s_ret = s_ret[-window:]
    m_ret = m_ret[-window:]
    m_var = float(np.var(m_ret))
    if m_var < 1e-12:
        return float("nan")
    beta = float(np.cov(s_ret, m_ret)[0, 1] / m_var)

    residuals = s_ret - beta * m_ret
    idio_ret = float(np.sum(residuals))
    idio_vol = float(np.std(residuals))

    if idio_vol < 1e-10 or not np.isfinite(idio_vol):
        return float("nan")

    ir = (idio_ret / idio_vol) / np.sqrt(window)
    return float(ir)


class Opus1InfRTltGldDefensive(Strategy):
    """SP500 IR-filtered 126d momentum, TLT+GLD defensive blend."""

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = MOM_LOOKBACK + IR_WINDOW + PORT_VOL_WINDOW + 10
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
            if "TLT" in live:
                target["TLT"] = DEFENSIVE_TLT_W
            if "GLD" in live:
                target["GLD"] = DEFENSIVE_GLD_W

        if not spy_bull:
            _go_defensive()
        else:
            need = MOM_LOOKBACK + IR_WINDOW + 5
            prices = ctx.closes_window(need)
            if len(prices) < need - 5:
                return []

            spy_prices_arr = spy_close.values[-need:]

            scores: dict[str, float] = {}
            inv_vols: dict[str, float] = {}

            for sym in prices.columns:
                if sym in ("SPY", "TLT", "GLD"):
                    continue
                col = prices[sym].dropna()
                if len(col) < MOM_LOOKBACK + 2:
                    continue

                p_end = float(col.iloc[-1])
                p_start = float(col.iloc[-MOM_LOOKBACK])
                if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                    continue
                ret = p_end / p_start - 1.0
                if not np.isfinite(ret):
                    continue

                stock_prices_arr = col.values[-need:]
                spy_arr = spy_prices_arr[-len(stock_prices_arr):]
                ir_val = _compute_ir(stock_prices_arr, spy_arr, IR_WINDOW)
                if not np.isfinite(ir_val) or ir_val < IR_THRESHOLD:
                    continue

                tail = col.values[-(VOL_WINDOW_INDIV + 1):]
                if len(tail) < VOL_WINDOW_INDIV + 1:
                    continue
                logr = np.log(tail[1:] / tail[:-1])
                rv = float(np.std(logr))
                if rv <= 1e-6 or not np.isfinite(rv):
                    continue

                # Mutation: rank by IR itself (alpha strength), not 126d raw return.
                # The IR-filtered subset is selected; we score by IR magnitude so the
                # highest-alpha names lead — different stock ordering than parent's
                # 126d return ranking, breaking corr while preserving IR mechanism.
                scores[sym] = ir_val
                inv_vols[sym] = 1.0 / rv

            if len(scores) < 5:
                _go_defensive()
            else:
                k = min(TOP_K, len(scores))
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]

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

                iv_sum = sum(inv_vols[s] for s in ranked)
                if iv_sum <= 0:
                    return []
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
    return sp500_tickers() + ["SPY", "TLT", "GLD"]


UNIVERSE = _universe

NAME = "opus1_infr_tlt_gld_defensive"
HYPOTHESIS = (
    "opus-1 mutation of gen10_sp500_infr_momentum (IS 1.33, h2=1.71): two-axis mutation — "
    "(1) defensive sleeve switched from IEF to TLT 60pct + GLD 37pct blend (long-duration "
    "+ gold), (2) risk-on RANKING SIGNAL switched from 126d raw return to IR magnitude "
    "(selects highest stock-specific alpha names, not highest beta-weighted return); "
    "IR>=0.8 filter, top-10 inverse-vol, 13pct vol-target; SPY 200d bear -> TLT+GLD — "
    "rank-by-alpha picks a structurally different stock subset than rank-by-momentum"
)

STRATEGY = Opus1InfRTltGldDefensive()
