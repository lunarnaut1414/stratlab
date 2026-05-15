"""SP500 momentum with per-stock information ratio quality filter.

Hypothesis (sonnet-9, gen_10):
    The Information Ratio (IR) measures stock-specific alpha relative to its
    own idiosyncratic (residual) risk. Stocks with high IR have genuine
    alpha-generating mechanisms beyond just riding market beta. Filtering to
    stocks with IR > 0.5 (idiosyncratic Sharpe > 0.5) before ranking by
    126d momentum selects names with strong risk-adjusted stock-specific returns,
    not just momentum from high-beta market exposure.

    IR = (stock 63d return - beta * SPY 63d return) / (63d residual vol * sqrt(63))

    Design:
      - Estimate beta (63d returns regression against SPY).
      - Compute idiosyncratic return (stock - beta * SPY).
      - Compute idiosyncratic vol (residual std dev).
      - IR = idiosyncratic Sharpe: idio_return / (idio_vol * sqrt(63)).
      - Only rank stocks with IR > 0.5.
      - Rank qualifying stocks by 126d momentum.
      - Hold top-15 inverse-vol weighted.
      - Portfolio vol-target (13% ann, 21d window) scales aggregate exposure 50-97%.
      - SPY 200d SMA outer bear gate to IEF.
      - Biweekly rebalance (10 bars).

Diversification angle vs leaderboard:
  - gen7_sp500_idiosyncratic_momentum (IS 1.20, OOS 0.70): ranks by idiosyncratic
    return (stock - beta*SPY). This strategy uses IR as a FILTER (minimum bar),
    then ranks by raw 126d momentum — different mechanism from ranking by idio return.
  - gen9_sp500_rsi_quality_momentum (OOS 0.88): RSI floor — directional oscillator,
    not risk-adjusted stock-specific quality.
  - gen6_nearhi_momentum_quality: price position relative to high — not IR.
  - No leaderboard strategy uses IR as a threshold filter before momentum ranking.

OOS resilience rationale:
  - IR > 0.5 filter is structurally regime-invariant: stock-specific alpha (versus
    market beta) doesn't depend on whether VIX is calm or stressed.
  - Portfolio vol-targeting provides automatic deleveraging in high-vol regimes
    without a VIX-level gate that can become miscalibrated OOS.
  - Selecting stocks with genuine alpha (not just beta exposure) improves
    portfolio resilience during periods when market beta is penalized.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10        # biweekly
MOM_LOOKBACK = 126          # ~6 months for momentum ranking
IR_WINDOW = 63              # 63-day window for IR calculation
IR_THRESHOLD = 0.5          # minimum information ratio
VOL_WINDOW_INDIV = 21       # for inverse-vol weights
SPY_TREND_WINDOW = 200      # outer gate
TOP_K = 15
VOL_TARGET = 0.13           # 13% annualized portfolio vol target
PORT_VOL_WINDOW = 21        # realized portfolio vol lookback
EXPOSURE_MIN = 0.50
EXPOSURE_MAX = 0.97
ANNUALIZATION = 252


def _compute_ir(stock_prices: np.ndarray, spy_prices: np.ndarray, window: int) -> float:
    """Compute information ratio for stock relative to SPY over last `window` bars.

    IR = idiosyncratic return / (idio residual vol * sqrt(window))
    Returns NaN if insufficient data.
    """
    n = min(len(stock_prices), len(spy_prices))
    need = window + 1
    if n < need:
        return float("nan")

    s = stock_prices[-need:]
    m = spy_prices[-need:]

    # Log returns
    s_ret = np.log(s[1:] / s[:-1])
    m_ret = np.log(m[1:] / m[:-1])

    if len(s_ret) < window or len(m_ret) < window:
        return float("nan")

    # OLS beta estimate
    s_ret = s_ret[-window:]
    m_ret = m_ret[-window:]
    m_var = float(np.var(m_ret))
    if m_var < 1e-12:
        return float("nan")
    beta = float(np.cov(s_ret, m_ret)[0, 1] / m_var)

    # Idiosyncratic (residual) returns
    residuals = s_ret - beta * m_ret
    idio_ret = float(np.sum(residuals))   # cumulative idio return over window
    idio_vol = float(np.std(residuals))

    if idio_vol < 1e-10 or not np.isfinite(idio_vol):
        return float("nan")

    # Information ratio = annualized idio return / annualized idio vol
    ir = (idio_ret / idio_vol) / np.sqrt(window)
    return float(ir)


class SP500InfoRatioMomentum(Strategy):
    """SP500 126d momentum filtered by IR(63d) >= 0.5; inverse-vol weighted;
    portfolio vol-targeting; SPY 200d gate; IEF defensive; biweekly rebalance.
    """

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = MOM_LOOKBACK + IR_WINDOW + PORT_VOL_WINDOW + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % REBALANCE_EVERY != 0:
            return []

        # SPY 200d SMA outer gate
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

        if not spy_bull:
            if "IEF" in closes_now.index:
                target["IEF"] = EXPOSURE_MAX
        else:
            need = MOM_LOOKBACK + IR_WINDOW + 5
            prices = ctx.closes_window(need)
            if len(prices) < need - 5:
                return []

            # Get SPY prices for IR calculation
            spy_prices_arr = spy_close.values[-need:]

            scores: dict[str, float] = {}
            inv_vols: dict[str, float] = {}

            for sym in prices.columns:
                if sym in ("SPY", "IEF"):
                    continue
                col = prices[sym].dropna()
                if len(col) < MOM_LOOKBACK + 2:
                    continue

                # 126d momentum
                p_end = float(col.iloc[-1])
                p_start = float(col.iloc[-MOM_LOOKBACK])
                if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                    continue
                ret = p_end / p_start - 1.0
                if not np.isfinite(ret):
                    continue

                # IR filter
                stock_prices_arr = col.values[-need:]
                spy_arr = spy_prices_arr[-len(stock_prices_arr):]
                ir_val = _compute_ir(stock_prices_arr, spy_arr, IR_WINDOW)
                if not np.isfinite(ir_val) or ir_val < IR_THRESHOLD:
                    continue

                # Inverse-vol weight
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
                if "IEF" in closes_now.index:
                    target["IEF"] = EXPOSURE_MAX
            else:
                k = min(TOP_K, len(scores))
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]

                # Portfolio vol-targeting
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

        # Build orders
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
    return sp500_tickers() + ["SPY", "IEF"]


UNIVERSE = _universe

NAME = "sp500_infr_momentum"
HYPOTHESIS = (
    "SP500 top-15 by 126d momentum with per-stock 126d information ratio filter: "
    "only hold stocks whose 63d return / 63d residual-vol (idiosyncratic Sharpe vs SPY beta) "
    "exceeds 0.5 (strong stock-specific alpha); inverse-vol weighted; portfolio vol-target "
    "(13% ann, 21d window); SPY 200d outer gate to IEF; biweekly rebalance — "
    "information-ratio screen selects stocks with genuine alpha not just market-beta lift, "
    "orthogonal to RSI/BB and raw-return quality screens"
)

STRATEGY = SP500InfoRatioMomentum()
