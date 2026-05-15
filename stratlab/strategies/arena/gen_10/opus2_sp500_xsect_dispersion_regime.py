"""SP500 cross-sectional return dispersion regime gate.

Hypothesis (opus-2, gen_10 gap_finder):
    Compute 21d cross-sectional standard deviation of SP500 stock 21d-returns
    each rebalance day:
        dispersion_t = std_i ( r_{i, t-21:t} ) across SP500 stocks.

    Regime gate uses the 63d rolling median of that dispersion series:
    - HIGH dispersion (cur >= 63d median): "stock-picking environment" — wide
      cross-section spread means momentum ranking has signal.
      Hold top-15 SP500 by 126d momentum, inverse-vol weighted.
    - LOW dispersion  (cur <  63d median): "correlated / mega-cap-led market" —
      stocks moving together, stock-picking edge compressed.
      Hold SPY 60% + TLT 37% blend (broad market + duration hedge).
    SPY 200d outer bear gate to TLT.  Biweekly rebalance.

Why this is an OPEN frontier (phase2_brief gap):
  - Cross-sectional dispersion of *individual SP500 stocks* has NOT been used
    as a regime gate in any prior arena gen (gens 5-10).
  - Most-similar prior work: gen9_sector_dispersion_gate uses **cross-SECTOR**
    dispersion (7 XL* ETF returns) — a coarser, between-sector signal.
    This strategy uses **within-SP500 cross-stock** dispersion — much more
    granular, captures internal-market decorrelation directly.
  - The 63d rolling median threshold self-adapts to changing baseline
    dispersion (post-COVID baseline is structurally higher), so no level
    overfit.

  - Theoretical rationale: when SP500 dispersion is high, alpha-momentum
    factors have signal-to-noise (wide spread between winners and losers).
    When dispersion compresses, the index moves as one block — mega-cap
    factors dominate and stock picking degrades. Bond defensive blend is
    classic risk-balanced when stocks aren't differentiating.

Distinct from:
  - gen9_sector_dispersion_gate: BETWEEN-sector dispersion (7 XL* ETFs)
  - gen10_rsp_breadth_regime_sp500: RSP-vs-SPY return SPREAD (breadth signal,
    not dispersion)
  - All quality-filter momentum variants: NO regime gate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10
RETURN_WINDOW = 21          # per-stock return window
DISPERSION_HISTORY = 63     # rolling median window of cross-sectional std
MOM_LOOKBACK = 126
VOL_WINDOW = 21
SPY_TREND_WINDOW = 200
TOP_K = 15
EXPOSURE = 0.97
W_SPY_LOW = 0.60            # low-dispersion regime defensive blend
W_TLT_LOW = 0.37


class SP500XSectDispersionRegime(Strategy):
    """Cross-stock dispersion within SP500 gates momentum vs balanced blend."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        return_window: int = RETURN_WINDOW,
        dispersion_history: int = DISPERSION_HISTORY,
        mom_lookback: int = MOM_LOOKBACK,
        vol_window: int = VOL_WINDOW,
        spy_trend_window: int = SPY_TREND_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            return_window=return_window,
            dispersion_history=dispersion_history,
            mom_lookback=mom_lookback,
            vol_window=vol_window,
            spy_trend_window=spy_trend_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.return_window = int(return_window)
        self.dispersion_history = int(dispersion_history)
        self.mom_lookback = int(mom_lookback)
        self.vol_window = int(vol_window)
        self.spy_trend_window = int(spy_trend_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def _compute_dispersion_series(self, prices: pd.DataFrame) -> tuple[float, float] | None:
        """Return (current_dispersion, median_dispersion) over last `dispersion_history` days.

        For each of the last `dispersion_history+1` end-bars, compute the
        cross-sectional std of `return_window`-day per-stock returns.
        """
        n_rows = len(prices)
        need = self.return_window + self.dispersion_history + 1
        if n_rows < need:
            return None

        # Only use columns with enough non-NaN data over the required window
        sub = prices.iloc[-(self.return_window + self.dispersion_history + 1):]

        disp_series: list[float] = []
        # We iterate over self.dispersion_history+1 historical end-points.
        # For each end-point t, compute return_t = price_t / price_{t - return_window} - 1
        # then take std across stocks.
        for k in range(self.dispersion_history + 1):
            # offset from the latest bar
            end_offset = self.dispersion_history - k  # 0 = today, dispersion_history = oldest
            end_idx = -1 - end_offset
            start_idx = end_idx - self.return_window
            try:
                p_end = sub.iloc[end_idx]
                p_start = sub.iloc[start_idx]
            except IndexError:
                continue
            ret = (p_end / p_start) - 1.0
            ret = ret.replace([np.inf, -np.inf], np.nan).dropna()
            if len(ret) < 50:
                continue
            disp = float(ret.std())
            if np.isfinite(disp):
                disp_series.append(disp)

        if len(disp_series) < max(20, self.dispersion_history // 3):
            return None
        cur_disp = disp_series[-1]
        median_disp = float(np.median(disp_series[:-1]))  # exclude current from baseline
        return cur_disp, median_disp

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.spy_trend_window, self.return_window + self.dispersion_history + self.mom_lookback) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.spy_trend_window + 2:
            return []
        spy_sma = float(spy_close.iloc[-self.spy_trend_window:].mean())
        spy_bull = float(spy_close.iloc[-1]) > spy_sma

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not spy_bull:
            if "TLT" in closes_now.index:
                target["TLT"] = self.exposure
        else:
            need = max(
                self.mom_lookback + self.vol_window + 2,
                self.return_window + self.dispersion_history + 2,
            )
            prices = ctx.closes_window(need)
            # Drop non-stock columns for dispersion compute
            cols_keep = [c for c in prices.columns if c not in ("SPY", "TLT", "IEF", "^OVX", "^MOVE", "^VIX")]
            disp_input = prices[cols_keep] if cols_keep else prices

            disp_pair = self._compute_dispersion_series(disp_input)

            high_dispersion = True  # default to risk-on if signal unavailable
            if disp_pair is not None:
                cur_disp, median_disp = disp_pair
                high_dispersion = cur_disp >= median_disp

            if not high_dispersion:
                # Low dispersion: defensive blend
                if "SPY" in closes_now.index:
                    target["SPY"] = W_SPY_LOW
                if "TLT" in closes_now.index:
                    target["TLT"] = W_TLT_LOW
            else:
                # High dispersion: SP500 momentum top-15 inverse-vol
                scores: dict[str, float] = {}
                inv_vols: dict[str, float] = {}
                for sym in disp_input.columns:
                    col = disp_input[sym].dropna()
                    if len(col) < self.mom_lookback + 1:
                        continue
                    p_now = float(col.iloc[-1])
                    p_then = float(col.iloc[-self.mom_lookback])
                    if p_then <= 0 or not np.isfinite(p_then) or not np.isfinite(p_now):
                        continue
                    ret = p_now / p_then - 1.0
                    if not np.isfinite(ret):
                        continue
                    tail = col.values[-(self.vol_window + 1):]
                    if len(tail) < self.vol_window + 1:
                        continue
                    logr = np.log(tail[1:] / tail[:-1])
                    rv = float(np.std(logr))
                    if rv <= 1e-6 or not np.isfinite(rv):
                        continue
                    scores[sym] = ret
                    inv_vols[sym] = 1.0 / rv

                if len(scores) < 5:
                    if "SPY" in closes_now.index:
                        target["SPY"] = self.exposure
                else:
                    k = min(self.top_k, len(scores))
                    ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                    iv_sum = sum(inv_vols[s] for s in ranked)
                    if iv_sum <= 0:
                        return []
                    for sym in ranked:
                        target[sym] = self.exposure * inv_vols[sym] / iv_sum

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
    return sp500_tickers() + ["SPY", "TLT"]


NAME = "opus2_sp500_xsect_dispersion_regime"
HYPOTHESIS = (
    "SP500 cross-sectional return dispersion regime gate: compute 21d cross-sectional std of SP500 "
    "stock 21d returns daily; when dispersion above 63d median (decorrelation/stock-picking regime) "
    "hold top-15 SP500 by 126d momentum inverse-vol weighted; when dispersion below median "
    "(correlated/mega-cap regime) hold SPY 60pct+TLT 37pct blend; SPY 200d outer bear gate to TLT; "
    "biweekly rebalance — within-SP500 stock dispersion is internal-market signal orthogonal to "
    "VIX timeseries vol and between-sector dispersion (gen9 sector_dispersion_gate)"
)

UNIVERSE = _universe

STRATEGY = SP500XSectDispersionRegime()
