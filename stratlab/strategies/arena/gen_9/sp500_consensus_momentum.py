"""SP500 Multi-Period Momentum Consensus — gen_9 sonnet-9

Hypothesis: Hold top-15 SP500 stocks where ALL THREE momentum windows
(21d, 63d, 126d) are simultaneously positive. This "consensus" requirement
eliminates:
  - Stocks in short-term reversal after long-term momentum (21d negative = fading)
  - Stocks with only recent short-term strength (126d negative = not sustained)
Only stocks with confirmed momentum across ALL three horizons pass through.
Inverse-vol weighted. SPY 200d SMA market gate. TLT defensive. Biweekly rebalance.

Rationale:
- Momentum works best when the signal is persistent across timeframes.
  A stock with 21d>0, 63d>0, 126d>0 is in a broadly confirmed uptrend, not
  a short spike or a mean-reversion trade.
- The consensus requirement is a quality filter: fewer candidates but higher
  signal quality per candidate. If fewer than 5 pass all 3 gates, fall back
  to IEF rather than forcing selection.
- Rank the qualifying stocks by composite score (equal-weight sum of
  normalized 21d, 63d, 126d returns) to order within the qualifying set.
- VIX-adaptive sizing: scale exposure between 0.60-0.97 based on VIX level
  (additional layer of risk management).

Differentiators vs leaderboard:
- Three-way momentum consensus (not used in any prior round)
- Composite ranking within consensus subset (not raw single-window)
- VIX exposure scaling combined with consensus filter
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10
WIN_SHORT = 21        # short-term momentum window
WIN_MID = 63          # mid-term momentum window
WIN_LONG = 126        # long-term momentum window (6 months)
TREND_WINDOW = 200    # SPY market gate
VOL_WINDOW = 21       # inverse-vol sizing window
TOP_K = 15
VIX_LOW = 15.0        # below this: max exposure
VIX_HIGH = 25.0       # above this: min exposure
EXPOSURE_HIGH = 0.97
EXPOSURE_LOW = 0.60

_SPY = "SPY"
_TLT = "TLT"
_IEF = "IEF"
_VIX = "^VIX"


class SP500ConsensusMomentum(Strategy):
    """SP500 consensus momentum — all 3 windows positive required."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        win_short: int = WIN_SHORT,
        win_mid: int = WIN_MID,
        win_long: int = WIN_LONG,
        trend_window: int = TREND_WINDOW,
        vol_window: int = VOL_WINDOW,
        top_k: int = TOP_K,
        vix_low: float = VIX_LOW,
        vix_high: float = VIX_HIGH,
        exposure_high: float = EXPOSURE_HIGH,
        exposure_low: float = EXPOSURE_LOW,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            win_short=win_short,
            win_mid=win_mid,
            win_long=win_long,
            trend_window=trend_window,
            vol_window=vol_window,
            top_k=top_k,
            vix_low=vix_low,
            vix_high=vix_high,
            exposure_high=exposure_high,
            exposure_low=exposure_low,
        )
        self.rebalance_every = int(rebalance_every)
        self.win_short = int(win_short)
        self.win_mid = int(win_mid)
        self.win_long = int(win_long)
        self.trend_window = int(trend_window)
        self.vol_window = int(vol_window)
        self.top_k = int(top_k)
        self.vix_low = float(vix_low)
        self.vix_high = float(vix_high)
        self.exposure_high = float(exposure_high)
        self.exposure_low = float(exposure_low)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.trend_window, self.win_long) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # --- SPY 200d market gate ---
        try:
            spy_hist = ctx.history(_SPY)
        except KeyError:
            return []
        if len(spy_hist) < self.trend_window + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.trend_window:
            return []
        spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
        spy_now = float(spy_close.iloc[-1])
        spy_bull = spy_now > spy_sma

        target: dict[str, float] = {}

        if not spy_bull:
            if _TLT in live:
                target[_TLT] = self.exposure_high
        else:
            # --- VIX-adaptive exposure scaling ---
            vix_val = 18.0  # default
            try:
                vix_hist = ctx.history(_VIX)
                if len(vix_hist) >= 2:
                    vix_val = float(vix_hist["close"].dropna().iloc[-1])
            except Exception:
                pass

            if vix_val <= self.vix_low:
                exposure = self.exposure_high
            elif vix_val >= self.vix_high:
                exposure = self.exposure_low
            else:
                frac = (vix_val - self.vix_low) / (self.vix_high - self.vix_low)
                exposure = self.exposure_high - frac * (
                    self.exposure_high - self.exposure_low
                )

            # --- Multi-period consensus momentum ---
            need = self.win_long + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.win_long + 1:
                if _IEF in live:
                    target[_IEF] = exposure
            else:
                # For each stock, require ALL THREE windows positive
                consensus_scores: dict[str, float] = {}
                vols: dict[str, float] = {}

                # Pre-compute SPY returns for normalization reference
                spy_r21 = spy_r63 = spy_r126 = 0.0
                if _SPY in prices.columns:
                    spy_col = prices[_SPY].dropna()
                    if len(spy_col) >= self.win_long + 1:
                        spy_r21 = float(
                            spy_col.iloc[-1] / spy_col.iloc[-self.win_short] - 1.0
                        )
                        spy_r63 = float(
                            spy_col.iloc[-1] / spy_col.iloc[-self.win_mid] - 1.0
                        )
                        spy_r126 = float(
                            spy_col.iloc[-1] / spy_col.iloc[-self.win_long] - 1.0
                        )

                # Stock-level stats: compute range of returns for normalization
                all_r21: list[float] = []
                all_r63: list[float] = []
                all_r126: list[float] = []

                raw: dict[str, tuple[float, float, float]] = {}
                for sym in prices.columns:
                    if sym in (_SPY, _TLT, _IEF):
                        continue
                    col = prices[sym].dropna()
                    if len(col) < self.win_long + 1:
                        continue
                    r21 = float(col.iloc[-1] / col.iloc[-self.win_short] - 1.0)
                    r63 = float(col.iloc[-1] / col.iloc[-self.win_mid] - 1.0)
                    r126 = float(col.iloc[-1] / col.iloc[-self.win_long] - 1.0)
                    if not (np.isfinite(r21) and np.isfinite(r63) and np.isfinite(r126)):
                        continue
                    raw[sym] = (r21, r63, r126)
                    all_r21.append(r21)
                    all_r63.append(r63)
                    all_r126.append(r126)

                # Normalize: z-score within cross-section for composite ranking
                def _zscore(arr: list[float]) -> tuple[float, float]:
                    if len(arr) < 2:
                        return 0.0, 1.0
                    m = float(np.mean(arr))
                    s = float(np.std(arr))
                    return m, max(s, 1e-6)

                mu21, sd21 = _zscore(all_r21)
                mu63, sd63 = _zscore(all_r63)
                mu126, sd126 = _zscore(all_r126)

                for sym, (r21, r63, r126) in raw.items():
                    # Consensus requirement: ALL THREE must be positive
                    if r21 <= 0.0 or r63 <= 0.0 or r126 <= 0.0:
                        continue

                    # Composite score: equal-weight z-scores across 3 windows
                    z21 = (r21 - mu21) / sd21
                    z63 = (r63 - mu63) / sd63
                    z126 = (r126 - mu126) / sd126
                    composite = (z21 + z63 + z126) / 3.0
                    if not np.isfinite(composite):
                        continue
                    consensus_scores[sym] = composite

                    # Inverse-vol sizing
                    col = prices[sym].dropna()
                    daily_rets = col.pct_change().dropna()
                    if len(daily_rets) >= self.vol_window:
                        rv = float(daily_rets.iloc[-self.vol_window:].std())
                        vols[sym] = max(rv, 1e-6)

                if len(consensus_scores) < 5:
                    # Not enough consensus stocks: IEF as risk-off
                    if _IEF in live:
                        target[_IEF] = exposure
                else:
                    k = min(self.top_k, len(consensus_scores))
                    ranked = sorted(
                        consensus_scores,
                        key=consensus_scores.__getitem__,
                        reverse=True,
                    )[:k]

                    inv_vols = {sym: 1.0 / vols.get(sym, 0.02) for sym in ranked}
                    total_inv = sum(inv_vols.values())
                    if total_inv <= 0:
                        per_w = exposure / len(ranked)
                        for sym in ranked:
                            if sym in live:
                                target[sym] = per_w
                    else:
                        for sym in ranked:
                            if sym in live:
                                target[sym] = exposure * (
                                    inv_vols[sym] / total_inv
                                )

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
    return sp500_tickers() + [_TLT, _IEF, _SPY, _VIX]


NAME = "sp500_consensus_momentum"
HYPOTHESIS = (
    "SP500 multi-period momentum consensus: hold top-15 SP500 stocks where ALL THREE "
    "momentum windows (21d, 63d, 126d) are positive simultaneously; ranked by composite "
    "equal-weight z-score across windows; inverse-vol weighted; VIX-adaptive exposure "
    "(0.97 at VIX<15, 0.60 at VIX>25); SPY 200d SMA gate; TLT defensive; IEF when "
    "insufficient consensus stocks; biweekly rebalance."
)

UNIVERSE = _universe

STRATEGY = SP500ConsensusMomentum()
