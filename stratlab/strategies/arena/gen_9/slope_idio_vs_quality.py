"""gen_9 sonnet-3 — Long-End Slope Regime Switches Between Idiosyncratic and Quality Momentum

Hypothesis: The 30Y-10Y long-end slope (TYX-TNX vs its 200d MA) as macro regime switch
BETWEEN two proven OOS-robust SP500 stock-selection modes:

  Steep long-end slope (TYX-TNX > 200d MA of TYX-TNX):
    - Duration risk premium expanding = growth/inflation favorable
    - Use IDIOSYNCRATIC momentum: rank by 63d beta-adjusted alpha (return minus beta*SPY return)
    - Selects stocks genuinely outperforming the market on a risk-adjusted basis
    - Proven OOS: gen7_sp500_idiosyncratic_momentum (IS 1.20, OOS 0.70, h2-stable)

  Flat/inverted long-end slope (TYX-TNX < 200d MA):
    - Duration premium compressing = growth outlook dimming
    - Use QUALITY/NEARHI momentum: rank by 126d return filtered to price>80% of 252d high
    - Selects stocks in sustained uptrends (not just short bursts)
    - Proven OOS: gen6_nearhi_momentum_quality (IS 1.16, OOS 0.63, h2-dominant)

  SPY 200d SMA outer bear gate -> TLT

Rationale: The two stock-selection modes have different loss-mode profiles. In rising
long-end rate environments (steep slope), idiosyncratic alpha (which strips market beta)
is more robust. In flat-rate environments (compressing term premium), quality/nearhi
filters for stocks with durable structural advantages. Switching between them based on
the macro duration-premium signal is a novel second-order combination.

Distinction from existing: gen8_opus1_longend_slope uses same slope signal but routes
between SP500-momentum vs SPY+TLT (ETF route, not two stock-selection modes). This
routes between two fundamentally different stock-ranking methodologies.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10
MOM_WINDOW_IDIO = 63       # idiosyncratic momentum lookback
BETA_WINDOW = 126          # beta estimation window
MOM_WINDOW_QUALITY = 126   # quality/nearhi momentum lookback
NEARHI_THRESHOLD = 0.80    # price must be > 80% of 252d high
HIGH_WINDOW = 252          # lookback for 52w high
SLOPE_TREND_WINDOW = 200   # window for slope's own MA
SPY_TREND_WINDOW = 200     # SPY 200d SMA bear gate
TOP_K = 15
EXPOSURE = 0.97
VOL_WINDOW = 21
_SPY = "SPY"
_TLT = "TLT"
_IEF = "IEF"
_TYX = "^TYX"
_TNX = "^TNX"


class SlopeIdioVsQuality(Strategy):
    """Long-end slope (TYX-TNX vs 200d MA) switches between idiosyncratic and quality momentum."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        mom_window_idio: int = MOM_WINDOW_IDIO,
        beta_window: int = BETA_WINDOW,
        mom_window_quality: int = MOM_WINDOW_QUALITY,
        nearhi_threshold: float = NEARHI_THRESHOLD,
        high_window: int = HIGH_WINDOW,
        slope_trend_window: int = SLOPE_TREND_WINDOW,
        spy_trend_window: int = SPY_TREND_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
        vol_window: int = VOL_WINDOW,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            mom_window_idio=mom_window_idio,
            beta_window=beta_window,
            mom_window_quality=mom_window_quality,
            nearhi_threshold=nearhi_threshold,
            high_window=high_window,
            slope_trend_window=slope_trend_window,
            spy_trend_window=spy_trend_window,
            top_k=top_k,
            exposure=exposure,
            vol_window=vol_window,
        )
        self.rebalance_every = int(rebalance_every)
        self.mom_window_idio = int(mom_window_idio)
        self.beta_window = int(beta_window)
        self.mom_window_quality = int(mom_window_quality)
        self.nearhi_threshold = float(nearhi_threshold)
        self.high_window = int(high_window)
        self.slope_trend_window = int(slope_trend_window)
        self.spy_trend_window = int(spy_trend_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)
        self.vol_window = int(vol_window)

    def _compute_spy_returns(self, ctx: BarContext, window: int) -> np.ndarray | None:
        """Compute SPY daily returns for beta estimation."""
        try:
            spy_hist = ctx.history(_SPY)
            if spy_hist is None or len(spy_hist) < window + 2:
                return None
            spy_close = spy_hist["close"].dropna()
            if len(spy_close) < window + 1:
                return None
            spy_rets = spy_close.iloc[-(window + 1):].pct_change().dropna().values
            return spy_rets
        except Exception:
            return None

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.slope_trend_window, self.spy_trend_window,
                     self.high_window, self.beta_window) + 10
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

        # --- SPY 200d SMA outer bear gate ---
        spy_bull = True
        try:
            spy_hist = ctx.history(_SPY)
            if spy_hist is not None and len(spy_hist) >= self.spy_trend_window + 2:
                spy_close = spy_hist["close"].dropna()
                if len(spy_close) >= self.spy_trend_window:
                    spy_sma = float(spy_close.iloc[-self.spy_trend_window:].mean())
                    spy_now = float(spy_close.iloc[-1])
                    spy_bull = spy_now > spy_sma
        except Exception:
            pass

        target: dict[str, float] = {}

        if not spy_bull:
            # Bear market — TLT
            if _TLT in live:
                target[_TLT] = self.exposure
        else:
            # --- Long-end slope regime: TYX-TNX vs own 200d MA ---
            steep_slope = True  # default risk-on
            try:
                tyx_hist = ctx.history(_TYX)
                tnx_hist = ctx.history(_TNX)
                if (tyx_hist is not None and tnx_hist is not None and
                        len(tyx_hist) >= self.slope_trend_window + 2 and
                        len(tnx_hist) >= self.slope_trend_window + 2):
                    tyx_close = tyx_hist["close"].dropna()
                    tnx_close = tnx_hist["close"].dropna()
                    n = min(len(tyx_close), len(tnx_close))
                    if n >= self.slope_trend_window + 1:
                        slope = tyx_close.values[-n:] - tnx_close.values[-n:]
                        slope_ma = float(np.mean(slope[-self.slope_trend_window:]))
                        slope_now = float(slope[-1])
                        steep_slope = slope_now > slope_ma
            except Exception:
                pass

            need = max(self.beta_window, self.mom_window_quality, self.high_window) + 10
            prices = ctx.closes_window(need)

            if steep_slope:
                # Steep long-end slope -> IDIOSYNCRATIC momentum (beta-adjusted)
                spy_rets = self._compute_spy_returns(ctx, self.beta_window)

                idio_scores: dict[str, float] = {}
                for sym in prices.columns:
                    if sym in (_SPY, _TLT, _IEF, _TYX, _TNX):
                        continue
                    col = prices[sym].dropna()
                    if len(col) < self.mom_window_idio + 5:
                        continue

                    # Compute 63d return
                    p_end = float(col.iloc[-1])
                    p_start = float(col.iloc[-self.mom_window_idio])
                    if p_start <= 0:
                        continue
                    stock_ret = p_end / p_start - 1.0

                    # Compute beta vs SPY over beta_window
                    beta = 1.0  # fallback
                    if spy_rets is not None and len(col) >= self.beta_window + 2:
                        stock_rets_raw = col.iloc[-(self.beta_window + 1):].pct_change().dropna().values
                        n_align = min(len(stock_rets_raw), len(spy_rets))
                        if n_align >= 30:
                            s_r = stock_rets_raw[-n_align:]
                            m_r = spy_rets[-n_align:]
                            var_spy = float(np.var(m_r))
                            if var_spy > 0:
                                beta = float(np.cov(s_r, m_r)[0, 1] / var_spy)
                                beta = max(-2.0, min(3.0, beta))

                    # SPY 63d return for beta-adjustment
                    try:
                        spy_h = ctx.history(_SPY)
                        spy_c = spy_h["close"].dropna()
                        if len(spy_c) >= self.mom_window_idio:
                            spy_ret_63 = float(spy_c.iloc[-1] / spy_c.iloc[-self.mom_window_idio] - 1.0)
                        else:
                            spy_ret_63 = 0.0
                    except Exception:
                        spy_ret_63 = 0.0

                    idio = stock_ret - beta * spy_ret_63
                    if np.isfinite(idio) and sym in live:
                        idio_scores[sym] = idio

                if len(idio_scores) < 5:
                    if _SPY in live:
                        target[_SPY] = self.exposure
                else:
                    # Inverse-vol weighting
                    ranked = sorted(idio_scores, key=idio_scores.__getitem__, reverse=True)
                    inv_vols: dict[str, float] = {}
                    count = 0
                    for sym in ranked:
                        if count >= self.top_k:
                            break
                        try:
                            s_hist = ctx.history(sym)
                            if s_hist is None:
                                continue
                            s_close = s_hist["close"].dropna()
                            if len(s_close) >= self.vol_window + 1:
                                rets = s_close.iloc[-(self.vol_window + 1):].pct_change().dropna()
                                rv = float(rets.std()) * np.sqrt(252)
                            else:
                                rv = 0.20
                        except Exception:
                            rv = 0.20
                        if rv <= 0:
                            rv = 0.20
                        if sym in live:
                            inv_vols[sym] = 1.0 / rv
                            count += 1

                    if not inv_vols:
                        if _SPY in live:
                            target[_SPY] = self.exposure
                    else:
                        total = sum(inv_vols.values())
                        for sym, iv in inv_vols.items():
                            target[sym] = self.exposure * iv / total

            else:
                # Flat/inverted long-end slope -> QUALITY/NEARHI momentum
                quality_scores: dict[str, float] = {}
                for sym in prices.columns:
                    if sym in (_SPY, _TLT, _IEF, _TYX, _TNX):
                        continue
                    col = prices[sym].dropna()
                    if len(col) < self.mom_window_quality + 5:
                        continue

                    # 126d return
                    p_end = float(col.iloc[-1])
                    p_start = float(col.iloc[-self.mom_window_quality])
                    if p_start <= 0:
                        continue
                    mom_ret = p_end / p_start - 1.0
                    if not np.isfinite(mom_ret):
                        continue

                    # Near-52w-high quality filter
                    if len(col) >= self.high_window:
                        hi_252 = float(col.iloc[-self.high_window:].max())
                    else:
                        hi_252 = float(col.max())
                    if hi_252 <= 0:
                        continue
                    nearhi_ratio = p_end / hi_252
                    if nearhi_ratio < self.nearhi_threshold:
                        continue  # not near-high quality filter

                    if sym in live:
                        quality_scores[sym] = mom_ret

                if len(quality_scores) < 5:
                    if _SPY in live:
                        target[_SPY] = self.exposure
                else:
                    # Inverse-vol weighting
                    ranked = sorted(quality_scores, key=quality_scores.__getitem__, reverse=True)
                    inv_vols: dict[str, float] = {}
                    count = 0
                    for sym in ranked:
                        if count >= self.top_k:
                            break
                        try:
                            s_hist = ctx.history(sym)
                            if s_hist is None:
                                continue
                            s_close = s_hist["close"].dropna()
                            if len(s_close) >= self.vol_window + 1:
                                rets = s_close.iloc[-(self.vol_window + 1):].pct_change().dropna()
                                rv = float(rets.std()) * np.sqrt(252)
                            else:
                                rv = 0.20
                        except Exception:
                            rv = 0.20
                        if rv <= 0:
                            rv = 0.20
                        if sym in live:
                            inv_vols[sym] = 1.0 / rv
                            count += 1

                    if not inv_vols:
                        if _SPY in live:
                            target[_SPY] = self.exposure
                    else:
                        total = sum(inv_vols.values())
                        for sym, iv in inv_vols.items():
                            target[sym] = self.exposure * iv / total

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
    return sp500_tickers() + [_TLT, _IEF, _SPY, _TYX, _TNX]


UNIVERSE = _universe

NAME = "slope_idio_vs_quality"
HYPOTHESIS = (
    "TYX-TNX long-end slope regime switches between two proven stock-selection modes: "
    "steep slope (>200d MA) routes to SP500 top-15 idiosyncratic momentum (beta-adjusted residual vs SPY, 63d); "
    "flat/inverted (<200d MA) routes to SP500 top-15 near-52w-high quality momentum (126d, price>80% of 252d high); "
    "inverse-vol weighted in both modes; SPY 200d SMA bear gate to TLT; biweekly rebalance"
)

STRATEGY = SlopeIdioVsQuality()
