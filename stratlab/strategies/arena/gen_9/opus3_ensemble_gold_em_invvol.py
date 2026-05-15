"""opus-3 gen_9 ensemble #1 — Inverse-vol weighted triplet (gold + EM + GDX/GLD).

Components (measured pairwise IS-return Pearson corrs via corr_dump):

    A. gen9_gdx_gld_momentum_sp500
       - IS Calmar 0.68, lmc 0.42 (orthogonal loss mode)
       - Signal: GDX 42d return vs GLD 42d return; commodity miners-vs-metal
         regime. Risk-on -> top-15 SP500 63d momentum above 200d SMA.
         Risk-off -> IEF.

    B. gen9_iau_gold_trend_equity_gate
       - IS Calmar 0.67
       - Signal: IAU vs 63d SMA. Gold weak -> top-15 SP500 126d-skip-21d
         momentum. Gold strong -> TLT 60% + LQD 37%. SPY 200d outer gate.

    C. gen9_em_dm_ratio_sp500_gate
       - IS Calmar 0.65, lowest corr-of-round at 0.49
       - Signal: VWO 60d return vs VEA 60d return. EM leads DM AND SPY > 200d SMA
         -> top-15 SP500 63d momentum + 200d filter. Otherwise TLT 60% + IEF 37%.

Measured pairwise corrs (corr_dump):
    A-B = +0.15  (lowest of any pair examined; GDX/GLD spread vs IAU level are
                  structurally distinct gold proxies — return delta vs trend MA)
    A-C = +0.31  (different cross-asset gates, both route to SP500 momentum)
    B-C = +0.24  (gold trend vs EM/DM differential — fully orthogonal signals)
    mean |corr| = 0.23

These satisfy the <0.3 informal target on 2 of 3 pairs and 0.31 on the third,
much stronger than the brief's proposed triplet (em_us + gdx_gld + idio_slope
had 0.65 on em_us vs idio_slope, which violates the orthogonality goal).

Combining rule: INVERSE-VOL WEIGHTED.
    On each rebalance day:
      1. Compute each component's target weights (None if warmup-pending).
      2. Estimate each component's recent realized return vol from a
         "shadow equity curve" — we approximate each component's PnL via the
         portfolio-weighted return of its current targets over the last 60
         bars; the 60d std of that proxy is the component's vol estimate.
         (We can't read tmp/arena/equity_curves/ from inside a strategy —
          curves only exist post-submit — so we shadow-track in self.state.)
      3. Component weights = (1/vol_i) / sum(1/vol_j), capped to a min/max
         band of [0.15, 0.55] to prevent any single component from
         dominating during a transient calm regime.
      4. Aggregate target weights by symbol (overlap nets), scale to
         EXPOSURE_CAP = 0.97 if total > cap.
      5. Emit delta orders to reach target.

Why inverse-vol vs equal-weight:
    Equal-weight (1/3 each) is the bond_credit_seasonal default and is the
    conservative choice. Inverse-vol is a slightly more aggressive
    diversifier that automatically reduces exposure to whichever component
    is in a high-vol regime (typically the SP500-momentum legs during
    drawdowns). For a triplet where all three components have similar
    long-run vols but transient regime divergence, inverse-vol should give
    smoother aggregate equity than equal-weight.

Stability target: beat the curated gen5_ensemble_bond_credit_seasonal's
OOS Calmar of 0.53 (retention 78%) rather than chase IS peak. The
brief's warning that gen8_ensemble_credit_volcarry_seasonal degraded
from IS 0.97 -> OOS 0.31 (32% retention) is the cautionary tale.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy
from stratlab.data.universe import sp500_tickers

# ---------------------------------------------------------------------------
# Component A — gdx_gld_momentum_sp500 parameters
# ---------------------------------------------------------------------------
A_MOMENTUM_WINDOW = 63
A_TREND_WINDOW = 200
A_SIGNAL_WINDOW = 42      # GDX-vs-GLD 42d return
A_TOP_K = 15

# ---------------------------------------------------------------------------
# Component B — iau_gold_trend_equity_gate parameters
# ---------------------------------------------------------------------------
B_MOM_LONG = 126
B_MOM_SKIP = 21
B_GOLD_MA_WINDOW = 63
B_TREND_WINDOW = 200
B_TOP_K = 15

# ---------------------------------------------------------------------------
# Component C — em_dm_ratio_sp500_gate parameters
# ---------------------------------------------------------------------------
C_MOMENTUM_WINDOW = 63
C_TREND_WINDOW = 200
C_SIGNAL_WINDOW = 60      # VWO-vs-VEA 60d return
C_TOP_K = 15

# ---------------------------------------------------------------------------
# Ensemble parameters
# ---------------------------------------------------------------------------
EXPOSURE_CAP = 0.97
REBALANCE_EVERY = 10        # biweekly (matches all 3 components)
VOL_WINDOW = 60             # 60-bar realized-vol of shadow component returns
WEIGHT_MIN = 0.15           # floor on each component's weight (inverse-vol band)
WEIGHT_MAX = 0.55           # cap on each component's weight
WARMUP_BARS = max(
    A_MOMENTUM_WINDOW + A_TREND_WINDOW,
    B_MOM_LONG + B_TREND_WINDOW,
    C_MOMENTUM_WINDOW + C_TREND_WINDOW,
) + VOL_WINDOW + 10

# Symbols used by components but excluded from SP500-momentum scoring
_NON_RANK_SYMS = {
    "SPY", "TLT", "IEF", "LQD", "GDX", "GLD", "IAU", "VWO", "VEA",
}


# ===========================================================================
# Component-A weight function
# ===========================================================================
def _component_a_weights(ctx: BarContext) -> dict[str, float] | None:
    warmup = max(A_TREND_WINDOW, A_SIGNAL_WINDOW) + 5
    if ctx.idx < warmup:
        return None

    closes = ctx.closes()
    if closes.empty:
        return None
    live_all = {s: float(p) for s, p in closes.items() if float(p) > 0}

    # SPY 200d gate
    try:
        spy_hist = ctx.history("SPY")
    except KeyError:
        return None
    if len(spy_hist) < A_TREND_WINDOW:
        return None
    spy_sma = float(spy_hist["close"].iloc[-A_TREND_WINDOW:].mean())
    spy_price = live_all.get("SPY", 0.0)
    spy_bull = spy_price > 0 and spy_price > spy_sma

    if not spy_bull:
        return {"IEF": 1.0} if "IEF" in live_all else None

    # GDX vs GLD 42d return differential
    try:
        gdx_hist = ctx.history("GDX")
        gld_hist = ctx.history("GLD")
    except KeyError:
        return {"IEF": 1.0} if "IEF" in live_all else None
    if len(gdx_hist) < A_SIGNAL_WINDOW + 1 or len(gld_hist) < A_SIGNAL_WINDOW + 1:
        return {"IEF": 1.0} if "IEF" in live_all else None

    gdx_close = gdx_hist["close"]
    gld_close = gld_hist["close"]
    gdx_ret = float(gdx_close.iloc[-1] / gdx_close.iloc[-A_SIGNAL_WINDOW] - 1.0)
    gld_ret = float(gld_close.iloc[-1] / gld_close.iloc[-A_SIGNAL_WINDOW] - 1.0)
    if not (np.isfinite(gdx_ret) and np.isfinite(gld_ret)):
        return {"IEF": 1.0} if "IEF" in live_all else None

    gdx_risk_on = gdx_ret > gld_ret
    if not gdx_risk_on:
        return {"IEF": 1.0} if "IEF" in live_all else None

    # Risk-on: top-15 SP500 63d momentum > 200d SMA
    prices = ctx.closes_window(A_MOMENTUM_WINDOW + 5)
    if len(prices) < A_MOMENTUM_WINDOW:
        return {"IEF": 1.0} if "IEF" in live_all else None

    scores: dict[str, float] = {}
    for sym in prices.columns:
        if sym in _NON_RANK_SYMS:
            continue
        col = prices[sym].dropna()
        if len(col) < A_MOMENTUM_WINDOW:
            continue
        p_start = float(col.iloc[-A_MOMENTUM_WINDOW])
        if p_start <= 0:
            continue
        r = float(col.iloc[-1] / p_start - 1.0)
        if np.isfinite(r):
            scores[sym] = r

    if len(scores) < A_TOP_K:
        return {"IEF": 1.0} if "IEF" in live_all else None

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    selected: list[str] = []
    for sym, _ in ranked:
        if len(selected) >= A_TOP_K:
            break
        hist = ctx.history(sym)
        if len(hist) < A_TREND_WINDOW:
            continue
        sma = float(hist["close"].iloc[-A_TREND_WINDOW:].mean())
        price = live_all.get(sym, 0.0)
        if price > sma:
            selected.append(sym)

    if not selected:
        return {"IEF": 1.0} if "IEF" in live_all else None

    per_w = 1.0 / len(selected)
    return {sym: per_w for sym in selected}


# ===========================================================================
# Component-B weight function
# ===========================================================================
def _component_b_weights(ctx: BarContext) -> dict[str, float] | None:
    warmup = max(B_TREND_WINDOW, B_MOM_LONG, B_GOLD_MA_WINDOW) + 10
    if ctx.idx < warmup:
        return None

    try:
        spy_hist = ctx.history("SPY")
    except KeyError:
        return None
    if len(spy_hist) < B_TREND_WINDOW + 5:
        return None
    spy_close = spy_hist["close"].dropna()
    if len(spy_close) < B_TREND_WINDOW:
        return None
    spy_sma = float(spy_close.iloc[-B_TREND_WINDOW:].mean())
    spy_now = float(spy_close.iloc[-1])
    spy_bull = spy_now > spy_sma

    closes_now = ctx.closes()
    if closes_now.empty:
        return None
    live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}

    # IAU gold trend
    gold_weak = True
    try:
        iau_hist = ctx.history("IAU")
        if iau_hist is not None and len(iau_hist) >= B_GOLD_MA_WINDOW + 2:
            iau_close = iau_hist["close"].dropna()
            if len(iau_close) >= B_GOLD_MA_WINDOW + 1:
                gold_ma = float(iau_close.iloc[-B_GOLD_MA_WINDOW:].mean())
                gold_now = float(iau_close.iloc[-1])
                gold_weak = gold_now < gold_ma
    except Exception:
        pass

    if not spy_bull:
        return {"TLT": 1.0} if "TLT" in live else None
    if not gold_weak:
        out: dict[str, float] = {}
        if "TLT" in live:
            out["TLT"] = 0.618
        if "LQD" in live:
            out["LQD"] = 0.379
        # renormalize to 1.0 since we'll scale via ensemble component weight
        s = sum(out.values())
        if s <= 0:
            return None
        return {k: v / s for k, v in out.items()}

    # Risk-on: top-15 skip-month momentum (126d-21d)
    need = B_MOM_LONG + 5
    prices = ctx.closes_window(need)
    if len(prices) < B_MOM_LONG:
        return {"SPY": 1.0} if "SPY" in live else None

    scores: dict[str, float] = {}
    for sym in prices.columns:
        if sym in _NON_RANK_SYMS:
            continue
        col = prices[sym].dropna()
        if len(col) < B_MOM_LONG:
            continue
        p_start = float(col.iloc[-B_MOM_LONG])
        p_end = float(col.iloc[-B_MOM_SKIP])
        if p_start <= 0:
            continue
        sm = p_end / p_start - 1.0
        if np.isfinite(sm):
            scores[sym] = sm

    if len(scores) < 5:
        return {"SPY": 1.0} if "SPY" in live else None

    k = min(B_TOP_K, len(scores))
    ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
    out2 = {sym: 1.0 / len(ranked) for sym in ranked if sym in live}
    if not out2:
        return {"SPY": 1.0} if "SPY" in live else None
    return out2


# ===========================================================================
# Component-C weight function
# ===========================================================================
def _component_c_weights(ctx: BarContext) -> dict[str, float] | None:
    warmup = max(C_TREND_WINDOW, C_SIGNAL_WINDOW) + 5
    if ctx.idx < warmup:
        return None

    closes = ctx.closes()
    if closes.empty:
        return None
    live_all = {s: float(p) for s, p in closes.items() if float(p) > 0}

    try:
        spy_hist = ctx.history("SPY")
    except KeyError:
        return None
    if len(spy_hist) < C_TREND_WINDOW:
        return None
    spy_sma = float(spy_hist["close"].iloc[-C_TREND_WINDOW:].mean())
    spy_price = live_all.get("SPY", 0.0)
    spy_bull = spy_price > 0 and spy_price > spy_sma

    if not spy_bull:
        return {"TLT": 0.619, "IEF": 0.381} if ("TLT" in live_all and "IEF" in live_all) else None

    # VWO vs VEA 60d return
    try:
        vwo_hist = ctx.history("VWO")
        vea_hist = ctx.history("VEA")
    except KeyError:
        return {"TLT": 0.619, "IEF": 0.381} if ("TLT" in live_all and "IEF" in live_all) else None

    em_risk_on = False
    if len(vwo_hist) >= C_SIGNAL_WINDOW + 1 and len(vea_hist) >= C_SIGNAL_WINDOW + 1:
        vwo_close = vwo_hist["close"]
        vea_close = vea_hist["close"]
        vwo_ret = float(vwo_close.iloc[-1] / vwo_close.iloc[-C_SIGNAL_WINDOW] - 1.0)
        vea_ret = float(vea_close.iloc[-1] / vea_close.iloc[-C_SIGNAL_WINDOW] - 1.0)
        if np.isfinite(vwo_ret) and np.isfinite(vea_ret):
            em_risk_on = vwo_ret > vea_ret

    if not em_risk_on:
        if "TLT" in live_all and "IEF" in live_all:
            return {"TLT": 0.619, "IEF": 0.381}
        return None

    prices = ctx.closes_window(C_MOMENTUM_WINDOW + 5)
    if len(prices) < C_MOMENTUM_WINDOW:
        if "TLT" in live_all and "IEF" in live_all:
            return {"TLT": 0.619, "IEF": 0.381}
        return None

    scores: dict[str, float] = {}
    for sym in prices.columns:
        if sym in _NON_RANK_SYMS:
            continue
        col = prices[sym].dropna()
        if len(col) < C_MOMENTUM_WINDOW:
            continue
        p_start = float(col.iloc[-C_MOMENTUM_WINDOW])
        if p_start <= 0:
            continue
        r = float(col.iloc[-1] / p_start - 1.0)
        if np.isfinite(r):
            scores[sym] = r

    if len(scores) < C_TOP_K:
        if "TLT" in live_all and "IEF" in live_all:
            return {"TLT": 0.619, "IEF": 0.381}
        return None

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    selected: list[str] = []
    for sym, _ in ranked:
        if len(selected) >= C_TOP_K:
            break
        hist = ctx.history(sym)
        if len(hist) < C_TREND_WINDOW:
            continue
        sma = float(hist["close"].iloc[-C_TREND_WINDOW:].mean())
        price = live_all.get(sym, 0.0)
        if price > sma:
            selected.append(sym)

    if not selected:
        if "TLT" in live_all and "IEF" in live_all:
            return {"TLT": 0.619, "IEF": 0.381}
        return None

    per_w = 1.0 / len(selected)
    return {sym: per_w for sym in selected}


# ===========================================================================
# Ensemble strategy
# ===========================================================================
class Opus3EnsembleGoldEmInvVol(Strategy):
    """Inverse-vol weighted triplet ensemble (opus-3, gen_9, ensemble #1).

    Components are evaluated independently each rebalance bar; each
    produces a normalized target dict (sum=1.0). The ensemble combines them
    with weights proportional to 1/vol_i, where vol_i is the 60-bar std of
    the shadow-portfolio return for component i. The shadow portfolio is
    each component's *previous* target dict marked-to-market on the daily
    closes (computed only between rebalances, ~10 bars apart).
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        exposure_cap: float = EXPOSURE_CAP,
        vol_window: int = VOL_WINDOW,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            exposure_cap=exposure_cap,
            vol_window=vol_window,
        )
        self.rebalance_every = int(rebalance_every)
        self.exposure_cap = float(exposure_cap)
        self.vol_window = int(vol_window)

        # Shadow-curve state — daily realized return of each component's last target dict.
        # We keep the rolling window of recent returns (most recent first).
        self._last_targets: list[dict[str, float] | None] = [None, None, None]
        self._last_prices: list[dict[str, float] | None] = [None, None, None]
        self._return_history: list[list[float]] = [[], [], []]

    def _update_component_returns(self, ctx: BarContext) -> None:
        """Roll forward each component's shadow return using yesterday's
        target weights and today's vs yesterday's prices. Called every bar."""
        closes_now = ctx.closes()
        if closes_now.empty:
            return
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}

        for i in range(3):
            tgt = self._last_targets[i]
            prev_prices = self._last_prices[i]
            if tgt is None or prev_prices is None:
                continue
            ret = 0.0
            total_w = 0.0
            for sym, w in tgt.items():
                p_now = live.get(sym)
                p_prev = prev_prices.get(sym)
                if p_now is None or p_prev is None or p_prev <= 0:
                    continue
                ret += w * (p_now / p_prev - 1.0)
                total_w += w
            # If no overlap, treat as zero (component effectively in cash this bar)
            if total_w > 0:
                self._return_history[i].append(ret)
                if len(self._return_history[i]) > self.vol_window:
                    self._return_history[i] = self._return_history[i][-self.vol_window:]
            # Update prev_prices to reflect today's prices for held symbols
            new_prices = {sym: live[sym] for sym in tgt if sym in live}
            if new_prices:
                self._last_prices[i] = new_prices

    def _component_weight_band(self, vols: list[float]) -> list[float]:
        """Compute inverse-vol weights with [WEIGHT_MIN, WEIGHT_MAX] band.

        If any vol is zero/unset, fall back to equal-weight (1/3 each).
        """
        if any(v is None or v <= 0 or not np.isfinite(v) for v in vols):
            return [1.0 / 3.0] * 3

        inv = [1.0 / v for v in vols]
        s = sum(inv)
        if s <= 0:
            return [1.0 / 3.0] * 3
        raw = [x / s for x in inv]

        # Clip into band then renormalize once (one-shot, no iteration)
        clipped = [min(max(w, WEIGHT_MIN), WEIGHT_MAX) for w in raw]
        cs = sum(clipped)
        if cs <= 0:
            return [1.0 / 3.0] * 3
        return [w / cs for w in clipped]

    def on_bar(self, ctx: BarContext) -> list[Order]:
        # Update shadow returns every bar (cheap; doesn't trade)
        self._update_component_returns(ctx)

        if ctx.idx < WARMUP_BARS:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # Re-compute each component's weights
        a_w = _component_a_weights(ctx)
        b_w = _component_b_weights(ctx)
        c_w = _component_c_weights(ctx)
        component_weights = [a_w, b_w, c_w]

        # Update last_targets/last_prices for components that returned something
        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live_all = {s: float(p) for s, p in closes_now.items() if float(p) > 0}

        for i, w in enumerate(component_weights):
            if w is not None:
                self._last_targets[i] = dict(w)
                self._last_prices[i] = {sym: live_all[sym] for sym in w if sym in live_all}

        # Compute inverse-vol band weights from shadow returns
        vols: list[float] = []
        for hist in self._return_history:
            if len(hist) >= max(20, self.vol_window // 2):
                vols.append(float(np.std(hist[-self.vol_window:])))
            else:
                vols.append(-1.0)  # signals "fall back to equal weight"
        ensemble_weights = self._component_weight_band(vols)

        # Aggregate symbol-level targets
        target: dict[str, float] = {}
        for cw, comp_target in zip(ensemble_weights, component_weights):
            if comp_target is None:
                continue
            for sym, w in comp_target.items():
                target[sym] = target.get(sym, 0.0) + w * cw

        total = sum(target.values())
        if total <= 0:
            return []
        if total > self.exposure_cap:
            scale = self.exposure_cap / total
            target = {k: v * scale for k, v in target.items()}

        # Portfolio value
        equity = ctx.portfolio_value(live_all)
        if equity <= 0:
            return []

        # Build target share counts
        target_shares: dict[str, int] = {}
        for sym, weight in target.items():
            price = live_all.get(sym)
            if not price or price <= 0:
                continue
            n = int(equity * weight / price)
            if n > 0:
                target_shares[sym] = n

        orders: list[Order] = []
        # Sells first (free cash for buys)
        for sym, pos in list(ctx.positions.items()):
            if sym not in target_shares and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))
        for sym, tgt in target_shares.items():
            cur = int(ctx.position(sym).size)
            delta = tgt - cur
            if delta < -1:
                orders.append(Order(side=OrderSide.SELL, size=abs(delta), symbol=sym))
        for sym, tgt in target_shares.items():
            cur = int(ctx.position(sym).size)
            delta = tgt - cur
            if delta > 1:
                orders.append(Order(side=OrderSide.BUY, size=delta, symbol=sym))
        return orders


def _universe() -> list[str]:
    extra = ["SPY", "TLT", "IEF", "LQD", "GDX", "GLD", "IAU", "VWO", "VEA"]
    return sp500_tickers() + extra


NAME = "opus3_ensemble_gold_em_invvol"
HYPOTHESIS = (
    "Inverse-vol weighted triplet: A=gdx_gld_momentum_sp500 (GDX-vs-GLD 42d return gate), "
    "B=iau_gold_trend_equity_gate (IAU vs 63d MA gate, 126d-21d skip-mom), "
    "C=em_dm_ratio_sp500_gate (VWO-vs-VEA 60d return gate); measured pairwise corrs "
    "A-B=0.15, A-C=0.31, B-C=0.24 via corr_dump; weights = inverse 60d realized vol of "
    "shadow component returns, bounded to [0.15, 0.55] band; biweekly rebalance, "
    "EXPOSURE_CAP=0.97."
)
UNIVERSE = _universe

STRATEGY = Opus3EnsembleGoldEmInvVol()
