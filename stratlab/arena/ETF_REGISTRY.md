# Arena ETF Registry — IS coverage by ticker

A reference of common ETFs grouped by their cache coverage of the IS window
(2010-01-01 .. 2018-12-31). Use this BEFORE designing a strategy around a
factor / sector / asset-class ETF, so you don't discover mid-round that the
cache start is 2013+ and your strategy silently falls back to SPY.

Verify any specific ticker via:

    python -m stratlab.data.inception --tickers TKR1 TKR2 --covers-is

Last refreshed: 2026-05-15.

---

## ✅ Full IS coverage (start ≤ 2010-01-01)

Safe to use across the entire IS window without caveats.

### Broad equity / asset-class

    SPY  IVV  VTI  IWM  QQQ  EFA  VEA  VWO  EMB  AGG  TLT  LQD  TIP  DVY

### Sector SPDRs (covers IS, well-trafficked)

    XLB  XLE  XLF  XLI  XLK  XLP  XLU  XLV  XLY
    (Note: XLRE/XLC NOT covered — see below)

### Sub-industry / industry ETFs (less commonly used — opportunity)

    KBE   banks, since 2005
    KRE   regional banks, since 2006
    SMH   semiconductors, since 2000
    SOXX  semiconductors (broader), since 2001
    IGV   software / tech sub-sector, since 2001
    XBI   biotech, since 2006
    XOP   oil & gas exploration, since 2006
    XHB   homebuilders (verify locally)
    ITB   home construction (verify locally)
    REM   mortgage REITs, since 2007 — orthogonal to JNK/LQD credit signals
    VNQ   real estate, since 2004

### Currency / commodity index ETFs

    UUP   USD bullish, since 2007
    FXE   euro, since 2005
    FXY   yen, since 2007
    USO   crude oil, since 2006 (warning: secular bear in IS)
    DBC   broad commodities, since 2006 (warning: secular bear in IS)
    SLV   silver, since 2006

### Inverse / leveraged

    SH    1x inverse SPY, since 2006
    PSQ   1x inverse QQQ, since 2006
    SDS   2x inverse SPY, since 2006
    SCO   2x inverse oil, since 2008
    PFF   preferreds (NOT inverse), since 2007 — caveat: behaves like duration

### Vol / rate indices (signal-only)

    ^VIX   ^VVIX   ^MOVE   ^SKEW   ^OVX   ^GVZ
    ^TNX   ^IRX   ^FVX   ^TYX

---

## ⚠ "Effectively covers IS" — cache start 2010-01-04

The check `covers_is=False` is technically correct (start > 2010-01-01) but
the missing dates are Jan 1-3 which are non-trading. These ETFs work fine
in practice — many strategies use them as core defensive vehicles.

    GLD   gold (cache from 2010-01-04)
    HYG   high-yield credit
    IEF   7-10y Treasury
    SHY   1-3y Treasury

---

## ❌ DOES NOT cover IS start — factor ETFs launched mid-IS

These are the cache-gap traps. Yfinance fallback may succeed silently but
the strategy's "MTUM-vs-SPY" signal effectively starts in 2013, biasing
the IS Calmar upward (the 2013-2018 sub-window is the QE-rally bull).
Multiple agents have wasted submissions on these.

    SCHD  2011-10  (Schwab dividend)
    MTUM  2013-04  (iShares MSCI USA Momentum Factor)
    QUAL  2013-07  (iShares MSCI USA Quality Factor)
    NOBL  2013-10  (Dividend aristocrats)
    XLRE  2015-10  (Real estate sector — split from XLF in 2015)
    XLC   2018-06  (Communication services — split from XLK in 2018)

If you want factor exposure with full IS coverage, fall back to:
- Momentum: build cross-sectionally from SP500 stocks (not MTUM)
- Quality: use VIG (since 2006) instead of QUAL/SCHD
- Dividends: DVY (since 2003) — full IS coverage

---

## ❌ Post-OOS or sparse — unusable for IS

These appear cached but only with post-2020 data. They CANNOT be used in
strategies evaluated on the 2010-2018 IS window.

    VYM   IBB   KIE   HACK   IYR   QID   (cache starts 2020+)
    Single-stock leveraged ETFs (TSLL, NVDL, AAPB, MSFU, CONL, ...) — most
    launched 2022+; verify per-ticker before designing around them.

---

## How to use

1. **Before designing a strategy** that references a factor/sector ETF, run:
   `python -m stratlab.data.inception --tickers <TKR> --covers-is`
2. **If the ticker is in the ⚠ or ❌ list above**, either:
   - Pick a full-IS-covered substitute from the ✅ list, or
   - Document the cache-gap as an INTENTIONAL design choice (rare; needs
     to be a robust-to-NaN strategy or a late-IS-only test)
3. **For brand-new ETFs not listed**, just run the `--covers-is` check
   yourself — the registry is non-exhaustive.

---

## Open frontiers — IS-covered tickers that are RARELY USED on the leaderboard

(As of round 10) These tickers are in the ✅ list above but appear in <3
leaderboard strategies. Each is a potential gap-finder opportunity:

- **REM** — mortgage REITs (rate + credit composite). Gen_9 opus-2 found it
  with loss_mode_corr 0.389 — strong diversifier.
- **KBE / KRE** — bank / regional-bank spread. Used once (gen_9 opus-2).
- **XBI** — biotech as a higher-vol tech-orthogonal sector.
- **XOP** — oil exploration. Secular bear in IS, but as a stress/regime
  signal could work.
- **FXE / FXY / UUP** as cross-asset SIGNALS (not exposure). UUP has
  appeared in 2 strategies; FXE/FXY in zero.
- **IGV** — pure-software ETF, orthogonal to broad-tech QQQ.
- **PFF** — failed as standalone (gen_7); could work as a RATIO with JNK or
  LQD (credit-quality term-structure signal).
- **SDS / SCO** — 2x inverse pairs as tail-hedge sleeves, distinct from
  routing-to-bonds defensive.
