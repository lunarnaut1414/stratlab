"""Curated ticker lists for ETFs and ETPs, organized by category.

These lists are intentionally hardcoded — there's no clean free source for
"all major ETFs" (ETFdb blocks scraping, Yahoo doesn't expose a list endpoint),
and ETF tickers turn over slowly, so a curated snapshot stays useful.

Tickers are Yahoo-formatted (use ``-`` not ``.``).
"""
from __future__ import annotations


# Mapping of category → list of tickers. Categories drive the on-disk folder
# layout (data/market/etfs/<category>/<ticker>.csv) and feed the catalog.
ETF_CATEGORIES: dict[str, list[str]] = {
    "broad_market": [
        # US equity total / large / mid / small / equal-weight
        "SPY", "IVV", "VOO", "VTI", "QQQ", "QQQM", "IWM", "DIA",
        "MDY", "IJH", "IJR", "RSP", "SCHX", "SCHB", "SCHG", "SCHV",
        "VTV", "VUG", "VYM", "VIG", "SCHD", "DVY", "NOBL",
    ],
    "factor": [
        "MTUM", "QUAL", "USMV", "VLUE", "SIZE", "SPLV", "MOAT",
    ],
    "sector": [
        # Sector SPDRs
        "XLF", "XLE", "XLU", "XLK", "XLI", "XLY", "XLP", "XLV", "XLB", "XLRE", "XLC",
        # Vanguard sector
        "VFH", "VHT", "VGT", "VDE", "VCR", "VDC", "VIS", "VPU",
    ],
    "industry": [
        # Sub-sector / industry slices
        "SMH", "SOXX",                         # semis
        "IBB", "XBI",                          # biotech
        "GDX", "GDXJ", "RING", "SIL",          # gold / silver miners
        "OIH", "XOP",                          # energy services / E&P
        "KIE", "KBE", "KRE",                   # insurance / banks
        "ITA", "XAR", "PPA",                   # defense / aerospace
        "IGV", "SKYY", "WCLD",                 # software / cloud
        "HACK", "CIBR",                        # cybersecurity
        "ROBO", "BOTZ", "IRBO",                # robotics / AI
        "JETS",                                # airlines
        "ESPO", "HERO",                        # gaming / esports
        "PEJ",                                 # entertainment
        "MOO",                                 # agriculture
        "PHO", "FIW",                          # water
        "BLOK",                                # blockchain equities
    ],
    "thematic": [
        "ARKK", "ARKG", "ARKW", "ARKQ", "ARKF",
        "ICLN", "TAN", "FAN", "PBW",           # clean energy
        "LIT",                                 # lithium / battery
        "URA", "URNM",                         # uranium
        "KWEB", "CQQQ",                        # China internet
    ],
    "international_developed": [
        "EFA", "IEFA", "VEA", "SCHF", "SPDW",
        "EWJ", "DXJ",                          # Japan
        "EWG", "EWU", "EWA", "EWC",            # Germany / UK / Australia / Canada
        "EWQ", "EWI", "EWP", "EWN", "EWL",     # France / Italy / Spain / Netherlands / Switzerland
    ],
    "international_emerging": [
        "EEM", "IEMG", "VWO", "SCHE", "SPEM",
        "INDA", "EPI",                         # India
        "FXI", "MCHI", "ASHR",                 # China
        "EWZ", "EWW",                          # Brazil, Mexico
        "EWY", "EWT",                          # Korea, Taiwan
        "EZA",                                 # South Africa
        "TUR",                                 # Turkey
        "ARGT",                                # Argentina
        "VNM",                                 # Vietnam
    ],
    "bonds": [
        "TLT", "IEF", "SHY", "BIL",            # treasury duration ladder
        "AGG", "BND", "SCHZ",                  # aggregate
        "LQD", "VCIT", "VCSH",                 # corporate
        "HYG", "JNK", "USHY",                  # high yield
        "EMB", "PCY",                          # emerging market debt
        "TIP", "VTIP", "SCHP",                 # TIPS
        "MBB",                                 # mortgage-backed
        "MUB", "VTEB",                         # municipals
        "PFF", "PGX",                          # preferred
        "SHV", "SGOV",                         # ultra-short
        "BNDX",                                # international agg
    ],
    "commodities": [
        "GLD", "IAU", "SGOL",                  # gold
        "SLV", "SIVR",                         # silver
        "PPLT", "PALL",                        # platinum / palladium
        "USO", "BNO", "UCO",                   # oil
        "UNG", "BOIL",                         # natgas
        "DBC", "PDBC",                         # broad commodity
        "DBA", "CORN", "WEAT", "SOYB",         # agriculture
        "COPX", "CPER",                        # copper
    ],
    "real_estate": [
        "VNQ", "IYR", "SCHH", "RWR",
        "VNQI",                                # international real estate
        "REM", "MORT",                         # mortgage REITs
    ],
    "currency": [
        "UUP", "UDN",                          # dollar bull / bear
        "FXE", "FXY", "FXB", "FXC", "FXA",     # major fx
        "CYB",                                 # chinese yuan
    ],
    "volatility": [
        "VXX", "VIXY", "VIXM",                 # VIX-tracking (decay-prone)
    ],
    "crypto": [
        "GBTC", "ETHE",                        # Grayscale trusts
        "BITO",                                # Bitcoin futures
        "IBIT", "FBTC", "BITB", "ARKB", "HODL",# spot Bitcoin
        "ETHA", "FETH", "ETHW",                # spot Ethereum
    ],
    "inverse": [
        # Broad US
        "SH", "SDS", "SPXU", "SPXS",           # SPY -1x / -2x / -3x
        "PSQ", "QID", "SQQQ",                  # QQQ -1x / -2x / -3x
        "RWM", "TWM", "SRTY",                  # IWM -1x / -2x / -3x
        "DOG", "DXD", "SDOW",                  # DIA -1x / -2x / -3x
        "SPDN",                                # alternate -1x SPY
        # Sectors
        "SOXS",                                # semis -3x
        "DRV",                                 # real estate -3x
        "LABD",                                # biotech -3x
        "FAZ",                                 # financials -3x
        "ERY",                                 # energy -3x
        "TZA",                                 # small-cap -3x
        "DUST",                                # gold miners -2x
        "SCO",                                 # oil -2x
        "KOLD",                                # natgas -2x
        # Bonds
        "TBF", "TBT", "TMV",                   # 20yr treasury -1x / -2x / -3x
        "PST",                                 # 7-10yr -2x
        # International
        "EUM", "EDZ",                          # EM -1x / -3x
        "EFZ",                                 # EAFE -1x
        "YANG",                                # China -3x
    ],
    "leveraged": [
        # Broad US
        "SSO", "UPRO",                         # SPY 2x / 3x
        "QLD", "TQQQ",                         # QQQ 2x / 3x
        "DDM", "UDOW",                         # DIA 2x / 3x
        "URTY", "UWM",                         # Russell 3x / 2x
        "MVV",                                 # mid-cap 2x
        # Sectors
        "ROM", "REW",                          # tech 2x / -2x (REW inverse, dedup'd)
        "FAS",                                 # financials 3x
        "ERX", "GUSH",                         # energy 3x
        "CURE",                                # healthcare 3x
        "SOXL",                                # semis 3x
        "DRN",                                 # real estate 3x
        "NUGT", "JNUG",                        # gold miners 2x
        "LABU",                                # biotech 3x
        "DPST",                                # regional banks 3x
        "RETL",                                # retail 3x
        "WEBL",                                # internet 3x
        "DFEN",                                # defense 3x
        # International
        "YINN",                                # China 3x bull
        "EDC",                                 # EM 3x bull
        "EZJ",                                 # Japan 2x
        # Bonds
        "UBT",                                 # 20yr 2x
        "TYD",                                 # 7-10yr 3x
        "TMF",                                 # 20yr 3x
        # Crypto leveraged
        "BITX", "BITU",                        # Bitcoin 2x
        "ETHU",                                # Ethereum 2x
        # Commodities leveraged
        "AGQ",                                 # silver 2x
        "UGL",                                 # gold 2x
    ],
}


def _flat(*categories: str) -> list[str]:
    """Flatten given categories into a single deduped list, preserving order."""
    seen: dict[str, None] = {}
    for cat in categories:
        for t in ETF_CATEGORIES.get(cat, []):
            if t not in seen:
                seen[t] = None
    return list(seen.keys())


# Compatibility wrappers — preserve the prior public API.
POPULAR_ETFS: list[str] = _flat(
    "broad_market", "factor", "sector", "industry", "thematic",
    "international_developed", "international_emerging",
    "bonds", "commodities", "real_estate", "currency",
    "volatility", "crypto",
)
INVERSE_ETFS: list[str] = ETF_CATEGORIES["inverse"]
LEVERAGED_ETFS: list[str] = ETF_CATEGORIES["leveraged"]


def etf_category_map() -> dict[str, str]:
    """Return ``{ticker: category}`` for every ETF in the curated lists.

    When a ticker appears in multiple categories (rare, e.g. inverse vol that's
    also under volatility), the *more specific* category wins by listing it
    later in the iteration order.
    """
    out: dict[str, str] = {}
    for cat, tickers in ETF_CATEGORIES.items():
        for t in tickers:
            out[t] = cat
    return out
