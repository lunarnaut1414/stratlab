"""Curated ticker lists for ETFs and ETPs.

These lists are intentionally hardcoded — there's no clean free source for
"all major ETFs" (ETFdb blocks scraping, Yahoo doesn't expose a list endpoint),
and ETF tickers turn over slowly, so a curated snapshot stays useful.

Tickers are Yahoo-formatted (use ``-`` not ``.``). Some niche or recently
launched ETFs may be missing — add what you need.
"""
from __future__ import annotations


# Broad market, sector, international, bonds, commodities, real estate,
# currency, factor, thematic, crypto, and volatility ETFs.
# Long-side, unlevered (or 1x).
POPULAR_ETFS: list[str] = [
    # --- Broad US equity ---
    "SPY", "IVV", "VOO", "VTI", "QQQ", "QQQM", "IWM", "DIA",
    "MDY", "IJH", "IJR", "RSP", "SCHX", "SCHB", "SCHG", "SCHV",
    "VTV", "VUG", "VYM", "VIG", "SCHD", "DVY", "NOBL",

    # --- Style / factor ---
    "MTUM", "QUAL", "USMV", "VLUE", "SIZE", "SPLV", "MOAT",

    # --- Sector SPDRs + alternates ---
    "XLF", "XLE", "XLU", "XLK", "XLI", "XLY", "XLP", "XLV", "XLB", "XLRE", "XLC",
    "VFH", "VHT", "VGT", "VDE", "VCR", "VDC", "VIS", "VPU",

    # --- Industries / thematic ---
    "SMH", "SOXX",                        # semis
    "IBB", "XBI",                          # biotech
    "GDX", "GDXJ", "RING", "SIL",          # gold / silver miners
    "OIH", "XOP",                          # energy services / E&P
    "KIE", "KBE", "KRE",                   # insurance / banks
    "ITA", "XAR", "PPA",                   # defense / aerospace
    "IGV", "SKYY", "WCLD",                 # software / cloud
    "HACK", "CIBR",                        # cybersecurity
    "ROBO", "BOTZ", "IRBO",                # robotics / AI
    "ARKK", "ARKG", "ARKW", "ARKQ", "ARKF",# innovation
    "ICLN", "TAN", "FAN", "PBW",           # clean energy
    "LIT",                                  # lithium / battery
    "URA", "URNM",                         # uranium
    "JETS",                                 # airlines
    "ESPO", "HERO",                        # gaming / esports
    "PEJ",                                  # entertainment
    "MOO",                                  # agriculture
    "PHO", "FIW",                          # water
    "KWEB", "CQQQ", "FXI", "MCHI", "ASHR", # China internet / equity
    "BLOK",                                 # blockchain equities

    # --- International developed ---
    "EFA", "IEFA", "VEA", "SCHF", "SPDW",
    "EWJ", "DXJ",                          # Japan
    "EWG",                                  # Germany
    "EWU",                                  # UK
    "EWA",                                  # Australia
    "EWC",                                  # Canada
    "EWQ",                                  # France
    "EWI",                                  # Italy
    "EWP",                                  # Spain
    "EWN",                                  # Netherlands
    "EWL",                                  # Switzerland

    # --- International emerging ---
    "EEM", "IEMG", "VWO", "SCHE", "SPEM",
    "INDA", "EPI",                         # India
    "EWZ", "EWW",                          # Brazil, Mexico
    "EWY",                                  # Korea
    "EWT",                                  # Taiwan
    "EZA",                                  # South Africa
    "TUR",                                  # Turkey
    "ARGT",                                 # Argentina
    "VNM",                                  # Vietnam

    # --- Bonds ---
    "TLT", "IEF", "SHY", "BIL",            # treasury duration ladder
    "AGG", "BND", "SCHZ",                  # aggregate
    "LQD", "VCIT", "VCSH",                 # corporate
    "HYG", "JNK", "USHY",                  # high yield
    "EMB", "PCY",                          # emerging market debt
    "TIP", "VTIP", "SCHP",                 # TIPS
    "MBB",                                  # mortgage-backed
    "MUB", "VTEB",                         # municipals
    "PFF", "PGX",                          # preferred
    "SHV", "SGOV",                         # ultra-short
    "BNDX",                                 # international agg

    # --- Commodities ---
    "GLD", "IAU", "SGOL",                  # gold
    "SLV", "SIVR",                         # silver
    "PPLT", "PALL",                        # platinum / palladium
    "USO", "BNO", "UCO",                   # oil
    "UNG", "BOIL",                         # natgas
    "DBC", "PDBC",                         # broad commodity
    "DBA", "CORN", "WEAT", "SOYB",         # agriculture
    "COPX", "CPER",                        # copper

    # --- Real estate ---
    "VNQ", "IYR", "SCHH", "RWR",
    "VNQI",                                 # international real estate
    "REM", "MORT",                         # mortgage REITs

    # --- Currency ---
    "UUP", "UDN",                          # dollar bull / bear
    "FXE", "FXY", "FXB", "FXC", "FXA",     # major fx
    "CYB",                                  # chinese yuan

    # --- Volatility ---
    "VXX", "VIXY", "VIXM",                 # VIX-tracking (decay-prone)

    # --- Crypto-adjacent ---
    "GBTC", "ETHE",                        # Grayscale trusts
    "BITO",                                 # Bitcoin futures
    "IBIT", "FBTC", "BITB", "ARKB", "HODL",# spot Bitcoin
    "ETHA", "FETH", "ETHW",                # spot Ethereum
]


# Inverse / short ETFs (1x, 2x, 3x). Going long these is equivalent to going
# short the underlying (without margin/borrow), at the cost of daily-rebalance
# decay on multi-day holds.
INVERSE_ETFS: list[str] = [
    # --- Broad US ---
    "SH", "SDS", "SPXU", "SPXS",           # SPY -1x / -2x / -3x
    "PSQ", "QID", "SQQQ",                  # QQQ -1x / -2x / -3x
    "RWM", "TWM", "SRTY",                  # IWM -1x / -2x / -3x
    "DOG", "DXD", "SDOW",                  # DIA -1x / -2x / -3x
    "SPDN",                                 # alternate -1x SPY

    # --- Sectors / industries ---
    "SOXS",                                 # semis -3x
    "DRV",                                  # real estate -3x
    "LABD",                                 # biotech -3x
    "FAZ",                                  # financials -3x
    "ERY",                                  # energy -3x
    "TZA",                                  # small-cap -3x
    "DUST",                                 # gold miners -2x
    "SCO",                                  # oil -2x
    "KOLD",                                 # natgas -2x

    # --- Bonds ---
    "TBF", "TBT", "TMV",                   # 20yr treasury -1x / -2x / -3x
    "PST",                                  # 7-10yr -2x

    # --- International ---
    "EUM", "EDZ",                          # EM -1x / -3x
    "EFZ",                                  # EAFE -1x
    "YANG",                                 # China -3x
]


# 2x and 3x leveraged long ETFs. Daily-rebalanced; multi-day holds drift from
# the simple multiple of the underlying due to volatility decay.
LEVERAGED_ETFS: list[str] = [
    # --- Broad US ---
    "SSO", "UPRO",                         # SPY 2x / 3x
    "QLD", "TQQQ",                         # QQQ 2x / 3x
    "DDM", "UDOW",                         # DIA 2x / 3x
    "URTY", "UWM",                         # Russell 3x / 2x
    "MVV",                                  # mid-cap 2x

    # --- Sectors / industries ---
    "ROM", "REW",                          # tech 2x / -2x (REW inverse, but Rydex)
    "FAS",                                  # financials 3x
    "ERX", "GUSH",                         # energy 3x
    "CURE",                                 # healthcare 3x
    "SOXL",                                 # semis 3x
    "DRN",                                  # real estate 3x
    "NUGT", "JNUG",                        # gold miners 2x
    "LABU",                                 # biotech 3x
    "DPST",                                 # regional banks 3x
    "RETL",                                 # retail 3x
    "WEBL",                                 # internet 3x
    "DFEN",                                 # defense 3x

    # --- International ---
    "YINN",                                 # China 3x bull
    "EDC",                                  # EM 3x bull
    "EZJ",                                  # Japan 2x

    # --- Bonds ---
    "UBT",                                  # 20yr 2x
    "TYD",                                  # 7-10yr 3x
    "TMF",                                  # 20yr 3x

    # --- Crypto leveraged ---
    "BITX",                                 # Bitcoin 2x
    "BITU",                                 # Bitcoin 2x (alt)
    "ETHU",                                 # Ethereum 2x

    # --- Commodities ---
    "AGQ",                                  # silver 2x
    "UGL",                                  # gold 2x
    "BOIL",                                 # natgas 2x (also above; dedup at universe level)
]
