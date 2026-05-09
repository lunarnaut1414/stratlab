"""Curated lists of continuous futures contracts, by category.

Yahoo's continuous-contract symbols use the ``=F`` suffix. These are the
back-adjusted continuous series (not specific contract months), which is what
you want for backtesting trend / carry / spread strategies. For specific
expirations you'd use ``CLF26.NYM`` style symbols, but those are out of scope
here.
"""
from __future__ import annotations


FUTURES_CATEGORIES: dict[str, list[str]] = {
    "energy": [
        "CL=F",      # WTI crude oil
        "BZ=F",      # Brent crude
        "NG=F",      # Henry Hub natural gas
        "RB=F",      # RBOB gasoline
        "HO=F",      # NY Harbor ULSD (heating oil)
    ],
    "metals": [
        "GC=F",      # gold
        "SI=F",      # silver
        "PL=F",      # platinum
        "PA=F",      # palladium
        "HG=F",      # copper
    ],
    "grains": [
        "ZC=F",      # corn
        "ZW=F",      # wheat (Chicago)
        "KE=F",      # wheat (Kansas City hard red winter)
        "ZS=F",      # soybeans
        "ZM=F",      # soybean meal
        "ZL=F",      # soybean oil
        "ZO=F",      # oats
        "ZR=F",      # rough rice
    ],
    "softs": [
        "KC=F",      # coffee
        "CC=F",      # cocoa
        "CT=F",      # cotton
        "SB=F",      # sugar #11
        "OJ=F",      # orange juice
    ],
    "meats": [
        "LE=F",      # live cattle
        "GF=F",      # feeder cattle
        "HE=F",      # lean hogs
    ],
    "lumber": [
        "LBR=F",     # lumber (CME post-2022 contract; LB=F was discontinued)
    ],
    "equity_index": [
        "ES=F",      # E-mini S&P 500
        "NQ=F",      # E-mini NASDAQ-100
        "YM=F",      # E-mini Dow
        "RTY=F",     # E-mini Russell 2000
        "MES=F",     # Micro E-mini S&P 500
        "MNQ=F",     # Micro E-mini NASDAQ-100
    ],
    "rates": [
        "ZB=F",      # 30-year T-bond
        "ZN=F",      # 10-year T-note
        "ZF=F",      # 5-year T-note
        "ZT=F",      # 2-year T-note
        "ZQ=F",      # 30-day Fed Funds
        "GE=F",      # Eurodollar (legacy; replaced by SR3 in 2023)
    ],
    "currency": [
        "6E=F",      # Euro FX
        "6J=F",      # Japanese Yen
        "6B=F",      # British Pound
        "6S=F",      # Swiss Franc
        "6C=F",      # Canadian Dollar
        "6A=F",      # Australian Dollar
        "6N=F",      # New Zealand Dollar
        "6M=F",      # Mexican Peso
    ],
    "crypto": [
        "BTC=F",     # CME Bitcoin
        "MBT=F",     # CME Micro Bitcoin
        "ETH=F",     # CME Ether
        "MET=F",     # CME Micro Ether
    ],
}


def futures_category_map() -> dict[str, str]:
    out: dict[str, str] = {}
    for cat, tickers in FUTURES_CATEGORIES.items():
        for t in tickers:
            out[t] = cat
    return out
