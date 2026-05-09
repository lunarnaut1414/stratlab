"""Daily aggregated sentiment features from scored news articles.

Reads the per-day JSON files produced by :mod:`stratlab.news.npr` (etc.)
*after* :mod:`stratlab.news.sentiment` has annotated each article with a
``sentiment`` payload. Aggregates to one row per (date, source, topic).

The default output is a wide DataFrame indexed by date with one column
per (source, topic) holding the mean **net** sentiment (= pos - neg,
bounded ``[-1, 1]``). For richer feature engineering — e.g., using the
fraction of neutral coverage as a "story importance" signal — pass
``breakdown=True`` to get the full pos/neg/neutral/count breakdown.

Example::

    from stratlab.news.features import daily_sentiment

    sent = daily_sentiment(start="2024-01-01", end="2024-12-31",
                            sources=["npr", "ap"], topics=["business", "economy"])
    # → DataFrame indexed by date with columns
    #   ('npr', 'business'), ('npr', 'economy'), ('ap', 'business'), ...

    # Or just one number per day across everything:
    overall = sent.mean(axis=1)
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import pandas as pd

from stratlab.news.storage import NEWS_DIR

_METRIC_KEYS = ("net", "pos", "neg", "neutral")


def _coerce_date(d: date | str | None) -> date | None:
    if d is None or isinstance(d, date):
        return d
    return date.fromisoformat(d)


def _iter_day_files(
    news_dir: Path,
    sources: list[str] | None,
    topics: list[str] | None,
    start: date | None,
    end: date | None,
):
    if not news_dir.exists():
        return
    src_dirs = [news_dir / s for s in sources] if sources else [
        d for d in news_dir.iterdir() if d.is_dir() and not d.name.startswith("_")
    ]
    for src_dir in src_dirs:
        if not src_dir.is_dir():
            continue
        source = src_dir.name
        topic_dirs = [src_dir / t for t in topics] if topics else [
            d for d in src_dir.iterdir() if d.is_dir()
        ]
        for topic_dir in topic_dirs:
            if not topic_dir.is_dir():
                continue
            topic = topic_dir.name
            for json_file in topic_dir.rglob("*.json"):
                stem = json_file.stem
                try:
                    file_date = date.fromisoformat(stem[-10:])
                except ValueError:
                    continue
                if start is not None and file_date < start:
                    continue
                if end is not None and file_date > end:
                    continue
                yield source, topic, file_date, json_file


def daily_sentiment(
    start: date | str | None = None,
    end: date | str | None = None,
    sources: list[str] | None = None,
    topics: list[str] | None = None,
    breakdown: bool = False,
    news_dir: Path = NEWS_DIR,
) -> pd.DataFrame:
    """Aggregate scored news articles into daily per-(source, topic) features.

    ``breakdown=False`` (default): one column per (source, topic) holding
    mean **net** sentiment. ``breakdown=True``: a MultiIndex column for
    every metric (net / pos / neg / neutral) plus an ``article_count``.

    Articles without a ``sentiment`` field (i.e., not yet scored by
    :mod:`stratlab.news.sentiment`) are silently skipped.
    """
    start_d = _coerce_date(start)
    end_d = _coerce_date(end)

    # accumulator[(source, topic, day)] = list of sentiment dicts
    bucket: dict[tuple[str, str, date], list[dict]] = defaultdict(list)
    for source, topic, day, jpath in _iter_day_files(
        news_dir, sources, topics, start_d, end_d
    ):
        try:
            with jpath.open() as f:
                articles = json.load(f)
        except Exception:
            continue
        for art in articles.values():
            if not isinstance(art, dict):
                continue
            sent = art.get("sentiment")
            if not isinstance(sent, dict):
                continue
            bucket[(source, topic, day)].append(sent)

    if not bucket:
        return pd.DataFrame()

    rows = []
    for (source, topic, day), sents in bucket.items():
        n = len(sents)
        agg = {k: sum(s.get(k, 0.0) for s in sents) / n for k in _METRIC_KEYS}
        rows.append({
            "date": pd.Timestamp(day),
            "source": source, "topic": topic,
            "article_count": n,
            **agg,
        })
    long_df = pd.DataFrame(rows).sort_values(["date", "source", "topic"])

    if not breakdown:
        wide = long_df.pivot_table(
            index="date", columns=["source", "topic"], values="net",
        )
        wide.columns.names = ["source", "topic"]
        return wide

    metrics = list(_METRIC_KEYS) + ["article_count"]
    wide = long_df.pivot_table(
        index="date", columns=["source", "topic"], values=metrics,
    )
    # plotly-style column order: (source, topic, metric)
    wide.columns = wide.columns.reorder_levels(["source", "topic", None])
    wide.columns.names = ["source", "topic", "metric"]
    wide = wide.sort_index(axis=1)
    return wide
