"""FinBERT sentiment scoring for stored news articles.

Scoring is decoupled from scraping: the news scrapers fetch and persist
articles, this module reads each ``data/news/<source>/<topic>/<year>/...``
JSON, scores any articles missing a ``sentiment`` field with FinBERT
(``ProsusAI/finbert``), and writes back. Resumable — already-scored
articles are skipped, so a crashed run can be re-invoked safely.

Device selection follows the standard preference order:
  1. CUDA (NVIDIA GPU) if available
  2. Apple Metal (MPS) if available
  3. CPU fallback

The output payload preserves all three class probabilities so downstream
features can use neutrality as a "noise" signal too:

    "sentiment": {
        "pos": 0.12, "neg": 0.78, "neutral": 0.10,
        "net": -0.66,   # pos - neg, bounded [-1, 1]
        "model": "ProsusAI/finbert",
    }

Usage:

    python -m stratlab.news.sentiment                    # score all unscored
    python -m stratlab.news.sentiment --sources npr ap   # subset
    python -m stratlab.news.sentiment --since 2024-01-01 # date filter

torch + transformers are an optional dependency. Install with::

    pip install -e ".[sentiment]"
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from stratlab.news.storage import NEWS_DIR

MODEL_NAME = "ProsusAI/finbert"
_MAX_INPUT_CHARS = 1200  # ~400-500 tokens; well under FinBERT's 512 cap

# Cached on first use; survives across calls within a process.
_model = None
_tokenizer = None
_device = None


@dataclass
class ScoreStats:
    files_scanned: int = 0
    articles_scored: int = 0
    articles_skipped: int = 0
    files_written: int = 0
    errors: int = 0
    by_source: dict[str, int] = field(default_factory=dict)


def get_device() -> str:
    """Pick the best available device: cuda > mps > cpu."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _ensure_loaded() -> None:
    """Lazy-load FinBERT on first use; cached process-wide."""
    global _model, _tokenizer, _device
    if _model is not None:
        return
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "FinBERT scoring requires `torch` and `transformers`. "
            "Install with `pip install -e \".[sentiment]\"`."
        ) from exc

    _device = get_device()
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    _model.eval()
    _model.to(_device)
    if _device == "cpu":
        torch.set_num_threads(max(1, (Path("/proc/cpuinfo").exists() and 4) or 4))


def _input_text(article: dict) -> str:
    """Title + lead paragraphs, truncated to ~_MAX_INPUT_CHARS chars."""
    parts = [article.get("title", ""), article.get("description", ""),
             article.get("content", "")]
    text = " ".join(p for p in parts if p).strip()
    if len(text) > _MAX_INPUT_CHARS:
        text = text[:_MAX_INPUT_CHARS]
    return text or article.get("title", "") or "(empty)"


def score_texts(texts: list[str], batch_size: int = 16) -> list[dict]:
    """Score a list of texts. Returns one dict per input."""
    if not texts:
        return []
    _ensure_loaded()
    import torch

    # ProsusAI/finbert label order is [positive, negative, neutral].
    label_order = ["positive", "negative", "neutral"]
    id2label = _model.config.id2label
    indices = {label: next(i for i, l in id2label.items() if l.lower() == label)
               for label in label_order}

    out: list[dict] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        enc = _tokenizer(chunk, padding=True, truncation=True,
                         max_length=512, return_tensors="pt").to(_device)
        with torch.no_grad():
            logits = _model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        for row in probs:
            pos = float(row[indices["positive"]])
            neg = float(row[indices["negative"]])
            neu = float(row[indices["neutral"]])
            out.append({
                "pos": round(pos, 4),
                "neg": round(neg, 4),
                "neutral": round(neu, 4),
                "net": round(pos - neg, 4),
                "model": MODEL_NAME,
            })
    return out


def _iter_day_files(news_dir: Path, sources: list[str] | None,
                    since: date | None) -> list[Path]:
    paths: list[Path] = []
    if not news_dir.exists():
        return paths
    src_dirs = [news_dir / s for s in sources] if sources else [
        d for d in news_dir.iterdir() if d.is_dir() and not d.name.startswith("_")
    ]
    for src_dir in src_dirs:
        if not src_dir.is_dir():
            continue
        for json_file in src_dir.rglob("*.json"):
            if since is not None:
                # Filenames are <source>-<topic>-<YYYY-MM-DD>.json
                stem = json_file.stem
                try:
                    file_date = date.fromisoformat(stem[-10:])
                except ValueError:
                    continue
                if file_date < since:
                    continue
            paths.append(json_file)
    return paths


def score_news_dir(
    news_dir: Path = NEWS_DIR,
    sources: list[str] | None = None,
    since: date | None = None,
    batch_size: int = 16,
    verbose: bool = True,
) -> ScoreStats:
    """Walk per-day JSONs and score any articles missing a `sentiment` field.

    File-level atomic: a file is rewritten only after all its articles are
    scored. A crash mid-file means that file is unchanged on next run.
    """
    stats = ScoreStats()
    files = _iter_day_files(news_dir, sources, since)
    if verbose:
        device = get_device()
        print(f"FinBERT sentiment: {len(files)} day-files to scan (device: {device})")

    for fpath in files:
        stats.files_scanned += 1
        try:
            with fpath.open() as f:
                articles = json.load(f)
        except Exception as exc:
            if verbose:
                print(f"  [load fail] {fpath}: {exc}")
            stats.errors += 1
            continue

        unscored_keys = [k for k, a in articles.items()
                         if isinstance(a, dict) and "sentiment" not in a]
        if not unscored_keys:
            stats.articles_skipped += len(articles)
            continue

        texts = [_input_text(articles[k]) for k in unscored_keys]
        try:
            scores = score_texts(texts, batch_size=batch_size)
        except Exception as exc:
            if verbose:
                print(f"  [score fail] {fpath}: {exc}")
            stats.errors += 1
            continue

        for k, sc in zip(unscored_keys, scores):
            articles[k]["sentiment"] = sc
            stats.articles_scored += 1
            source = fpath.parts[-4] if len(fpath.parts) >= 4 else "unknown"
            stats.by_source[source] = stats.by_source.get(source, 0) + 1
        stats.articles_skipped += len(articles) - len(unscored_keys)

        tmp = fpath.with_suffix(".json.tmp")
        with tmp.open("w") as f:
            json.dump(articles, f, separators=(",", ":"))
        tmp.replace(fpath)
        stats.files_written += 1

        if verbose:
            print(f"  + {fpath.relative_to(news_dir)}: scored {len(unscored_keys)}")

    return stats


def _print_summary(stats: ScoreStats) -> None:
    print("\n" + "=" * 60)
    print("FinBERT scoring summary")
    print(f"  files scanned          : {stats.files_scanned}")
    print(f"  files written          : {stats.files_written}")
    print(f"  articles scored        : {stats.articles_scored}")
    print(f"  articles skipped (already scored): {stats.articles_skipped}")
    print(f"  errors                 : {stats.errors}")
    if stats.by_source:
        print("  by source              :")
        for source, n in sorted(stats.by_source.items(), key=lambda kv: -kv[1]):
            print(f"    {source:8s}: {n}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sources", nargs="*",
                        help="Only score these sources (default: all).")
    parser.add_argument("--since", type=str, default=None,
                        help="Only score files dated YYYY-MM-DD or later.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    since = date.fromisoformat(args.since) if args.since else None
    stats = score_news_dir(
        sources=args.sources, since=since,
        batch_size=args.batch_size, verbose=not args.quiet,
    )
    _print_summary(stats)
    return 0 if stats.errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
