"""Hypothesis pre-commit registry for the strategies arena.

Lets parallel generator agents avoid converging on the same idea by
recording their hypothesis BEFORE running a backtest. Subsequent agents
read prior intents and pick non-overlapping angles.

File: ``tmp/arena/intents.csv`` — single CSV across all generations,
append-only, exclusive ``flock`` on every write.

Status lifecycle::

    committed                                  ← agent reserves the slot
       │
       ├──→ submitted (strategy passed gates)  ← submit.py auto-marks
       └──→ abandoned (rejected / bailed)      ← submit.py or manual

Typical agent workflow:

1. ``python -m stratlab.arena.intents read --gen N``   # see what's claimed
2. Form a hypothesis that doesn't overlap
3. ``python -m stratlab.arena.intents commit --agent-id sonnet-1 \\``
   ``    --gen N --hypothesis "<one sentence>"``        # returns intent_id
4. Write code
5. ``python -m stratlab.arena.submit ... --intent-id <id>`` (auto-marks)

Note on race conditions: ``flock`` serializes writes but two agents can
both ``read`` an empty file before either commits. The lock window is
seconds; agent thinking time is minutes. In practice, distinct agents
launched simultaneously will still commit different ideas because their
LLM responses diverge naturally. If you need stronger guarantees,
stagger agent launches by 30 seconds.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from stratlab.arena import config


COLUMNS = [
    "intent_id",
    "timestamp",
    "generation",
    "agent_id",
    "hypothesis",
    "parent_id",
    "status",       # "committed" | "submitted" | "abandoned"
    "strategy_id",  # filled in on submit
    "notes",
]

_VALID_STATUSES = ("committed", "submitted", "abandoned")


def _intents_path() -> Path:
    return config.ARENA_DIR / "intents.csv"


@contextmanager
def _locked(path: Path, mode: str):
    """Open ``path`` with an exclusive ``flock``. Auto-creates parent dir."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.touch()
    fp = open(path, mode)
    try:
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
        yield fp
    finally:
        fcntl.flock(fp.fileno(), fcntl.LOCK_UN)
        fp.close()


def _read_rows(path: Path | None = None) -> list[dict]:
    path = path or _intents_path()
    if not path.exists() or path.stat().st_size == 0:
        return []
    with _locked(path, "r") as fp:
        reader = csv.DictReader(fp)
        return list(reader)


def _write_rows(rows: list[dict], path: Path | None = None) -> None:
    path = path or _intents_path()
    with _locked(path, "w") as fp:
        writer = csv.DictWriter(fp, fieldnames=COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c, "") for c in COLUMNS})


def commit_intent(
    agent_id: str,
    generation: int,
    hypothesis: str,
    parent_id: str = "",
    notes: str = "",
    *,
    path: Path | None = None,
) -> str:
    """Reserve an intent. Returns a unique ``intent_id``."""
    intent_id = f"ic_{uuid.uuid4().hex[:8]}"
    row = {
        "intent_id": intent_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "generation": str(generation),
        "agent_id": agent_id,
        "hypothesis": hypothesis,
        "parent_id": parent_id,
        "status": "committed",
        "strategy_id": "",
        "notes": notes,
    }
    rows = _read_rows(path)
    rows.append(row)
    _write_rows(rows, path)
    return intent_id


def mark_status(
    intent_id: str,
    status: str,
    strategy_id: str = "",
    notes: str = "",
    *,
    path: Path | None = None,
) -> None:
    """Update the status (and optionally strategy_id / notes) of an intent."""
    if status not in _VALID_STATUSES:
        raise ValueError(f"unknown status {status!r} (valid: {_VALID_STATUSES})")
    rows = _read_rows(path)
    found = False
    for r in rows:
        if r["intent_id"] == intent_id:
            r["status"] = status
            if strategy_id:
                r["strategy_id"] = strategy_id
            if notes:
                r["notes"] = notes
            found = True
            break
    if not found:
        raise ValueError(f"intent_id {intent_id!r} not found")
    _write_rows(rows, path)


def read_intents(
    generation: int | None = None,
    status: str | None = None,
    *,
    path: Path | None = None,
) -> list[dict]:
    """Read intents, optionally filtered."""
    rows = _read_rows(path)
    if generation is not None:
        rows = [r for r in rows if r["generation"] == str(generation)]
    if status is not None:
        rows = [r for r in rows if r["status"] == status]
    return rows


_STATUS_MARKER = {"committed": "[?]", "submitted": "[+]", "abandoned": "[x]"}


def format_for_prompt(intents: list[dict]) -> str:
    """Format a list of intents for human (or agent-prompt) display."""
    if not intents:
        return "(no current intents)"
    lines = []
    for r in intents:
        marker = _STATUS_MARKER.get(r["status"], "[?]")
        sid_part = f" -> {r['strategy_id']}" if r.get("strategy_id") else ""
        lines.append(
            f"  {marker} g{r['generation']} {r['agent_id']}: "
            f"{r['hypothesis']}{sid_part}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_read = sub.add_parser("read", help="List existing intents.")
    p_read.add_argument("--gen", type=int, default=None,
                        help="Filter to this generation.")
    p_read.add_argument("--status", choices=_VALID_STATUSES, default=None)

    p_commit = sub.add_parser("commit", help="Reserve a new intent.")
    p_commit.add_argument("--agent-id", required=True)
    p_commit.add_argument("--gen", type=int, required=True)
    p_commit.add_argument("--hypothesis", required=True,
                          help="One-sentence rationale for your strategy.")
    p_commit.add_argument("--parent-id", default="")
    p_commit.add_argument("--notes", default="")

    p_mark = sub.add_parser("mark", help="Update the status of an intent.")
    p_mark.add_argument("intent_id")
    p_mark.add_argument("status", choices=_VALID_STATUSES)
    p_mark.add_argument("--strategy-id", default="")
    p_mark.add_argument("--notes", default="")

    args = parser.parse_args(argv)

    if args.cmd == "read":
        intents = read_intents(generation=args.gen, status=args.status)
        if not intents:
            print("(no intents)")
            return 0
        print(format_for_prompt(intents))
        return 0

    if args.cmd == "commit":
        intent_id = commit_intent(
            agent_id=args.agent_id,
            generation=args.gen,
            hypothesis=args.hypothesis,
            parent_id=args.parent_id,
            notes=args.notes,
        )
        gen_intents = read_intents(generation=args.gen)
        print(f"intent_id: {intent_id}")
        print(f"intents_in_gen_{args.gen}: {len(gen_intents)}")
        return 0

    if args.cmd == "mark":
        mark_status(
            args.intent_id, args.status,
            strategy_id=args.strategy_id, notes=args.notes,
        )
        print(f"marked {args.intent_id} -> {args.status}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
