"""
dimensions.py
-------------
Load all five MLB Pitch Data (2015-2018) CSV files and report:
  - High-level shape summary (rows, columns, memory usage)
  - Per-dataset column detail (dtype, null count, null %)
  - First-5-row preview of the core pitches table

Dataset folder: DATASET/ (relative to this script)
"""

from pathlib import Path
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
HERE    = Path(__file__).parent
DATASET = HERE / "DATASET"

# ── Load ───────────────────────────────────────────────────────────────────────
pitches      = pd.read_csv(DATASET / "pitches.csv")
atbats       = pd.read_csv(DATASET / "atbats.csv")
ejections    = pd.read_csv(DATASET / "ejections.csv")
games        = pd.read_csv(DATASET / "games.csv")
player_names = pd.read_csv(DATASET / "player_names.csv")

dfs = {
    "Pitches":   pitches,
    "At-Bats":   atbats,
    "Ejections": ejections,
    "Games":     games,
    "Players":   player_names,
}

# ── High-level shape summary ───────────────────────────────────────────────────
print("=" * 60)
print("DATASET DIMENSIONS SUMMARY")
print("=" * 60)
print(f"{'Dataset':<12} {'Rows':>12} {'Columns':>9} {'Memory':>10}")
print("-" * 60)
for name, df in dfs.items():
    mem = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
    print(f"{name:<12} {df.shape[0]:>12,} {df.shape[1]:>9} {mem:>9.2f}MB")

# ── Per-dataset column detail ──────────────────────────────────────────────────
for name, df in dfs.items():
    print(f"\n{'=' * 60}")
    print(f"  {name.upper()}")
    print(f"{'=' * 60}")
    print(f"  Shape: {df.shape[0]:,} rows  x  {df.shape[1]} columns")
    print(f"\n  {'Column':<30} {'Dtype':<12} {'Nulls':>8} {'Null %':>8}")
    print(f"  {'-' * 60}")
    for col in df.columns:
        nulls = df[col].isna().sum()
        pct   = (nulls / len(df)) * 100
        print(f"  {col:<30} {str(df[col].dtype):<12} {nulls:>8,} {pct:>7.1f}%")

# ── Pitches preview ────────────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("PITCHES — First 5 Rows")
print("=" * 60)
print(pitches.head().to_string())
