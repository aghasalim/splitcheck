"""Make the README figures: python scripts/make_figures.py -> docs/*.png

1. docs/demo_run.png: a real run of the CLI on a split with four planted leaks,
   and on a clean split of the same data. The output text is captured, not typed.
2. docs/checks.png: what each of the four checks compares.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"


def example(seed=0, n_users=400, rows_per_user=5):
    """Event log: one row per user-day, with a binary target and a leaky column."""
    r = np.random.default_rng(seed)
    n = n_users * rows_per_user
    df = pd.DataFrame({
        "user_id": np.repeat(np.arange(n_users), rows_per_user),
        "day": pd.Timestamp("2024-01-01") + pd.to_timedelta(r.integers(0, 120, n), unit="D"),
        "amount": r.gamma(2.0, 30.0, n).round(2),
        "visits": r.poisson(4, n),
    })
    df["y"] = (r.random(n) < 1 / (1 + np.exp(-(df["visits"] - 4) / 2))).astype(int)
    df["refund_flag"] = df["y"]  # recorded after the outcome: a target leak
    return df


def leaky_split(df, seed=0):
    """Random row split (users and days mixed) plus 50 train rows copied into test."""
    r = np.random.default_rng(seed)
    is_test = r.random(len(df)) < 0.25
    train, test = df[~is_test], df[is_test]
    return train, pd.concat([test, train.sample(50, random_state=seed)])


def clean_split(df):
    """Split by user and by time: test users are new, and test starts after train ends."""
    test_users = df["user_id"] >= 300
    cut = pd.Timestamp("2024-04-01")
    train = df[~test_users & (df["day"] < cut)].drop(columns="refund_flag")
    test = df[test_users & (df["day"] >= cut)].drop(columns="refund_flag")
    return train, test


def run_cli(train, test, tmp, name):
    tr, te = Path(tmp) / f"{name}_train.csv", Path(tmp) / f"{name}_test.csv"
    train.to_csv(tr, index=False)
    test.to_csv(te, index=False)
    args = [f"{name}_train.csv", f"{name}_test.csv", "--target", "y", "--group", "user_id", "--time", "day"]
    p = subprocess.run([sys.executable, str(ROOT / "splitcheck.py"), *args], cwd=tmp,
                       capture_output=True, text=True)
    return "$ splitcheck " + " ".join(args), p.stdout.rstrip().splitlines(), p.returncode


def terminal(runs, path):
    lines = []
    for cmd, out, code in runs:
        lines += [(cmd, "#c9d1d9")]
        for o in out:
            colors = {"FAIL": "#ff7b72", "WARN": "#e3b341", "OK": "#7ee787"}
            lines.append((o, colors.get(o.split(":")[0], "#c9d1d9")))
        lines += [(f"$ echo $?\n{code}", "#8b949e"), ("", None)]
    text_lines = [(s, c) for block, c in lines[:-1] for s in (block.split("\n") or [""])]
    h = 0.32 + 0.22 * len(text_lines)
    fig = plt.figure(figsize=(9, h))
    fig.patch.set_facecolor("#0d1117")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, h)
    for i, (s, c) in enumerate(text_lines):
        ax.text(0.02, h - 0.25 - 0.22 * i, s, color=c or "#c9d1d9", family="DejaVu Sans Mono",
                fontsize=9.5, va="center", parse_math=False)
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def checks_diagram(path):
    checks = [
        ("Duplicates", "hash every feature row;\ntest rows also in train", "--max-dup"),
        ("Shared ids", "values of an id column\nfound in both splits", "--group COL"),
        ("Time overlap", "latest train time vs\nearliest test time", "--time COL"),
        ("Target leak", "AUC of each column alone\non the train target", "--auc-threshold"),
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 4)
    ax.set_axis_off()
    box = dict(boxstyle="round,pad=0.02,rounding_size=0.08", linewidth=1.2)
    for y, label in [(2.75, "train.csv"), (1.25, "test.csv")]:
        ax.add_patch(FancyBboxPatch((0.15, y - 0.35), 1.3, 0.7, facecolor="#eef2f7", edgecolor="#4a5568", **box))
        ax.text(0.8, y, label, ha="center", va="center", fontsize=10, family="DejaVu Sans Mono")
    ys = [3.45, 2.5, 1.55, 0.6]
    for (name, what, flag), y in zip(checks, ys):
        sev = "WARN" if name == "Target leak" else "FAIL"
        ax.add_patch(FancyBboxPatch((2.4, y - 0.36), 3.6, 0.72, facecolor="white", edgecolor="#4a5568", **box))
        ax.text(2.55, y + 0.13, name, fontsize=10, weight="bold", va="center")
        ax.text(2.55, y - 0.15, what.replace("\n", " "), fontsize=8, va="center", color="#333")
        ax.text(5.85, y + 0.13, flag, fontsize=7.5, va="center", ha="right", family="DejaVu Sans Mono", color="#555")
        for sy in (2.75,) if name == "Target leak" else (2.75, 1.25):  # target leak reads train only
            ax.annotate("", (2.38, y), (1.47, sy), arrowprops=dict(arrowstyle="-", color="#a0aec0", lw=0.8))
        color = "#c0392b" if sev == "FAIL" else "#b7791f"
        ax.annotate("", (6.55, y), (6.02, y), arrowprops=dict(arrowstyle="->", color="#4a5568", lw=1))
        ax.text(6.6, y, sev, fontsize=9.5, weight="bold", color=color, va="center", family="DejaVu Sans Mono")
    ax.text(6.6, 0.05, "any FAIL: exit 1", fontsize=8.5, color="#333", family="DejaVu Sans Mono")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    DOCS.mkdir(exist_ok=True)
    df = example()
    with tempfile.TemporaryDirectory() as tmp:
        runs = [run_cli(*leaky_split(df), tmp, "leaky"), run_cli(*clean_split(df), tmp, "clean")]
    for cmd, out, code in runs:
        print(cmd, *out, f"exit {code}", sep="\n")
    terminal(runs, DOCS / "demo_run.png")
    checks_diagram(DOCS / "checks.png")
