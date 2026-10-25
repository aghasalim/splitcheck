"""splitcheck: catch train/test leakage in a tabular split before you trust the score."""
import argparse
import sys

import pandas as pd


def duplicates(train, test, features):
    """Fraction of test rows whose feature values also appear in train."""
    key = lambda d: pd.util.hash_pandas_object(d[features], index=False)
    return key(test).isin(set(key(train))).mean()


def overlap(train, test, col):
    """Values of `col` present in both splits, and the share of test rows they cover."""
    shared = set(train[col].dropna()) & set(test[col].dropna())
    return len(shared), test[col].isin(shared).mean()


def time_overlap(train, test, col):
    """Seconds by which train reaches past the start of test (<= 0 means a clean cut)."""
    t, s = pd.to_datetime(train[col]), pd.to_datetime(test[col])
    return (t.max() - s.min()).total_seconds()


def single_feature_auc(df, target, col):
    """AUC of one column as a score for a binary target, orientation-free (0.5 to 1)."""
    x = df[col]
    if not pd.api.types.is_numeric_dtype(x):
        # target-mean encode; optimistic on high-cardinality columns, which is the point
        x = x.map(df.groupby(col)[target].mean())
    r = x.rank()
    pos = df[target] == df[target].max()
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0 or x.nunique() < 2:
        return 0.5
    auc = (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return max(auc, 1 - auc)


def run(train, test, target, groups=(), time=None, auc_threshold=0.98, max_dup=0.0):
    """Return a list of (severity, message). Severity is FAIL or WARN."""
    out = []
    features = [c for c in train.columns if c != target and c in test.columns]
    skip = set(groups) | ({time} if time else set())
    d = duplicates(train, test, features)
    if d > max_dup:
        out.append(("FAIL", f"{d:.1%} of test rows are exact duplicates of train rows"))
    for g in groups:
        n, share = overlap(train, test, g)
        if n:
            out.append(("FAIL", f"{n} '{g}' values in both splits, covering {share:.1%} of test rows"))
    if time:
        s = time_overlap(train, test, time)
        if s > 0:
            out.append(("FAIL", f"train extends {s / 86400:.1f} days past the first test row in '{time}'"))
    if train[target].nunique() == 2:
        for c in features:
            if c in skip:
                continue
            a = single_feature_auc(train, target, c)
            if a >= auc_threshold:
                out.append(("WARN", f"'{c}' alone scores AUC {a:.3f}: possible target leak"))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(prog="splitcheck", description=__doc__)
    p.add_argument("train")
    p.add_argument("test")
    p.add_argument("--target", required=True)
    p.add_argument("--group", action="append", default=[], help="id/group column; repeatable")
    p.add_argument("--time", help="timestamp column; train must end before test starts")
    p.add_argument("--auc-threshold", type=float, default=0.98)
    p.add_argument("--max-dup", type=float, default=0.0, help="allowed duplicate fraction")
    a = p.parse_args(argv)
    found = run(pd.read_csv(a.train), pd.read_csv(a.test), a.target, a.group, a.time,
                a.auc_threshold, a.max_dup)
    for sev, msg in found:
        print(f"{sev}: {msg}")
    if not found:
        print("OK: no leakage found by these checks (absence of a flag is not proof)")
    return 1 if any(s == "FAIL" for s, _ in found) else 0


if __name__ == "__main__":
    sys.exit(main())
