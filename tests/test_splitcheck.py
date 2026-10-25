import numpy as np
import pandas as pd

import splitcheck as sc


def frame(n, seed):
    r = np.random.default_rng(seed)
    return pd.DataFrame({"x": r.normal(size=n), "y": r.integers(0, 2, n)})


def test_clean_split_passes():
    assert sc.run(frame(500, 0), frame(500, 1), "y") == []


def test_duplicates_fail():
    tr = frame(500, 0)
    te = pd.concat([frame(400, 1), tr.iloc[:100]])
    assert sc.run(tr, te, "y")[0][0] == "FAIL"


def test_group_overlap_fails():
    tr, te = frame(100, 0), frame(100, 1)
    tr["id"], te["id"] = range(100), range(50, 150)
    assert any("id" in m for _, m in sc.run(tr, te, "y", groups=["id"]))


def test_time_leak_fails_and_clean_cut_passes():
    tr, te = frame(10, 0), frame(10, 1)
    tr["t"] = pd.date_range("2024-01-01", periods=10)
    te["t"] = pd.date_range("2024-01-05", periods=10)
    assert sc.run(tr, te, "y", time="t")
    te["t"] = pd.date_range("2024-02-01", periods=10)
    assert not sc.run(tr, te, "y", time="t")


def test_target_leak_warns():
    tr, te = frame(500, 0), frame(500, 1)
    tr["leak"], te["leak"] = tr["y"] * 3.0, te["y"] * 3.0
    assert ("WARN", "'leak' alone scores AUC 1.000: possible target leak") in sc.run(tr, te, "y")
