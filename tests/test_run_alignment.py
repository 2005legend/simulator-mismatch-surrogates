import numpy as np

from scripts import run_alignment as al


def test_select_term_picks_most_correlated():
    rng = np.random.default_rng(0)
    d = rng.normal(size=30)
    norms = np.stack([rng.normal(size=30), d + 0.01 * rng.normal(size=30), -d])
    chosen, _ = al.select_term(d, ["a", "b", "c"], norms)
    assert chosen == "b"


def test_count_reads_k_of_10_proportionally():
    assert al.count([True] * 18 + [False] * 2, 0.9)["met"]
    assert not al.count([True] * 17 + [False] * 3, 0.9)["met"]
    assert al.count([True] * 18 + [False] * 2, 0.9)["required"] == ">= 18 of 20"
