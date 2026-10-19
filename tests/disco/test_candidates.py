import numpy as np
import torch

from disco.search.candidates import action_2x8, allocate


def test_allocate_budget_exact():
    rng = np.random.default_rng(0)
    for n in range(1, 8):
        alloc = allocate(n, K=2, B=32, rng=rng)
        assert sum(alloc.values()) * 2 == 32
        assert len(alloc) == n and all(v >= 1 for v in alloc.values())


def test_allocate_more_objects_than_budget():
    alloc = allocate(20, K=2, B=32, rng=np.random.default_rng(0))
    assert len(alloc) == 16 and all(v == 1 for v in alloc.values())


def test_action_tokens():
    a = action_2x8(torch.arange(7.), torch.arange(7.) + 10)
    assert a.shape == (2, 8) and a[0, 7] == 1 and a[1, 7] == 0 and a[1, 0] == 10
