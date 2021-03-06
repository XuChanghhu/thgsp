import pytest

from thgsp.graphs.is_bipartite import is_bipartite
from thgsp.graphs.generators import rand_bipartite, rand_udg
from ..utils4t import devices


@pytest.mark.parametrize("device", devices)
def test_is_bipartite1(device):
    ts_spm = rand_udg(50, device=device).adj
    assert not is_bipartite(ts_spm)[0]


@pytest.mark.parametrize("device", devices)
def test_is_bipartite2(device):
    ts_spm = rand_bipartite(4, 6, device=device).adj
    assert is_bipartite(ts_spm)[0]
