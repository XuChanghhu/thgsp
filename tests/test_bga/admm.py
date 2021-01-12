import pytest
import ray
from thgsp.graphs.generators import rand_udg
from ..utils4t import float_dtypes, devices, partition_strategy
from thgsp.bga.admm import admm_bga, is_bipartite_fix, admm_lbga_ray


@pytest.mark.parametrize('dtype', float_dtypes[::-1])
@pytest.mark.parametrize('density', [0.2, 0.6])
def test_admm_bga(density, dtype):
    N = 20
    M = 2
    G = rand_udg(N, density, dtype)
    bptGs = admm_bga(G.adj.to_dense(), M=M)
    for i in range(M):
        assert is_bipartite_fix(bptGs[i])[0]


@pytest.mark.parametrize('device', devices)
@pytest.mark.parametrize('dtype', float_dtypes)
@pytest.mark.parametrize('density', [0.1])
@pytest.mark.parametrize('style', [1, 2])
@pytest.mark.parametrize('M', [1, 2])
@pytest.mark.parametrize('part', partition_strategy)
class TestAdmmLbga:
    def test_admm_lbga_ray(self, density, style, M, dtype, device, part):
        N = 32 * 7
        G = rand_udg(N, density, dtype=dtype, device=device)
        ray.init(log_to_driver=False)
        bptGs, beta, partptr, perm = admm_lbga_ray(G.adj, M, block_size=64, style=style, part=part)
        print("\n-----num_node: {}-|-density: {}-|-strategy: {}-|-M: {}-|-dtype: {}-|-device: {}-|part: {}".
              format(N, density, style, M, str(dtype), str(device), part))
        print("total weights: {}".format(G.adj.sum().item()))
        for i, bptG in enumerate(bptGs):
            assert is_bipartite_fix(bptG)[0]
            print("{}-th subgraph, weights: {}".format(i, bptG.sum()))
        print("------------------------------------------------------------------------".format(N, density))
        ray.shutdown()
