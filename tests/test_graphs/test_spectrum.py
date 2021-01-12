from thgsp.graphs.spectrum import dynamic_spectrum_estimate_dense, spl, tripb
import numpy as np
import scipy.sparse as ssp
import scipy.sparse.linalg as lin
from ..utils4t import float_np_dts
import pytest

np.set_printoptions(linewidth=2000000, precision=5)


@pytest.mark.parametrize("dtype", float_np_dts[1:2])
class TestSpectrum:
    def test_dynamic_spectrum_estimate_dense(self, dtype):
        scale = 5
        largest = True
        N = 15
        A = np.random.rand(N, N).astype(dtype) * scale
        np.fill_diagonal(A, 0)
        A = A + A.T
        row = np.array([0, 3, 8, 2, 6, 9])
        col = np.array([2, 6, 9, 0, 3, 8])
        vmin = A[row, col] - scale * 2
        vmax = A[row, col] + scale * 2

        lam_es, B_es = dynamic_spectrum_estimate_dense(A, index=np.stack([row, col]),
                                                       fluctuation=np.stack([vmin, vmax]), rho=2, verbose=True)

        print("lambda: ", lam_es)
        # check if largest
        At = np.copy(A)
        At[row, col] = vmin
        val, _ = spl.eigsh(At, k=1, which='LA' if largest else 'SA')
        assert val[0] < lam_es

        At[row, col] = vmax
        val, _ = spl.eigsh(At, k=1, which='LA' if largest else 'SA')
        assert val[0] < lam_es

        val, _ = spl.eigsh(B_es, k=1, which='LA' if largest else 'SA')
        assert val[0] < lam_es
        assert abs(val[0] - lam_es) < 1e-2

        val, _ = spl.eigsh(At, k=1, which='LA' if largest else 'SA')
        assert val[0] < lam_es

    def test_tripb(self, dtype):
        N = 10
        T = 4
        k = 5
        A0 = ssp.rand(N, N, 0.6, dtype=dtype)
        A0.setdiag(0)
        A0 = A0 + A0.T
        perturbation = []
        for _ in range(T):
            deltaA = ssp.rand(N, N, 0.1)
            deltaA.setdiag(0)
            perturbation.append(deltaA + deltaA.T)
        val0, vec0 = lin.eigsh(A0, k=k, which='LM')
        lam, U = tripb(perturbation, val0, vec0)
        AT = A0 + sum(perturbation)
        valT, vecT = lin.eigsh(AT, k=k, which='LM')
        print("\n-----------eigenvalue----------")
        print("estimated: ", lam[-1], "\n   gt    :", valT)
        print("-----------eigenvector----------")
        print("estimated:", U[-1], "\n   gt    :", vecT)
