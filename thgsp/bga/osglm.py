import numpy as np
from scipy.sparse import lil_matrix, eye

from .utils import bipartite_mask


def osglm(A, lc=None, vtx_color=None):
    if vtx_color is None:
        from thgsp.alg.coloring import dsatur_coloring
        vtx_color = dsatur_coloring(A)
    vtx_color = np.asarray(vtx_color)
    n_color = max(vtx_color) + 1

    if lc is None:
        lc = n_color // 2
    assert 1 <= lc < n_color

    A = A.to_scipy(layout='csr').tolil()
    Gb = lil_matrix(A.shape, dtype=A.dtype)  # the foundation bipartite graph Gb
    N = A.shape[-1]

    bt = np.in1d(vtx_color, range(lc))
    idx_s1 = np.nonzero(bt)[0]  # L
    idx_s2 = np.nonzero(~bt)[0]  # H

    mask = bipartite_mask(bt)  # the desired edges
    Gb[mask] = A[mask]
    A[mask] = 0
    eye_mask = eye(N, N, dtype=bool)
    A[eye_mask] = 1  # add vertical edges

    degree = A.sum(0).getA1()  # 2D np.matrix -> 1D np.array
    append_nodes = (degree != 0).nonzero()[0]

    Nos = len(append_nodes) + N  # oversampled size
    bptG = [lil_matrix((Nos, Nos), dtype=A.dtype)]  # the expanded graph
    bptG[0][:N, N:] = A[:, append_nodes]
    bptG[0][:N, :N] = Gb
    bptG[0][N:, :N] = A[append_nodes, :]

    beta = np.zeros((Nos, 1), dtype=np.bool)
    beta[idx_s1, 0] = 1
    # appended nodes corresponding to idx_s2 are assigned to the L channel of oversampled graph with idx_s1
    _, node_ordinal_append, _ = np.intersect1d(append_nodes, idx_s2, return_indices=True)
    beta[N + node_ordinal_append, 0] = 1
    return bptG, beta, append_nodes, vtx_color
