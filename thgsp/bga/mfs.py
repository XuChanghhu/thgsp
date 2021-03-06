from typing import List, Tuple

from scipy.sparse import lil_matrix, eye
from scipy.sparse.csgraph import structural_rank, breadth_first_order
from scipy.sparse.linalg import inv
from sksparse.cholmod import cholesky
from torch_sparse import SparseTensor

from thgsp.alg import dsatur
from .utils import laplace, bipartite_mask, np


def amfs(A: SparseTensor, Sigma=None, level=None, delta=0.1, thresh_kld=1e-6, priority=True, verbose=False) \
        -> Tuple[List[lil_matrix], np.ndarray]:
    N = A.size(-1)
    A = A.to_scipy(layout='coo')  # compute_sigma consists of laplace matrix which prefers "coo"
    if Sigma is None:
        Sigma = compute_sigma(A, delta)
    else:
        assert Sigma.shape == (N, N)
    if level is None:
        chromatic = dsatur(A).n_color
        level = np.ceil(np.log2(chromatic))

    A = A.tolil()
    beta = np.zeros((N, level), dtype=bool)
    bptG = [lil_matrix((N, N), dtype=A.dtype) for _ in range(level)]
    for i in range(level):
        if verbose:
            print("\n|----------------------decomposition in level: {:4d} ------------------------|".format(i))
        s1, s2 = amfs1level(A, Sigma, delta, thresh_kld, priority, verbose)
        bt = beta[:, i]
        bt[s1] = 1  # set s1 True
        mask = bipartite_mask(bt)
        bptG[i][mask] = A[mask]
        A[mask] = 0
    return bptG, beta


def amfs1level(W: lil_matrix, Sigma: lil_matrix = None, delta=0.1, thresh_kld=1e-6, priority=True, verbose=True):
    if Sigma is None:
        Sigma = compute_sigma(W, delta)
    N = W.shape[-1]
    not_arrived = np.arange(N)
    nodes = breadth_first_order(W, i_start=0, return_predecessors=False)
    not_arrived = np.setdiff1d(not_arrived, nodes)
    s1 = [0]
    s2 = []
    nodes = nodes[1:]

    while len(not_arrived) > 0:
        new_root = not_arrived[0]
        other_nodes = breadth_first_order(W, i_start=new_root, return_predecessors=False)
        not_arrived = np.setdiff1d(not_arrived, other_nodes)
        s1.append(new_root)
        nodes = np.append(nodes, other_nodes[1:])

    balance_flag = True
    for i,v in enumerate(nodes):
        if verbose:
            print("handling {:5d}-th node: {:5d}, ".format(i,v), end='')
        N1 = len(s1)
        s = [*s1, v, *s2]
        W_local = W[np.ix_(s, s)]

        Wb1 = W_local.copy()
        Wb2 = W_local.copy()
        Wb2[:N1, :N1] = 0
        Wb2[N1:, N1:] = 0
        Wb1[:N1 + 1, :N1 + 1] = 0
        Wb1[N1 + 1:, N1 + 1:] = 0
        if priority:  # KLD holds priority
            S_local = Sigma[np.ix_(s, s)]
            DK1 = dkl(Wb1, S_local, delta)
            DK2 = dkl(Wb2, S_local, delta)
            diff = DK1 - DK2
            if verbose:
                print("DK1-DK2: {:5f}".format(diff))
            if abs(diff) > thresh_kld:
                if diff > 0:
                    s2.append(v)
                else:
                    s1.append(v)
            else:
                rank1 = structural_rank(Wb1.tocsr())
                rank2 = structural_rank(Wb2.tocsr())
                if rank1 > rank2:
                    s1.append(v)
                elif rank1 < rank2:
                    s2.append(v)
                else:
                    if balance_flag:
                        s1.append(v)
                    else:
                        s2.append(v)
                    balance_flag = not balance_flag
        else:
            rank1 = structural_rank(Wb1)
            rank2 = structural_rank(Wb2)
            if rank1 > rank2:
                s1.append(v)
            elif rank1 < rank2:
                s2.append(v)
            else:
                S_local = Sigma[np.ix_(s, s)]
                DK1 = dkl(Wb1, S_local, delta)
                DK2 = dkl(Wb2, S_local, delta)
                if DK1 < DK2:
                    s1.append(v)
                elif DK1 > DK2:
                    s2.append(v)
                else:
                    if balance_flag:
                        s1.append(v)
                    else:
                        s2.append(v)
                    balance_flag = not balance_flag
    return s1, s2


def dkl(Wb: lil_matrix, Sigma, delta: float):
    N = Wb.shape[-1]
    Lb = laplace(Wb, lap_type="comb").tocsc()  # coo -> csc
    temp = Lb + delta * eye(N, dtype=Lb.dtype, format='csc')
    try:
        dk = (Lb @ Sigma).diagonal().sum() - cholesky(temp).logdet()  # cholesky prefers `csc`
    except Exception as err:
        raise err
    return dk


def compute_sigma(A, delta, precision_mat=False) -> lil_matrix:
    Sigma_inv = laplace(A, lap_type="comb").tocsc() + delta * eye(A.shape[-1], dtype=A.dtype, format='csc')
    if precision_mat:
        return Sigma_inv
    Sigma = inv(Sigma_inv)  # csc more efficient
    Sigma = Sigma + Sigma.T
    Sigma.data*=0.5
    return Sigma.tolil()
