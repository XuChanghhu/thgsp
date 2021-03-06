from functools import partial
from typing import List

import numpy as np
import torch
from scipy.sparse import diags
from torch_sparse import SparseTensor

from thgsp.bga import beta2channel_mask, beta_dist2channel_name, is_bipartite_fix, laplace
from thgsp.bga import harary, osglm, amfs, admm_bga, admm_lbga_ray
from thgsp.graphs import Graph
from .approximation import cheby_coeff, cheby_op, polyval, cheby_op_basis
from .kernels import meyer_kernel, meyer_mirror_kernel, get_kernel_name, design_biorth_kernel


class QmfCore:
    def __init__(self, bptG: List[SparseTensor], beta, analyze_kernels=None, synthesis_kernels=None, in_channels=1,
                 order=24, lam_max=2., zeroDC=False):
        assert len(bptG) == beta.shape[-1]
        assert bptG[0].size(-1) == beta.shape[0]
        assert lam_max > 0
        assert order > 0

        self.N, self.M = beta.shape
        self.in_channels = self.Ci = in_channels

        self.bptG = bptG
        self.beta = beta

        self.order = order
        self.dtype = bptG[0].dtype()
        self.device = bptG[0].device()

        self.lam_max = lam_max
        self.zeroDC = zeroDC

        self.bptL, self.bptD05 = self.compute_laplace(bptG)
        self.channel_mask, self.beta_dist = beta2channel_mask(beta)
        self.channel_mask = self.channel_mask.to(self.device)
        self.out_channels, _ = self.beta_dist.shape
        self.Co = self.out_channels  # alias of out_channels
        self.channel_name = beta_dist2channel_name(self.beta_dist)

        self.kernel_a = self.parse_kernels(analyze_kernels)
        self.kernel_s = self.parse_kernels(synthesis_kernels)

        self.coefficient_a = cheby_coeff(self.kernel_a, lam_max=lam_max, K=order, dtype=self.dtype,
                                         device=self.device)

        if synthesis_kernels is None:  # GraphQmf Meyer
            self.coefficient_s = self.coefficient_a
        else:  # GraphBiorth
            self.coefficient_s = cheby_coeff(self.kernel_s, lam_max=lam_max, K=order, dtype=self.dtype,
                                             device=self.device)

    def compute_laplace(self, bptG):
        bptL = []
        N = bptG[0].size(-1)
        bptD05 = torch.zeros(self.M, self.N, device=self.device) if self.zeroDC else None
        loop_index = torch.arange(N, device=self.device).unsqueeze_(0)
        for i, adj in enumerate(bptG):
            deg = adj.sum(0)
            row, col, val = adj.clone().coo()
            deg05 = deg.pow(-0.5)
            if self.zeroDC:
                deg05dc = deg05.clone().detach()
                deg05dc[deg05dc == float('inf')] = 1
                bptD05[i] = deg05dc

            deg05[deg05 == float('inf')] = 0
            wgt = deg05[row] * val * deg05[col]
            wgt = torch.cat([-wgt.unsqueeze_(0), val.new_ones(1, N)], 1).squeeze_()

            row = torch.cat([row.unsqueeze_(0), loop_index], 1).squeeze_()
            col = torch.cat([col.unsqueeze_(0), loop_index], 1).squeeze_()
            lap = SparseTensor(row=row, col=col, value=wgt)
            bptL.append(lap)
        return bptL, bptD05

    def parse_kernels(self, raw_kernels):
        if raw_kernels is None:
            kernel = self.kernel_array_from_beta_dist(meyer_kernel, meyer_mirror_kernel)

        elif isinstance(raw_kernels, tuple):  # only pass two kernel functions, the then remainder will be derived
            kernel = self.kernel_array_from_beta_dist(*raw_kernels)

        elif isinstance(raw_kernels, np.ndarray):  # a complete np.ndarray of kernels(functions)
            assert raw_kernels.shape == (self.M, self.Co, self.Ci)
            kernel = raw_kernels

        else:
            raise TypeError("{} is not a valid type of kernel".format(type(raw_kernels)))
        return kernel

    def kernel_array_from_beta_dist(self, kernel1, kernel2):
        f1c = np.where(self.beta_dist, kernel1, kernel2)
        f1c = np.transpose(f1c)
        return np.stack([f1c] * self.Ci, axis=-1)

    def empty_channels(self):
        empty_channels = (self.channel_mask.sum(1) == 0).nonzero().view(-1)
        return empty_channels

    def not_empty_channels(self):
        channels = (self.channel_mask.sum(1) != 0).nonzero().view(-1)
        return channels

    def __repr__(self):
        info = "{}(in_channels={}, order={}, max_lambda={}, " \
               "    n_channel={}, n_channel(non-empty)={}, N={},\n" \
               "    analyze_kernels:\n{},\n synthesize_kernels:\n{} \n)". \
            format(self.__class__.__name__, self.in_channels, self.order,
                   self.lam_max, self.out_channels, len(self.not_empty_channels()), self.N,
                   get_kernel_name(self.kernel_a[0], True),
                   get_kernel_name(self.kernel_s[0], True))
        return info

    def _check_signal(self, x):
        if x.dim() == 1:  # N -> 1 x  N x 1
            x = x.reshape(1, -1, 1)
        elif x.dim() == 2:  # N x Ci -> 1 x N x Ci
            x = x.unsqueeze(0)
        elif x.dim() == 3:  # keep Co x N x Ci or 1 x N x Ci # Check for synthesis
            x = x
        else:
            raise RuntimeError("rank-1,2,3 tensor expected, but got rank-{}".format(x.dim()))

        if x.shape[-2] != self.N:
            raise RuntimeError(f"The penultimate dimension of signal:{x.shape[-2]}!= the number of nodes: {self.N}")
        if x.shape[-1] != self.Ci:
            raise RuntimeError("{} input channels expected, but got {}".format(self.Ci, x.shape[-1]))
        return x.to(self.dtype)

    def _analyze(self, x):
        y = x  # Co x N x Ci
        for g in range(self.M):
            if self.zeroDC:
                y = self.bptD05[g].pow(-1) * y
            y = cheby_op(y, self.bptL[g], self.coefficient_a[g], lam_max=self.lam_max)
        mask = self.channel_mask.unsqueeze(-1)  # Co x N --> Co x N x 1 for broadcast 'masked_fill_'
        y.masked_fill_(~mask, 0)
        return y

    def analyze(self, x):
        x = self._check_signal(x)
        return self._analyze(x)

    def _synthesize(self, y):
        z = y
        for g in range(self.M - 1, -1, -1):  # M-1, M-2, ..., 0 totally M bipartite graphs
            z = cheby_op(z, self.bptL[g], self.coefficient_s[g], lam_max=self.lam_max)
            if self.zeroDC:
                z = self.bptD05[g] * z
        return z  # Co x N x Ci

    def synthesize(self, y):
        y = self._check_signal(y)
        return self._synthesize(y)


class QmfOperator:
    def __init__(self, bptG, beta, order=24, lam_max=2., device=None):
        N, M = beta.shape
        assert len(bptG) == M

        self.N, self.M = N, M
        self.order = order
        self.device = device
        self.lam_max = lam_max

        krn = np.array([[[meyer_kernel],
                         [meyer_mirror_kernel]]])  # 1(n_graph) x 2(Cout) x 1(Cin) kernel
        coeff = cheby_coeff(krn, K=order, lam_max=lam_max).squeeze_()  # (1,2,1,K) --> (2,K)

        operator = self.compute_basis(bptG, coeff, beta, lam_max)
        self.operator = SparseTensor.from_scipy(operator).to(device)

        self.dtype = self.operator.dtype()
        self.device = self.operator.device()

    def transform(self, x):
        return self.operator @ x

    def inverse_transform(self, y):
        return self.operator.t() @ y

    @staticmethod
    def compute_basis(bptG, coeff, beta, lam_max):
        M = len(bptG)
        dt = bptG[0].dtype
        beta = np.asarray(beta, dtype=dt)
        beta = 1 - beta * 2

        bptL = [laplace(B, lap_type='sym', add_loop=True) for B in bptG]

        H0, H1 = cheby_op_basis(bptL[0].tocsr(), coeff, lam_max)
        t1 = H0 + H1
        t2 = H1 - H0
        Ta = t1 + diags(beta[:, 0]) @ t2

        for i in range(1, M):
            H0, H1 = cheby_op_basis(bptL[i].tocsr(), coeff, lam_max)
            t1 = H0 + H1
            t2 = H1 - H0
            Ta_sub = t1 + diags(beta[:, i]) @ t2
            Ta = Ta_sub @ Ta
        Ta *= 0.5 ** M
        return Ta

    def __call__(self, x):
        return self.transform(x)


class BiorthOperator:
    def __init__(self, bptG, beta, k=4, lam_max=2., device=None):
        h0_c, g0_c, orthogonality = design_biorth_kernel(k)
        h0 = partial(polyval, torch.from_numpy(h0_c))
        h0.__name__ = 'h0'
        g0 = partial(polyval, torch.from_numpy(g0_c))
        g0.__name__ = 'g0'

        def h1(x):
            return g0(2 - x)

        def g1(x):
            return h0(2 - x)

        self.orthogonality = orthogonality
        self.analysis_krn = np.array([[[h0],
                                       [h1]]])

        self.synthesis_krn = np.array([[[g0],
                                        [g1]]])

        ana_coeff = cheby_coeff(self.analysis_krn, K=2 * k, lam_max=lam_max).squeeze_()  # (1,2,1,K) --> (2,K)

        syn_coeff = cheby_coeff(self.synthesis_krn, K=2 * k, lam_max=lam_max).squeeze_()  # (1,2,1,K) --> (2,K)
        operator = QmfOperator.compute_basis(bptG, ana_coeff, beta, lam_max)
        self.operator = SparseTensor.from_scipy(operator).to(device)
        inv_operator = QmfOperator.compute_basis(bptG, syn_coeff, beta, lam_max)
        self.inv_operator = SparseTensor.from_scipy(inv_operator).to(device)

    def transform(self, x):
        return self.operator @ x

    def inverse_transform(self, y):
        return self.inv_operator.t() @ y


class ColorQmf(QmfCore):
    def __init__(self, G: Graph, kernel=None, in_channels=1, order=24, strategy="harary", vtx_color=None, lam_max=2.,
                 zeroDC=False, **kwargs):
        self.adj = G
        self.strategy = strategy

        if strategy is "harary":
            bptG, beta, beta_dist, vtx_color, mapper = harary(self.adj, vtx_color=vtx_color, **kwargs)
        elif strategy is "osglm":
            bptG, beta, append_nodes, vtx_color = osglm(self.adj, vtx_color=vtx_color, **kwargs)
            self.append_nodes = append_nodes
        else:
            raise RuntimeError("{} is not a valid color-based decomposition algorithm.".format(str(strategy)))
        self.vtx_color = vtx_color

        bptG = [SparseTensor.from_scipy(B).to(G.device()) for B in bptG]

        super(ColorQmf, self).__init__(bptG, beta, analyze_kernels=kernel, in_channels=in_channels,
                                       order=order, lam_max=lam_max, zeroDC=zeroDC)
        self.N = self.adj.size(-1)  # osglm compatible

    def analyze(self, x):
        x = self._check_signal(x)
        if self.strategy is "osglm":
            x_append = x[:, self.append_nodes, :]
            x = torch.cat([x, x_append], 1)
        return self._analyze(x)

    def synthesize(self, y):
        z = self._synthesize(y)
        if self.strategy is "osglm":
            z = z[:, :self.N, :]
        return z


class NumQmf(QmfCore):
    def __init__(self, G, kernel=None, in_channels=1, order=24, strategy: str = "admm", M=1, lam_max=2., zeroDC=False,
                 **kwargs):
        self.adj = G
        N = self.adj.size(-1)

        self.strategy = strategy
        self.M = M

        device = self.adj.device()
        dtype = self.adj.dtype()
        if strategy is "admm":
            if N < 80:
                bptG_dense = admm_bga(self.adj.to_dense().to(torch.double), M=M, **kwargs)
                beta = bptG_dense.new_zeros(N, M, dtype=bool)
                bptG = []
                for i, B in enumerate(bptG_dense):
                    _, vtx_color, _ = is_bipartite_fix(B, fix_flag=True)
                    beta[:, i] = torch.as_tensor(vtx_color)
                    bptG.append(SparseTensor.from_dense(B).to(dtype).to(device))

            else:
                bptG, beta, self.partptr, self.perm = admm_lbga_ray(self.adj, M, **kwargs)
                bptG = [SparseTensor.from_scipy(B).to(dtype).to(device) for B in bptG]

        elif strategy is "amfs":
            bptG, beta = amfs(self.adj, level=self.M, **kwargs)
            bptG = [SparseTensor.from_scipy(B).to(dtype).to(device) for B in bptG]

        else:
            raise RuntimeError(
                "{} is not a valid numerical decomposition algorithm supported at present.".format(str(strategy)))

        super(NumQmf, self).__init__(bptG, beta, analyze_kernels=kernel, in_channels=in_channels, order=order,
                                     lam_max=lam_max, zeroDC=zeroDC)


class BiorthCore(QmfCore):
    def __init__(self, bptG, beta, k=8, in_channels=1, order=16, lam_max=2., zeroDC=False):
        h0_c, g0_c, orthogonality = design_biorth_kernel(k)
        h0 = partial(polyval, torch.from_numpy(h0_c))
        h0.__name__ = 'h0'
        g0 = partial(polyval, torch.from_numpy(g0_c))
        g0.__name__ = 'g0'

        def h1(x):
            return g0(2 - x)

        def g1(x):
            return h0(2 - x)

        self.orthogonality = orthogonality
        super(BiorthCore, self).__init__(bptG, beta, analyze_kernels=(h0, h1), synthesis_kernels=(g0, g1),
                                         in_channels=in_channels, order=order, lam_max=lam_max, zeroDC=zeroDC)

    def __repr__(self):
        info = super().__repr__()
        info = info[:-2] + ", orthogonality={}".format(self.orthogonality)
        return info + '\n)'


class ColorBiorth(BiorthCore):
    def __init__(self, G: Graph, k=8, in_channels=1, order=16, strategy="harary", vtx_color=None, lam_max=2.,
                 zeroDC=False, **kwargs):
        self.adj = G
        self.lam_max = lam_max
        self.strategy = strategy

        if strategy is "harary":
            bptG, beta, beta_dist, vtx_color, mapper = harary(self.adj, vtx_color=vtx_color, **kwargs)
        elif strategy is "osglm":
            bptG, beta, append_nodes, vtx_color = osglm(self.adj, vtx_color=vtx_color, **kwargs)
            self.append_nodes = append_nodes
        else:
            raise RuntimeError("{} is not a valid color-based decomposition algorithm.".format(str(strategy)))
        self.vtx_color = vtx_color

        bptG = [SparseTensor.from_scipy(B).to(G.device()) for B in bptG]

        super(ColorBiorth, self).__init__(bptG, beta, k, in_channels, order, lam_max, zeroDC)
        self.N = self.adj.size(-1)  # osglm compatible

    def analyze(self, x):
        x = self._check_signal(x)
        if self.strategy is "osglm":
            x_append = x[:, self.append_nodes, :]
            x = torch.cat([x, x_append], 1)
        return self._analyze(x)

    def synthesize(self, y):
        z = self._synthesize(y)
        if self.strategy is "osglm":
            z = z[:, :self.N, :]
        return z


class NumBiorth(BiorthCore):
    def __init__(self, G, k=8, in_channels=1, order=16, strategy="admm", M=1, lam_max=2., zeroDC=False, **kwargs):
        self.adj = G
        N = self.adj.size(-1)
        self.lam_max = lam_max

        self.strategy = strategy
        self.M = M

        device = self.adj.device()
        dtype = self.adj.dtype()
        if strategy is "admm":
            if N < 80:
                bptG_dense = admm_bga(self.adj.to_dense().to(torch.double), M=M, **kwargs)
                beta = bptG_dense.new_zeros(N, M, dtype=bool)
                bptG = []
                for i, B in enumerate(bptG_dense):
                    _, vtx_color, _ = is_bipartite_fix(B, fix_flag=True)
                    beta[:, i] = torch.as_tensor(vtx_color)
                    bptG.append(SparseTensor.from_dense(B).to(dtype).to(device))

            else:
                bptG, beta, self.partptr, self.perm = admm_lbga_ray(self.adj, M, **kwargs)
                bptG = [SparseTensor.from_scipy(B).to(dtype).to(device) for B in bptG]

        elif strategy is "amfs":
            bptG, beta = amfs(self.adj, level=self.M, **kwargs)
            bptG = [SparseTensor.from_scipy(B).to(dtype).to(device) for B in bptG]

        else:
            raise RuntimeError(
                "{} is not a valid numerical decomposition algorithm supported at present.".format(str(strategy)))

        super(NumBiorth, self).__init__(bptG, beta, k, in_channels, order, lam_max, zeroDC)
