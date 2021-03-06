import importlib
import os.path as osp

import torch

__version__ = '0.1.0'

cpp_tools = ['_version', '_dsatur']
for tool in cpp_tools:
    torch.ops.load_library(importlib.machinery.PathFinder().find_spec(
        tool, [osp.dirname(__file__)]).origin)

if torch.version.cuda is not None:  # pragma: no cover
    cuda_version = torch.ops.torch_gsp.cuda_version()

    if cuda_version == -1:
        major = minor = 0
    elif cuda_version < 10000:
        major, minor = int(str(cuda_version)[0]), int(str(cuda_version)[2])
    else:
        major, minor = int(str(cuda_version)[0:2]), int(str(cuda_version)[3])
    t_major, t_minor = [int(x) for x in torch.version.cuda.split('.')]

    if t_major != major or t_minor != minor:
        raise RuntimeError(
            f'Detected that PyTorch and torch_sparse were compiled with '
            f'different CUDA versions. PyTorch has CUDA version '
            f'{t_major}.{t_minor} and torch_sparse has CUDA version '
            f'{major}.{minor}. Please reinstall the torch_sparse that '
            f'matches your PyTorch install.')

from .convert import to_torch_sparse  # noqa
from .io import loadmat  # noqa

import thgsp.graphs  # noqa
import thgsp.alg  # noqa
import thgsp.filters  # noqa
import thgsp.bga  # noqa
import thgsp.utils  # noqa
import thgsp.datasets  # noqa

__all__ = ['to_torch_sparse',
           'loadmat']
