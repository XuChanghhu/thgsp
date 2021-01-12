import torch
import numpy as np

dtypes = [torch.float, torch.double, torch.int, torch.long]

grad_dtypes = [torch.float, torch.double]
float_dtypes = grad_dtypes
float_np_dts = [np.float32, np.float64]
int_dtypes = [torch.int, torch.long]

lap_types = ['comb', 'sym', 'rw', None]

color_strategies = ["harary", "osglm"]
num_strategies = ["admm", "amfs"]

devices = [torch.device('cpu')]
if torch.cuda.is_available():
    devices += [torch.device(f'cuda:{torch.cuda.current_device()}')]

partition_strategy = ["graclus", 'metis']

def to_tensor(x, dtype, device=None):
    return None if x is None else torch.as_tensor(x, dtype=dtype, device=device)
