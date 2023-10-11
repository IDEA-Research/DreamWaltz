import random
import os
from pathlib import Path

import numpy as np
import torch


def tensor2numpy(tensor:torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu().numpy()
    tensor = (tensor * 255).clip(0, 255).astype(np.uint8)
    return tensor


def make_path(path: Path) -> Path:
    path.mkdir(exist_ok=True, parents=True)
    return path


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


def normalized_cross_correlation(x, y):
    std_x, mean_x = torch.std_mean(x)
    std_y, mean_y = torch.std_mean(y)
    return torch.mean((x - mean_x) * (y - mean_y)) / (std_x * std_y)
