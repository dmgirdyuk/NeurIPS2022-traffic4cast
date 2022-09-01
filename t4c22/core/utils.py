import logging
import random
from pathlib import Path

import numpy as np
import torch

from t4c22.misc.t4c22_logging import t4c_apply_basic_logging_config


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


def seed_everything(seed: int = 314159) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_logger(name: str) -> logging.Logger:
    t4c_apply_basic_logging_config("DEBUG")
    return logging.getLogger(name)
