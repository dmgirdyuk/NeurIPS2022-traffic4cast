# Copyright 2022 STIL@home.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import random
import time
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


def timeit(func):
    def timed(*args, **kwargs):
        ts = time.time()
        result = func(*args, **kwargs)
        te = time.time() - ts
        print(f"func '{func.__name__}' took {te} sec")
        return result

    return timed
