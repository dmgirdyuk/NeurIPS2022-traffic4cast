# Copyright 2022 STIL at home.
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

import os
from dataclasses import dataclass
from os.path import join as pjoin
from pathlib import Path
from shutil import copyfile
from typing import List, Optional, Union

import torch
import torch.nn as nn
from accelerate import Accelerator

from t4c22.core.utils import get_logger

_logger = get_logger(__name__)


@dataclass
class Checkpoint:
    metric_val: float
    epoch: int
    save_path: Path


class CheckpointSaver:
    def __init__(
        self,
        accelerator: Accelerator,
        model: nn.Module,
        metric_name: str,
        save_dir: str,
        max_history: int = 1,
        should_minimize: Optional[bool] = True,
    ):
        self._accelerator = accelerator
        self._model = model
        self.metric_name = metric_name
        self.save_dir = Path(save_dir)
        self.max_history = max_history
        self.should_minimize = should_minimize

        self._storage: List[Checkpoint] = []
        os.makedirs(self.save_dir, exist_ok=True)

    def save(self, metric_val: float, epoch: int) -> None:
        save_name_prefix = f"model_e{epoch:03d}_checkpoint"
        save_path = self._save_checkpoint(
            model=self._model, epoch=epoch, save_name_prefix=save_name_prefix
        )
        self._storage.append(
            Checkpoint(
                metric_val=metric_val,
                epoch=epoch,
                save_path=save_path,
            )
        )
        self._storage = sorted(
            self._storage, key=lambda x: x.metric_val, reverse=not self.should_minimize
        )
        if len(self._storage) > self.max_history:
            worst_item = self._storage.pop()
            os.remove(worst_item.save_path)

        copyfile(
            src=self._storage[0].save_path,
            dst=self.save_dir / "model_checkpoint_best.pt",
        )
        _logger.info(
            "Best epoch %s value is %.4f on %d epoch",
            self.metric_name,
            self._storage[0].metric_val,
            self._storage[0].epoch,
        )

    def _save_checkpoint(
        self, model: nn.Module, epoch: int, save_name_prefix: str
    ) -> Path:
        save_path = pjoin(self.save_dir, f"{save_name_prefix}.pt")
        self._accelerator.wait_for_everyone()
        unwrapped_model = self._accelerator.unwrap_model(model)
        self._accelerator.save(
            obj={"epoch": epoch, "model_state_dict": unwrapped_model.state_dict()},
            f=save_path,
        )
        return Path(save_path)


def load_checkpoint(
    model: nn.Module, load_path: Union[str, Path], accelerator: Accelerator
) -> nn.Module:
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    return accelerator.prepare(model)
