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

# TODO:
#  - CUDA Out-of-Memory: https://huggingface.co/docs/accelerate/usage_guides/memory
#  - add checkpointer callback
import os
from os.path import join as pjoin
from pathlib import Path
from typing import Any, Callable, Type, Union

import torch.functional
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from torch.optim.lr_scheduler import _LRScheduler  # noqa
from torch_geometric.loader.dataloader import DataLoader
from tqdm.auto import tqdm

from t4c22.core.utils import get_logger

_logger = get_logger(__file__)


def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    loss_function: Callable[[Any, Any], torch.Tensor],
    lr_scheduler: _LRScheduler,
    accelerator: Accelerator,
    epoch_num: int,
    checkpoint_save_folder: Union[str, Path],
    checkpoint_save_prefix: str,
) -> None:
    for epoch in tqdm(range(epoch_num)):
        _logger.info("Epoch %d/%d", epoch, epoch_num)
        total_train_loss, total_val_loss = torch.zeros(1), torch.zeros(1)

        model.train()
        for batch_step, batch in enumerate(tqdm(train_dataloader)):
            with accelerator.accumulate(model):
                preprocess_batch(batch)
                optimizer.zero_grad()
                outputs = model(batch)
                loss = loss_function(outputs, batch.y)
                total_train_loss += loss.sum().item()
                accelerator.backward(loss)
                optimizer.step()

        lr_scheduler.step()
        total_train_loss /= len(train_dataloader)
        accelerator.log({"training_loss_epoch": total_train_loss}, step=epoch)
        _logger.info("Training loss: %.5f", total_train_loss)

        model.eval()
        for batch_step, batch in enumerate(tqdm(val_dataloader)):
            with torch.no_grad():
                preprocess_batch(batch)
                outputs = model(batch)
                loss = loss_function(outputs, batch.y)
                total_val_loss += loss.sum().item()

                # accelerate.gather for distributed evaluation
                # predictions = outputs.argmax(dim=-1)

        total_val_loss /= len(val_dataloader)
        accelerator.log({"validation_loss_epoch": total_val_loss}, step=epoch)
        _logger.info("Validation loss: %.5f", total_val_loss)

    save_checkpoint(
        model=model,
        accelerator=accelerator,
        epoch=epoch_num,
        save_folder=checkpoint_save_folder,
        tag_prefix=checkpoint_save_prefix,
    )
    accelerator.end_training()


def preprocess_batch(data):
    # Both data and labels are sparse. Loss function is masked by -1's
    data.x = data.x.nan_to_num(-1)
    data.edge_attr = data.edge_attr.nan_to_num(-1)
    data.y = data.y.nan_to_num(-1)
    data.y = data.y.long()


def save_checkpoint(
    model: nn.Module,
    accelerator: Accelerator,
    epoch: int,
    save_folder: Union[str, Path],
    tag_prefix: str = "model",
) -> None:
    os.makedirs(save_folder, exist_ok=True)
    accelerator.save(
        obj={"epoch": epoch, "model_state_dict": model.state_dict()},
        f=pjoin(save_folder, f"{tag_prefix}_epoch_{epoch}.pt"),
    )


def load_checkpoint(
    model_class: Type[nn.Module], load_path: Union[str, Path], accelerator: Accelerator
) -> nn.Module:
    # TODO: revisit
    model = model_class()
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    return accelerator.prepare(model)
