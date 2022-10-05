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

from typing import Any, Callable

import torch
from accelerate import Accelerator
from torch import nn, optim
from torch.optim.lr_scheduler import _LRScheduler  # noqa
from torch_geometric.data import Data
from torch_geometric.loader.dataloader import DataLoader
from tqdm.auto import tqdm

from t4c22.core.checkpointer import CheckpointSaver
from t4c22.core.utils import get_logger

_logger = get_logger(__name__)


def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    loss_function: Callable[[Any, Any], torch.Tensor],
    lr_scheduler: _LRScheduler,
    accelerator: Accelerator,
    epoch_num: int,
    checkpoint_saver: CheckpointSaver,
) -> None:
    for epoch in tqdm(range(epoch_num)):
        _logger.info("Epoch %d/%d", epoch, epoch_num)
        total_train_loss, total_val_loss = torch.zeros(1), torch.zeros(1)

        model.train()
        for batch in tqdm(train_dataloader):
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
        for batch in tqdm(val_dataloader):
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

        checkpoint_saver.save(metric_val=total_val_loss.detach().numpy(), epoch=epoch)

    accelerator.end_training()


def preprocess_batch(data: Data):
    # Both data and labels are sparse. Loss function is masked by -1's
    data["x"] = torch.log10((data.x + 1).nan_to_num(1e-1))
    data["edge_attr"] = data.edge_attr.nan_to_num(0)
    data["y"]["target"] = data.y["target"].nan_to_num(-1).long()
