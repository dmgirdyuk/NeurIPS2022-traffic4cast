# TODO:
#  - CUDA Out-of-Memory: https://huggingface.co/docs/accelerate/usage_guides/memory
#  - add checkpointer callback

from os.path import join as pjoin
from pathlib import Path
from typing import Any, Callable, Union, Type

import torch.functional
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm


def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    loss_function: Callable[[Any, Any], torch.Tensor],
    lr_scheduler: optim.lr_scheduler._LRScheduler,
    accelerator: Accelerator,
    epoch_num: int,
    checkpoint_save_folder: Union[str, Path],
    checkpoint_save_prefix: str,
) -> None:
    for epoch in range(epoch_num):
        total_train_loss, total_val_loss = torch.zeros(1), torch.zeros(1)

        model.train()
        for batch_step, batch in enumerate(tqdm(train_dataloader)):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                inputs = batch["inputs"]
                outputs = model(inputs)
                loss = loss_function(outputs, batch["label"])
                total_train_loss += loss.sum().item()
                accelerator.backward(loss)
                optimizer.step()

        lr_scheduler.step()
        total_train_loss /= len(train_dataloader)
        accelerator.log({"training_loss_epoch": total_train_loss}, step=epoch)

        model.eval()
        for batch_step, batch in val_dataloader:
            inputs = batch["inputs"]
            with torch.no_grad():
                outputs = model(inputs)
                loss = loss_function(outputs, batch["label"])
                total_val_loss += loss.sum().item()

                # accelerate.gather for distributed evaluation
                # predictions = outputs.argmax(dim=-1)

        total_val_loss /= len(val_dataloader)
        accelerator.log({"validation_loss_epoch": total_val_loss}, step=epoch)

    save_checkpoint(
        model=model,
        accelerator=accelerator,
        epoch=epoch_num,
        save_folder=checkpoint_save_folder,
        tag_prefix=checkpoint_save_prefix,
    )
    accelerator.end_training()


def save_checkpoint(
    model: nn.Module,
    accelerator: Accelerator,
    epoch: int,
    save_folder: Union[str, Path],
    tag_prefix: str = "model",
) -> None:
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
