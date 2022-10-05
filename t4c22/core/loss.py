import torch
from torch import Tensor, nn


class T4c22CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, weight: Tensor | None = None, ignore_index: int = -100):
        super().__init__(weight=weight, ignore_index=ignore_index)

    def forward(self, input: dict[str, Tensor], target: Tensor) -> Tensor:
        ce_loss = super().forward(input["cc_scores"], target["target"])
        mse_loss = torch.sqrt(((input["t"] - target["t"]) ** 2).mean()) / 100
        # mse_loss = torch.abs(input["t"].mean() - target["t"]) / 100
        return ce_loss + mse_loss
