import torch
from torch import Tensor, nn


class CustomCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(
        self,
        weight: Tensor | None = None,
        ignore_index: int = -100,
        add_custom_term: bool = True,
    ):
        super().__init__(weight=weight, ignore_index=ignore_index)

        self.add_custom_term = add_custom_term

    def forward(self, input: dict[str, Tensor], target: Tensor) -> Tensor:
        ce_loss = super().forward(input["cc_scores"], target["target"])
        if self.add_custom_term:
            mse_loss = torch.sqrt(((input["t"] - target["t"]) ** 2).mean()) / 100
            return ce_loss + mse_loss
        return ce_loss
