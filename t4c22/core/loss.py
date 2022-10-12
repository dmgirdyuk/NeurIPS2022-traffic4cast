import torch
from torch import Tensor, nn


class CustomCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(
        self,
        weight: Tensor | None = None,
        ignore_index: int = -100,
        add_t: bool = True,
        add_day: bool = True,
        add_working_day: bool = True,
    ):
        super().__init__(weight=weight, ignore_index=ignore_index)

        self.add_t = add_t
        self.add_day = add_day
        self.add_working_day = add_working_day

    def forward(self, input: dict[str, Tensor], target: Tensor) -> Tensor:
        ce_loss = super().forward(input["cc_scores"], target["target"])
        mse_t_loss = torch.zeros_like(ce_loss)
        bce_d_loss = torch.zeros_like(ce_loss)
        bce_wd_loss = torch.zeros_like(ce_loss)

        if self.add_t:
            mse_t_loss = torch.sqrt(((input["t"] - target["t"]) ** 2).mean()) / 100

        if self.add_day:
            key = "day"
            d_target = target[key] * torch.ones_like(input[key])
            bce_d_loss = (
                torch.binary_cross_entropy_with_logits(input[key], d_target).mean() / 10
            )

        if self.add_working_day:
            key = "working_day"
            wd_target = target[key] * torch.ones_like(input[key])
            bce_wd_loss = (
                torch.binary_cross_entropy_with_logits(input[key], wd_target).mean()
                / 10
            )

        return ce_loss + mse_t_loss + bce_d_loss + bce_wd_loss
