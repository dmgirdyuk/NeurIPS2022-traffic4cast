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

from typing import Tuple

import torch
from torch.utils.data import random_split
from torch_geometric.data import Dataset
from torch_geometric.loader.dataloader import DataLoader

from t4c22.metric.masked_crossentropy import get_weights_from_class_fractions
from t4c22.t4c22_config import class_fractions


def get_train_val_dataloaders(
    dataset: Dataset, batch_size: int = 1, shuffle: bool = True, num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    spl = int(((0.8 * len(dataset)) // 2) * 2)
    train_dataset, val_dataset = random_split(dataset, [spl, len(dataset) - spl])

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    return train_dataloader, val_dataloader


def get_city_class_weights(city: str = "london") -> torch.Tensor:
    city_class_fractions = class_fractions[city]
    city_class_weights = torch.Tensor(
        get_weights_from_class_fractions(
            [city_class_fractions[c] for c in ["green", "yellow", "red"]]
        )
    ).float()
    return city_class_weights
