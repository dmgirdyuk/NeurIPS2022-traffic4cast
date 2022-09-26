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

import random
from functools import partial
from pathlib import Path
from typing import Optional, Tuple, cast

import torch
from torch.utils.data import Subset
from torch_geometric.data import Data, Dataset
from torch_geometric.loader.dataloader import DataLoader

from t4c22.core.utils import get_logger
from t4c22.dataloading.road_graph_mapping import TorchRoadGraphMapping
from t4c22.metric.masked_crossentropy import get_weights_from_class_fractions
from t4c22.t4c22_config import (
    DAY_T_FILTER,
    cc_dates,
    class_fractions,
    day_t_filter_to_df_filter,
    day_t_filter_weekdays_daytime_only,
)

_logger = get_logger(__name__)


def get_train_val_dataloaders(
    dataset: Dataset,
    ratio: float = 0.97,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    spl = int(((ratio * len(dataset)) // 2) * 2)

    train_dataset = cast(Dataset, Subset(dataset, range(spl)))
    val_dataset = cast(Dataset, Subset(dataset, range(spl, len(dataset))))

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


def get_avg_class_weights() -> torch.Tensor:
    city_class_weights = torch.Tensor(
        [
            (0.5367906303432076 + 0.4976221039083026 + 0.7018930324884697) / 3,
            (0.35138063340805714 + 0.3829591430424158 + 0.2223245729555099) / 3,
            (0.11182873624873524 + 0.1194187530492816 + 0.0757823945560204) / 3,
        ]
    )
    city_class_weights = city_class_weights.sum() / city_class_weights
    return city_class_weights.float()


CITIES = ["madrid", "melbourne", "london"]


class T4c22TrainDataset(Dataset):  # pylint: disable=abstract-method  # noqa
    def __init__(
        self,
        root: Path | str,
        edge_attributes: Optional[list[str]] = None,
        cachedir: Optional[Path | str] = None,
        limit: Optional[int] = None,
        day_t_filter: DAY_T_FILTER = day_t_filter_weekdays_daytime_only,
        counters_only: bool = False,
    ):
        super().__init__()
        self.root: Path = root
        self.cachedir = cachedir
        self.limit = limit
        self.day_t_filter = day_t_filter
        self.counters_only = counters_only

        self.city_day_t: list[tuple[str, str, int]] = []
        for city in CITIES:
            self.city_day_t.extend(
                [
                    (city, day, t)
                    for day in cc_dates(self.root, city=city, split="train")
                    for t in range(4, 96)
                    if self.day_t_filter(day, t)
                ]
            )
        # random.shuffle(self.city_day_t)  # seed should be already fixed in the main func

        df_filter = partial(day_t_filter_to_df_filter, filter=day_t_filter)
        self.city_road_graph_mapping = {
            city: TorchRoadGraphMapping(
                city=city,
                edge_attributes=edge_attributes,
                root=root,
                df_filter=df_filter,
                skip_supersegments=True,  # next time
                counters_only=self.counters_only,
            )
            for city in CITIES
        }

        self.city = CITIES[0]
        self.torch_road_graph_mapping = self.city_road_graph_mapping[self.city]

    def len(self) -> int:
        dataset_len = len(self.city_day_t)
        if self.limit is not None:
            return min(self.limit, dataset_len)
        return dataset_len

    def get(self, idx: int) -> Data:
        self.city, day, t = self.city_day_t[idx]
        self.torch_road_graph_mapping = self.city_road_graph_mapping[self.city]

        cache_file = self.cachedir / f"cc_labels_{self.city}_{day}_{t}.pt"
        if self.cachedir is not None:
            if cache_file.exists():
                data = torch.load(cache_file)
                return data

        x = self.torch_road_graph_mapping.load_inputs_day_t(
            basedir=self.root, city=self.city, split="train", day=day, t=t, idx=idx,
        )
        y = self.torch_road_graph_mapping.load_cc_labels_day_t(
            basedir=self.root, city=self.city, split="train", day=day, t=t, idx=idx,
        )

        data = Data(
            x=x,
            edge_index=self.torch_road_graph_mapping.edge_index,
            y=y,
            edge_attr=self.torch_road_graph_mapping.edge_attr,
        )

        if self.cachedir is not None:
            self.cachedir.mkdir(exist_ok=True, parents=True)
            torch.save(data, cache_file)

        return data
