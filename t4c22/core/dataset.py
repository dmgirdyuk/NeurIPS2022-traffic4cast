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

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Generator, Optional, Tuple

import numpy as np
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
    load_inputs,
)

_logger = get_logger(__name__)


def get_train_val_dataloaders(
    dataset: Dataset,
    split_ratio: float = 0.5,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:

    train_idxs, val_idxs = sample_dataset_by_weeks(dataset, split_ratio)
    train_dataset = Subset(dataset, train_idxs)
    val_dataset = Subset(dataset, val_idxs)

    train_dataloader = DataLoader(
        dataset=train_dataset,  # noqa
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,  # noqa
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    return train_dataloader, val_dataloader


def sample_dataset_by_weeks(
    dataset: T4c22STILDataset, ratio: float
) -> tuple[list[int], list[int]]:
    week_idxs = {idx: dataset.get_week(idx) for idx in range(len(dataset))}
    weeks_to_sample = np.unique(list(week_idxs.values()))
    np.random.shuffle(weeks_to_sample)
    train_weeks = weeks_to_sample[: int(len(weeks_to_sample) * ratio)]
    val_weeks = weeks_to_sample[int(len(weeks_to_sample) * ratio) :]
    train_idxs = [i for i, v in week_idxs.items() if v in train_weeks]
    val_idxs = [i for i, v in week_idxs.items() if v in val_weeks]
    return train_idxs, val_idxs


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


CITIES = ["london", "madrid", "melbourne"]


class T4c22STILDataset(Dataset):  # pylint: disable=abstract-method  # noqa
    def __init__(
        self,
        root: Path | str,
        edge_attributes: Optional[list[str]] = None,
        split: str = "train",
        cachedir: Optional[Path | str] = None,
        limit_ratio: Optional[float] = 1.0,
        day_t_filter: DAY_T_FILTER = day_t_filter_weekdays_daytime_only,
        counters_only: bool = False,
        cities: Optional[list[str]] = None,
    ):
        super().__init__()
        self.root: Path = root
        self.split = split
        self.cachedir = cachedir
        self.limit_ratio = limit_ratio
        self.day_t_filter = day_t_filter
        self.counters_only = counters_only
        self.cities = cities if cities is not None else CITIES

        self.city_day_t: list[tuple[str, int, str, int]] = []
        if self.split == "train":
            df_filter = partial(day_t_filter_to_df_filter, filter=day_t_filter)
            self.city_day_t = self._get_train_indexes()
        elif self.split == "test":
            df_filter = None
            self.city_day_t = self._get_test_indexes()
        else:
            raise ValueError

        self.city_road_graph_mapping = {
            city: TorchRoadGraphMapping(
                city=city,
                edge_attributes=edge_attributes,
                root=root,
                df_filter=df_filter,
                skip_supersegments=True,
                counters_only=self.counters_only,
            )
            for city in self.cities
        }
        self.city = self.cities[0]
        self.torch_road_graph_mapping = self.city_road_graph_mapping[self.city]

    def _get_train_indexes(self) -> list[tuple[str, int, str, int]]:
        city_day_t: list[tuple[str, int, str, int]] = []
        slice_val = int(1 / self.limit_ratio)
        for city in self.cities:
            all_dates = cc_dates(self.root, city=city, split="train")
            for i, week in enumerate(sample_dates(all_dates, 0, slice_val)):
                city_day_t.extend(
                    (city, i, day, t)
                    for day in week
                    for t in range(4, 96)
                    if self.day_t_filter(day, t)
                )
        return city_day_t

    def _get_test_indexes(self) -> list[tuple[str, int, str, int]]:
        assert len(self.cities) == 1, "For test assume 1 dataset for 1 city."
        city = self.cities[0]
        num_tests = (
            load_inputs(
                basedir=self.root, split="test", city=city, day="test", df_filter=None
            )["test_idx"].max()
            + 1
        )  # noqa
        city_day_t = [(city, 0, "test", t) for t in range(num_tests)]
        return city_day_t

    def len(self) -> int:
        dataset_len = len(self.city_day_t)
        return dataset_len

    def get_week(self, idx: int) -> int:
        _, week, _, _ = self.city_day_t[idx]
        return week

    def get(self, idx: int) -> Data:
        self.city, _, day, t = self.city_day_t[idx]
        self.torch_road_graph_mapping = self.city_road_graph_mapping[self.city]

        cache_file = self.cachedir / f"cc_labels_{self.city}_{day}_{t}.pt"
        cache_file_edge = self.cachedir / f"edge_attr_{self.city}.pt"
        if self.cachedir is not None:
            if cache_file.exists():
                data = torch.load(cache_file)
                data["edge_attr"] = torch.load(cache_file_edge)
                if self.split == "train":
                    data["y"] = {"target": data.y, "t": t}
                return data

        x = self.torch_road_graph_mapping.load_inputs_day_t(
            basedir=self.root,
            city=self.city,
            split=self.split,
            day=day,
            t=t,
            idx=idx,
        )

        y = None
        if self.split == "train":
            y = self.torch_road_graph_mapping.load_cc_labels_day_t(
                basedir=self.root,
                city=self.city,
                split="train",
                day=day,
                t=t,
                idx=idx,
            )

        data = Data(
            x=x,
            edge_index=self.torch_road_graph_mapping.edge_index,
            y=y,
        )
        edge_attr = self.torch_road_graph_mapping.edge_attr
        if self.cachedir is not None:
            self.cachedir.mkdir(exist_ok=True, parents=True)
            torch.save(data, cache_file)
            torch.save(edge_attr, cache_file_edge)
        data["edge_attr"] = edge_attr
        return data


def sample_dates(
    dates: list, start: int = 0, every_val: int = 2
) -> Generator[list[str], None, None]:
    weeks = split_dates_list(dates)
    for i in range(start, len(weeks), every_val):
        yield weeks[i]


def split_dates_list(dates: list, chunk_size: int = 7) -> list[list]:
    return [dates[i : i + chunk_size] for i in range(0, len(dates), chunk_size)]
