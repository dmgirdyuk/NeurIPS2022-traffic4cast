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

import sys
import zipfile
from os.path import join as pjoin
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import torch
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor, nn
from torch_geometric.data import Data
from tqdm.auto import tqdm

import hydra
from t4c22.core.dataset import CITIES
from t4c22.core.utils import get_logger, get_project_root
from t4c22.dataloading.t4c22_dataset_geometric import T4c22GeometricDataset
from t4c22.misc.parquet_helpers import write_df_to_parquet

_logger = get_logger(__file__)


@torch.no_grad()
def inference_cc_city_torch_geometric_to_pandas(
    test_dataset: T4c22GeometricDataset, predict: Callable[[Data], Tensor]
) -> pd.DataFrame:
    dfs = []
    for idx, data in tqdm(enumerate(test_dataset), total=len(test_dataset)):
        data["x"] = torch.log10((data.x + 1).nan_to_num(1e-6))
        data["edge_attr"] = data.edge_attr.nan_to_num(0)
        y_hat: Tensor = predict(data)
        df = test_dataset.torch_road_graph_mapping._torch_to_df_cc(  # NOQA
            data=y_hat, day="test", t=idx
        )
        dfs.append(df)
    df = pd.concat(dfs)

    df["test_idx"] = df["t"]
    del df["day"]
    del df["t"]
    return df


def make_submission(
    model: nn.Module,
    base_dir: Path,
    submission_name: str,
    edge_attributes: list[str],
    cities: Optional[list[str]] = None,
):
    if cities is None:
        cities = CITIES

    for city in cities:
        print("City ==> ", city)
        test_dataset = T4c22GeometricDataset(
            root=base_dir, city=city, split="test", edge_attributes=edge_attributes
        )
        df_city = inference_cc_city_torch_geometric_to_pandas(
            test_dataset=test_dataset, predict=model
        )
        (base_dir / "submission" / submission_name / city / "labels").mkdir(
            exist_ok=True, parents=True
        )
        write_df_to_parquet(
            df=df_city,
            fn=base_dir
            / "submission"
            / submission_name
            / city
            / "labels"
            / f"cc_labels_test.parquet",
        )

    submission_zip = base_dir / "submission" / f"{submission_name}.zip"
    with zipfile.ZipFile(submission_zip, "w") as z:
        for city in cities:
            z.write(
                filename=base_dir
                / "submission"
                / submission_name
                / city
                / "labels"
                / "cc_labels_test.parquet",
                arcname=pjoin(city, "labels", "cc_labels_test.parquet"),
            )
    assert submission_zip.exists()
    for city in cities:
        assert (
            base_dir
            / "submission"
            / submission_name
            / city
            / "labels"
            / "cc_labels_test.parquet"
        ).exists()


def main(cfg: DictConfig) -> None:
    accelerator: Accelerator = instantiate(cfg.accelerator)
    cfg.dataset.root = Path(cfg.dataset.root)
    cfg.dataset.cachedir = Path(cfg.dataset.cachedir)

    model: nn.Module = instantiate(cfg.model)
    checkpoint = torch.load(cfg.checkpoint_path, map_location=accelerator.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = accelerator.prepare(model)
    model.eval()
    make_submission(
        model,
        base_dir=cfg.dataset.root,
        submission_name=cfg.submission_name,
        edge_attributes=cfg.dataset.edge_attributes,
    )


if __name__ == "__main__":
    CONFIG_PATH = pjoin(get_project_root(), "t4c22", "core", "config")
    CONFIG_NAME = sys.argv[1] if len(sys.argv) > 1 else "config_example"
    hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base="1.2")(
        main
    )()
