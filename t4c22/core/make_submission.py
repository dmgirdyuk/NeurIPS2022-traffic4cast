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

import hydra
import pandas as pd
import torch
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.loader.dataloader import DataLoader
from tqdm.auto import tqdm

from t4c22.core.checkpointer import load_checkpoint
from t4c22.core.dataset import CITIES
from t4c22.core.utils import get_logger, get_project_root
from t4c22.misc.parquet_helpers import write_df_to_parquet

_logger = get_logger(__file__)


def prepare_model(cfg, checkpoint_path, accelerator):
    model: nn.Module = instantiate(cfg.model)
    model = load_checkpoint(model=model,
                            load_path=checkpoint_path)
    model = accelerator.prepare(model)
    model.eval()
    return model

def main(cfg: DictConfig) -> None:
    accelerator: Accelerator = instantiate(cfg.accelerator)
    cfg.dataset.root = Path(cfg.dataset.root)
    cfg.dataset.cachedir = Path(cfg.dataset.cachedir)

    models = {}

    if not cfg.split_city_models:
        model = prepare_model(cfg, pjoin(cfg.checkpoint_dir, cfg.checkpoint_name+'.pt'), accelerator)
    else:
        model = None

    for city in CITIES:
        if cfg.split_city_models:
            model = prepare_model(cfg, pjoin(cfg.checkpoint_dir, city, cfg.checkpoint_name+'.pt'), accelerator)
        models[city] = model

    dataloaders = {
        city: accelerator.prepare(
            DataLoader(
                dataset=instantiate(cfg.dataset.test, cities=[city]),
                num_workers=cfg.num_workers,
            )
        )
        for city in CITIES
    }
    for city in CITIES:
        make_submission(
            models[city],
            dataloaders=dataloaders,
            base_dir=cfg.dataset.root,
            submission_name=cfg.submission_name,
            cities=[city]
        )


def make_submission(
    model: nn.Module,
    dataloaders: dict[str, DataLoader],
    base_dir: Path,
    submission_name: str,
    cities: Optional[list[str]] = None,
):
    if cities is None:
        cities = CITIES

    for city in cities:
        print("City ==> ", city)
        assert city in dataloaders
        test_dataloader = dataloaders[city]
        df_city = inference_cc_city_torch_geometric_to_pandas(
            test_dataloader=test_dataloader, predict=model
        )

        labels_dir = base_dir / "submission" / submission_name / city / "labels"
        labels_dir.mkdir(exist_ok=True, parents=True)
        write_df_to_parquet(df=df_city, fn=labels_dir / f"cc_labels_test.parquet")

    submission_zip = base_dir / "submission" / f"{submission_name}.zip"
    with zipfile.ZipFile(submission_zip, "w") as z:
        for city in cities:
            labels_dir = base_dir / "submission" / submission_name / city / "labels"
            z.write(
                filename=labels_dir / "cc_labels_test.parquet",
                arcname=pjoin(city, "labels", "cc_labels_test.parquet"),
            )

    assert submission_zip.exists()
    for city in cities:
        labels_dir = base_dir / "submission" / submission_name / city / "labels"
        assert (labels_dir / "cc_labels_test.parquet").exists()


@torch.no_grad()
def inference_cc_city_torch_geometric_to_pandas(
    test_dataloader: DataLoader, predict: Callable[[Data], Tensor]
) -> pd.DataFrame:
    dfs = []
    for idx, data in enumerate(tqdm(test_dataloader)):
        data["x"] = torch.log10((data.x + 1).nan_to_num(1e-1))
        data["edge_attr"] = data.edge_attr.nan_to_num(0)
        y_hat: Tensor = predict(data)
        df = test_dataloader.dataset.torch_road_graph_mapping._torch_to_df_cc(  # noqa
            data=y_hat, day="test", t=idx
        )
        dfs.append(df)
    df = pd.concat(dfs)

    df["test_idx"] = df["t"]
    del df["day"]
    del df["t"]

    return df


if __name__ == "__main__":
    CONFIG_PATH = pjoin(get_project_root(), "t4c22", "core", "config")
    CONFIG_NAME = sys.argv[1] if len(sys.argv) > 1 else "config_example"
    hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base="1.2")(
        main
    )()
