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

from typing import Tuple

import torch
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing  # noqa


class T4c22GNN(torch.nn.Module):
    def __init__(
        self,
        in_node_features: int,
        in_edge_features: int,
        cat_features_num: int,
        cat_emb_sizes: list[tuple[int, int]],
        hidden_features_mlp: tuple[int, int],
        hidden_features_gnn: int,
        hidden_final: int,
        out_features: int,
        gnn_layer_num: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.cat_features_num = cat_features_num

        self.node_emb_mlp = nn.Sequential(
            nn.Linear(in_node_features, hidden_features_mlp[0]),
            nn.BatchNorm1d(hidden_features_mlp[0]),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_features_mlp[0], hidden_features_mlp[1]),
            nn.BatchNorm1d(hidden_features_mlp[1]),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )

        self.edge_emb_cat_layers = nn.ModuleList(
            nn.Embedding(num, dim) for num, dim in cat_emb_sizes
        )
        hidden_edge_cat = sum((dim for _, dim in cat_emb_sizes))
        numer_to_cat_ratio = hidden_edge_cat / hidden_features_mlp[0]
        cat_emb_hidden_dim = int(numer_to_cat_ratio * hidden_features_mlp[1])
        self.edge_emb_cat_mlp = nn.Sequential(
            nn.BatchNorm1d(hidden_edge_cat),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_edge_cat, cat_emb_hidden_dim),
            nn.BatchNorm1d(cat_emb_hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )
        hidden_edge_numer = hidden_features_mlp[0] - hidden_edge_cat
        numer_emb_hidden_dim = hidden_features_mlp[1] - cat_emb_hidden_dim
        self.edge_emb_numer_mlp = nn.Sequential(
            nn.Linear(
                in_edge_features - cat_features_num, hidden_edge_numer
            ),
            nn.BatchNorm1d(hidden_edge_numer),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_edge_numer, numer_emb_hidden_dim),
            nn.BatchNorm1d(numer_emb_hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )

        assert hidden_features_gnn == hidden_features_mlp[-1], "Dimensions mismatch."
        self.node_gnn_layers = torch.nn.ModuleList(
            modules=[
                GNNLayer(
                    in_features=hidden_features_gnn,
                    out_features=hidden_features_gnn,
                )
                for _ in range(gnn_layer_num)
            ]
        )

        self.final_aggregation_layer = nn.Sequential(
            nn.Linear(3 * hidden_features_gnn, hidden_final),
            nn.BatchNorm1d(hidden_final),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_final, out_features),
        )

    def forward(self, data: Data) -> Tensor:
        edge_index: Tensor = data.edge_index
        node_features: Tensor = data.x
        edge_numer_features: Tensor = data.edge_attr[:, : -self.cat_features_num]
        edge_cat_features: Tensor = data.edge_attr[:, -self.cat_features_num :].int()

        node_emb = self.node_emb_mlp(node_features)
        edge_emb_numer = self.edge_emb_numer_mlp(edge_numer_features)
        edge_emb_cat = torch.cat(
            [
                emb_layer(edge_cat_features[:, i])
                for i, emb_layer in enumerate(self.edge_emb_cat_layers)
            ],
            dim=-1,
        )
        edge_emb_cat = self.edge_emb_cat_mlp(edge_emb_cat)
        edge_emb = torch.cat((edge_emb_numer, edge_emb_cat), dim=-1)
        node_emb_pre_gnn, edge_emb_pre_gnn = node_emb, edge_emb

        for i, layer in enumerate(self.node_gnn_layers):
            node_emb, edge_emb = layer(
                x=node_emb, edge_index=edge_index, edge_attr=edge_emb
            )

        node_emb += node_emb_pre_gnn
        edge_emb += edge_emb_pre_gnn

        node_emb_i = torch.index_select(node_emb, 0, data.edge_index[0])
        node_emb_j = torch.index_select(node_emb, 0, data.edge_index[1])

        general_emb = torch.cat((node_emb_i, node_emb_j, edge_emb), dim=-1)
        general_emb = self.final_aggregation_layer(general_emb)

        return general_emb


# pylint: disable=abstract-method, arguments-differ
class GNNLayer(MessagePassing):  # noqa
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.0):
        super().__init__(node_dim=-2, aggr="sum")

        self.message_net = nn.Sequential(
            nn.Linear(3 * in_features, out_features),
            nn.LayerNorm(out_features),
            nn.Dropout(p=dropout),
            nn.GELU(),
        )
        self.node_update_net = nn.Sequential(
            nn.Linear(2 * in_features, out_features),
            nn.LayerNorm(out_features),
            nn.Dropout(p=dropout),
            nn.GELU(),
        )
        self.edge_update_net = nn.Sequential(
            nn.Linear(3 * in_features, out_features),
            nn.LayerNorm(out_features),
            nn.Dropout(p=dropout),
            nn.GELU(),
        )

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_attr: Tensor
    ) -> Tuple[Tensor, Tensor]:
        x = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        edge_attr = self.edge_updater(edge_index, x=x, edge_attr=edge_attr)
        return x, edge_attr

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:  # noqa
        message = self.message_net(torch.cat((x_i, x_j, edge_attr), dim=-1))
        return message

    def update(self, message: Tensor, x: Tensor) -> Tensor:  # noqa
        x += self.node_update_net(torch.cat((x, message), dim=-1))
        return x

    def edge_update(  # noqa
        self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor  # noqa
    ) -> Tensor:  # noqa
        edge_attr += self.edge_update_net(torch.cat((edge_attr, x_i, x_j), dim=-1))
        return edge_attr
