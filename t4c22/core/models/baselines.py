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
        hidden_features: int,
        out_features: int,
        gnn_layer_num: int = 3,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        self.in_node_features = in_node_features
        self.in_edge_features = in_edge_features
        self.cat_features_num = cat_features_num
        self.cat_emb_sizes = cat_emb_sizes
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.gnn_layer_num = gnn_layer_num
        self.dropout_p = dropout_p

        self.node_emb_mlp = nn.Sequential(
            nn.Linear(self.in_node_features, self.hidden_features),
            nn.BatchNorm1d(self.hidden_features),
            nn.GELU(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(self.hidden_features, self.hidden_features),
            nn.BatchNorm1d(self.hidden_features),
            nn.GELU(),
            nn.Dropout(p=self.dropout_p),
        )

        self.edge_emb_cat_layers = nn.ModuleList(
            nn.Embedding(num, dim) for num, dim in self.cat_emb_sizes
        )
        self.hidden_edge_cat = sum((dim for _, dim in self.cat_emb_sizes))
        self.hidden_edge_num = self.hidden_features - self.hidden_edge_cat
        self.edge_emb_cat_mlp = nn.Sequential(
            nn.BatchNorm1d(self.hidden_edge_cat),
            nn.GELU(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(self.hidden_edge_cat, self.hidden_edge_cat),
            nn.BatchNorm1d(self.hidden_edge_cat),
            nn.GELU(),
            nn.Dropout(p=self.dropout_p),
        )
        self.edge_emb_num_mlp = nn.Sequential(
            nn.Linear(
                self.in_edge_features - self.cat_features_num, self.hidden_edge_num
            ),
            nn.BatchNorm1d(self.hidden_edge_num),
            nn.GELU(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(self.hidden_edge_num, self.hidden_edge_num),
            nn.BatchNorm1d(self.hidden_edge_num),
            nn.GELU(),
            nn.Dropout(p=self.dropout_p),
        )

        self.node_gnn_layers = torch.nn.ModuleList(
            modules=[
                GNNLayer(
                    in_features=self.hidden_features,
                    out_features=self.hidden_features,
                )
                for _ in range(self.gnn_layer_num)
            ]
        )

        self.final_aggregation_layer = nn.Sequential(
            nn.Linear(3 * self.hidden_features, self.hidden_features),
            nn.BatchNorm1d(self.hidden_features),
            nn.GELU(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(self.hidden_features, self.out_features),
        )

    def forward(self, data: Data) -> Tensor:
        edge_index: Tensor = data.edge_index
        node_features: Tensor = data.x
        # 4 last columns are categorical and should be processed separately
        edge_num_features: Tensor = data.edge_attr[:, : -self.cat_features_num]
        edge_cat_features: Tensor = data.edge_attr[:, -self.cat_features_num :].int()

        node_emb = self.node_emb_mlp(node_features)
        edge_emb_num = self.edge_emb_num_mlp(edge_num_features)
        edge_emb_cat = torch.cat(
            [
                emb_layer(edge_cat_features[:, i])
                for i, emb_layer in enumerate(self.edge_emb_cat_layers)
            ],
            dim=-1,
        )
        edge_emb_cat = self.edge_emb_cat_mlp(edge_emb_cat)
        edge_emb = torch.cat((edge_emb_num, edge_emb_cat), dim=-1)

        for layer in self.node_gnn_layers:
            node_emb, edge_emb = layer(
                x=node_emb, edge_index=edge_index, edge_attr=edge_emb
            )

        node_emb_i = torch.index_select(node_emb, 0, data.edge_index[0])
        node_emb_j = torch.index_select(node_emb, 0, data.edge_index[1])

        general_emb = torch.cat((node_emb_i, node_emb_j, edge_emb), dim=-1)
        general_emb = self.final_aggregation_layer(general_emb)

        return general_emb


# pylint: disable=abstract-method, arguments-differ
class GNNLayer(MessagePassing):  # noqa
    def __init__(self, in_features: int, out_features: int):  # , hidden_features: int
        super().__init__(node_dim=-2, aggr="mean")

        self.message_net = nn.Sequential(
            nn.Linear(3 * in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.GELU(),
        )
        self.node_update_net = nn.Sequential(
            nn.Linear(2 * in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.GELU(),
        )
        self.edge_update_net = nn.Sequential(
            nn.Linear(3 * in_features, out_features),
            nn.BatchNorm1d(out_features),
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
