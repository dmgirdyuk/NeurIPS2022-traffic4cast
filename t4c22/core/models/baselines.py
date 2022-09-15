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
import torch.nn.functional as F  # noqa
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing  # noqa


class T4c22GNN(torch.nn.Module):
    def __init__(
        self,
        in_node_features: int,
        in_edge_features: int,
        hidden_features: int,
        out_features: int,
        gnn_layer_num: int = 3,
        dropout_p: float = 0.0,
    ):

        super().__init__()
        self.in_node_features = in_node_features
        self.in_edge_features = in_edge_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.gnn_layer_num = gnn_layer_num
        self.dropout_p = dropout_p

        self.node_embedding_mlp = nn.Sequential(
            nn.Linear(self.in_node_features, self.hidden_features),
            nn.BatchNorm1d(self.hidden_features),
            nn.PReLU(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(self.hidden_features, self.hidden_features),
            nn.BatchNorm1d(self.hidden_features),
            nn.PReLU(),
            nn.Dropout(p=self.dropout_p),
        )

        # TODO: rework edge embedding
        self.edge_embedding_mlp = nn.Sequential(
            nn.Linear(self.in_edge_features, self.hidden_features),
            nn.BatchNorm1d(self.hidden_features),
            nn.PReLU(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(self.hidden_features, self.hidden_features),
            nn.BatchNorm1d(self.hidden_features),
            nn.PReLU(),
            nn.Dropout(p=self.dropout_p),
        )

        self.node_gnn_layers = torch.nn.ModuleList(
            modules=[
                GNNLayer(
                    in_features=self.hidden_features,
                    out_features=self.hidden_features,
                    hidden_features=self.hidden_features,
                )
                for _ in range(self.gnn_layer_num)
            ]
        )

        self.final_aggregation_layer = nn.Sequential(
            nn.Linear(2 * self.hidden_features, self.hidden_features),
            nn.BatchNorm1d(self.hidden_features),
            nn.PReLU(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(self.hidden_features, self.out_features),
        )

    def forward(self, data: Data) -> Tensor:
        edge_index: Tensor = data.edge_index
        node_emb: Tensor = data.x
        edge_emb: Tensor = data.edge_attr

        node_emb = self.node_embedding_mlp(node_emb)
        edge_emb = self.edge_embedding_mlp(edge_emb)

        for layer in self.node_gnn_layers:
            node_emb, edge_emb = layer(
                x=node_emb, edge_index=edge_index, edge_attr=edge_emb
            )

        node_emb_i = torch.index_select(node_emb, 0, data.edge_index[0])
        node_emb_j = torch.index_select(node_emb, 0, data.edge_index[1])

        general_emb = torch.cat((node_emb_j - node_emb_i, edge_emb), dim=-1)
        general_emb = self.final_aggregation_layer(general_emb)

        return general_emb


class Swish(nn.Module):
    def __init__(self, beta: int = 1):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(self.beta * x)


class GNNLayer(MessagePassing):  # noqa
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int,
    ):
        super().__init__(node_dim=-2, aggr="mean")

        self.message_net = nn.Sequential(
            nn.Linear(3 * in_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.PReLU(),
            nn.Linear(hidden_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.PReLU(),
        )
        self.node_update_net = nn.Sequential(
            nn.Linear(in_features + hidden_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.PReLU(),
            nn.Linear(hidden_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.PReLU(),
        )
        self.edge_update_net = nn.Sequential(
            nn.Linear(in_features + hidden_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.PReLU(),
            nn.Linear(hidden_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.PReLU(),
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
        edge_attr += self.edge_update_net(torch.cat((edge_attr, x_j - x_i), dim=-1))
        return edge_attr
