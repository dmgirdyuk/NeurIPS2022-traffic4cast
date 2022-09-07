#  Copyright 2022 STIL at home.
#  IARAI licenses this file to You under the Apache License, Version 2.0
#  (the "License"); you may not use this file except in compliance with
#  the License. You may obtain a copy of the License at
#  http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import torch
import torch.nn.functional as F  # noqa
from torch import nn
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
        node_edge_aggregation_layers_num: int = 2,
        dropout_p: float = 0.0,
    ):

        super().__init__()
        self.in_node_features = in_node_features
        self.in_edge_features = in_edge_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.gnn_layer_num = gnn_layer_num
        self.node_edge_aggregation_layers_num = node_edge_aggregation_layers_num
        self.dropout_p = dropout_p

        self.node_embedding_mlp = nn.Sequential(
            nn.Linear(self.in_node_features, self.hidden_features),
            Swish(),
            nn.BatchNorm1d(self.hidden_features),
        )

        # TODO: rework edge embedding
        self.edge_embedding_mlp = nn.Sequential(
            nn.Linear(self.in_edge_features, self.hidden_features),
            Swish(),
            nn.BatchNorm1d(self.hidden_features),
        )

        self.node_gnn_layers = torch.nn.ModuleList(
            modules=[
                GNNLayer(
                    self.hidden_features, self.hidden_features, self.hidden_features
                )
                for _ in range(self.gnn_layer_num)
            ]
        )
        self.node_edge_aggregation_layers = torch.nn.ModuleList(
            modules=[
                NodeEdgeAggregationLayer(
                    in_features=2 * self.hidden_features,
                    out_features=self.hidden_features,
                    dropout_p=self.dropout_p,
                ),
                *[
                    NodeEdgeAggregationLayer(
                        in_features=self.hidden_features,
                        out_features=self.hidden_features,
                        dropout_p=self.dropout_p,
                    )
                    for _ in range(self.node_edge_aggregation_layers_num - 2)
                ],
                NodeEdgeAggregationLayer(
                    in_features=self.hidden_features,
                    out_features=self.out_features,
                    dropout_p=self.dropout_p,
                ),
            ]
        )

    def forward(self, data: Data) -> torch.Tensor:
        edge_index: torch.Tensor = data.edge_index
        node_emb: torch.Tensor = data.x
        edge_emb: torch.Tensor = data.edge_attr

        node_emb = self.node_embedding_mlp(node_emb)
        edge_emb = self.edge_embedding_mlp(edge_emb)

        for layer in self.node_gnn_layers:
            node_emb = layer(x=node_emb, edge_index=edge_index)

        node_emb_i = torch.index_select(node_emb, 0, data.edge_index[0])
        node_emb_j = torch.index_select(node_emb, 0, data.edge_index[1])

        general_emb = torch.cat((node_emb_j - node_emb_i, edge_emb), dim=-1)

        for layer in self.node_edge_aggregation_layers:
            general_emb = layer(edge_emb=general_emb)

        return general_emb


class Swish(nn.Module):
    def __init__(self, beta: int = 1):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.beta * x)


class GNNLayer(MessagePassing):
    def __init__(self, in_features: int, out_features: int, hidden_features: int):
        super().__init__(node_dim=-2, aggr="mean")

        self.message_net = nn.Sequential(
            nn.Linear(2 * in_features, hidden_features),
            Swish(),
            nn.BatchNorm1d(hidden_features),
            nn.Linear(hidden_features, out_features),
            Swish(),
        )
        self.update_net = nn.Sequential(
            nn.Linear(in_features + hidden_features, hidden_features),
            Swish(),
            nn.Linear(hidden_features, out_features),
            Swish(),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        x = self.propagate(edge_index, x=x)
        return x

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor):  # noqa
        message = self.message_net(torch.cat((x_i, x_j), dim=-1))
        return message

    def update(self, message: torch.Tensor, x: torch.Tensor):  # noqa
        x += self.update_net(torch.cat((x, message), dim=-1))
        return x


class NodeEdgeAggregationLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout_p: float = 0.0):
        super().__init__()

        self.dropout_p = dropout_p
        self.emb_layer = nn.Sequential(
            nn.Linear(in_features, out_features),
            Swish(),
            nn.BatchNorm1d(out_features),
        )

    def forward(self, edge_emb: torch.Tensor) -> torch.Tensor:
        edge_emb = self.emb_layer(edge_emb)
        edge_emb = F.dropout(edge_emb, p=self.dropout_p, training=self.training)
        return edge_emb
