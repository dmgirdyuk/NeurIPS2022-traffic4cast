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
from torch.nn import functional as F  # noqa
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing  # noqa


class T4c22GNN(torch.nn.Module):
    def __init__(
        self,
        in_node_features: int,
        hidden_features_mlp: tuple[int, int],
        hidden_features_gnn: int,
        hidden_final: int,
        out_features: int,
        gnn_layer_num: int = 3,
        dropout: float = 0.0,
        dropout_gnn: float = 0.0,
    ):
        super().__init__()

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

        assert hidden_features_gnn == hidden_features_mlp[-1], "Dimensions mismatch."
        self.node_gnn_layers = torch.nn.ModuleList(
            modules=[
                GNNLayer(
                    in_features=hidden_features_gnn,
                    out_features=hidden_features_gnn,
                    dropout=dropout_gnn,
                )
                for _ in range(gnn_layer_num)
            ]
        )

        self.final_aggregation_layer = nn.Sequential(
            nn.Linear(hidden_features_gnn, hidden_final),
            nn.BatchNorm1d(hidden_final),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_final, out_features),
        )

    def forward(self, data: Data) -> dict[str, Tensor]:
        edge_index: Tensor = data.edge_index
        node_features: Tensor = data.x
        node_emb = self.node_emb_mlp(node_features)

        node_emb_pre_gnn = node_emb
        for i, layer in enumerate(self.node_gnn_layers):
            node_emb = layer(x=node_emb, edge_index=edge_index)
        node_emb += node_emb_pre_gnn

        node_emb_i = torch.index_select(node_emb, 0, data.edge_index[0])
        node_emb_j = torch.index_select(node_emb, 0, data.edge_index[1])

        general_emb = node_emb_j - node_emb_i
        output = self.final_aggregation_layer(general_emb)

        return self._postprocess_output(output)

    @staticmethod
    def _postprocess_output(output: Tensor) -> dict[str, Tensor]:
        return {"cc_scores": output}


class T4c22GNNwTwWD(T4c22GNN):
    @staticmethod
    def _postprocess_output(output: Tensor) -> dict[str, Tensor]:
        return {
            "cc_scores": output[:, :-2],
            "t": output[:, -2],
            "working_day": output[:, -1],
        }


# pylint: disable=abstract-method, arguments-differ
class GNNLayer(MessagePassing):  # noqa
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.0):
        super().__init__(node_dim=-2, aggr="sum")

        self.message_net = nn.Sequential(
            nn.Linear(2 * in_features, out_features),
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

    def forward(self, x: Tensor, edge_index: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.propagate(edge_index, x=x)
        return x

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:  # noqa
        message = self.message_net(torch.cat((x_i, x_j), dim=-1))
        return message

    def update(self, message: Tensor, x: Tensor) -> Tensor:  # noqa
        x += self.node_update_net(torch.cat((x, message), dim=-1))
        return x
