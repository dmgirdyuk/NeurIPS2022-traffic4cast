# moved code from exploration/example_torch_geometric_dummy_GNN.ipynb

import statistics
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_geometric
from torch import nn
from torch_geometric.nn import MessagePassing
from tqdm.auto import tqdm

import t4c22
from t4c22.dataloading.t4c22_dataset_geometric import T4c22GeometricDataset
from t4c22.metric.masked_crossentropy import get_weights_from_class_fractions
from t4c22.misc.notebook_helpers import restartkernel  # noqa:F401
from t4c22.misc.t4c22_logging import t4c_apply_basic_logging_config
from t4c22.t4c22_config import class_fractions, load_basedir


HIDDEN_CHANNELS = 128
NUM_LAYERS = 3
BATCH_SIZE = 1
NUM_WORKERS = 8
EVAL_STEPS = 1
EPOCHS = 20
RUNS = 1
DROPOUT = 0.0
NUM_EDGE_CLASSES = 3
NUM_FEATURES = 4


def main(city: str = "london", cache_dir: str = "/tmp/processed"):
    t4c_apply_basic_logging_config(loglevel="DEBUG")
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, val_dataset = get_datasets(city, cache_dir)

    city_class_fractions = class_fractions[city]
    city_class_weights = torch.tensor(
        get_weights_from_class_fractions(
            [city_class_fractions[c] for c in ["green", "yellow", "red"]]
        )
    ).float()
    city_class_weights = city_class_weights.to(device)

    model = CongestionNN(NUM_FEATURES, HIDDEN_CHANNELS, HIDDEN_CHANNELS, NUM_LAYERS)
    model = model.to(device)
    predictor = LinkPredictor(
        HIDDEN_CHANNELS, HIDDEN_CHANNELS, NUM_EDGE_CLASSES, NUM_LAYERS, DROPOUT
    ).to(device)

    run_training(model, predictor, device, train_dataset, val_dataset, city_class_weights)


def get_datasets(city: str, cache_dir: str):
    dataset = T4c22GeometricDataset(
        root=load_basedir(fn=Path("t4c22_config.json"), pkg=t4c22),
        city=city,
        split="train",
        cachedir=Path(cache_dir),
    )
    spl = int(((0.8 * len(dataset)) // 2) * 2)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [spl, len(dataset) - spl]
    )

    return train_dataset, val_dataset


def run_training(model, predictor, device, train_dataset, val_dataset, city_class_weights):
    train_losses = defaultdict(lambda: [])
    val_losses = defaultdict(lambda: -1)

    for run in tqdm(range(RUNS), desc="runs", total=RUNS):
        # model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.AdamW(
            [{"params": model.parameters()}, {"params": predictor.parameters()}],
            lr=5e-4,
            weight_decay=0.001,
        )

        for epoch in tqdm(range(1, 1 + EPOCHS), "epochs", total=EPOCHS):
            losses = train_epoch(
                model,
                predictor,
                dataset=train_dataset,
                optimizer=optimizer,
                batch_size=BATCH_SIZE,
                device=device,
                city_class_weights=city_class_weights,
            )
            train_losses[(run, epoch)] = losses

            print(statistics.mean(losses))
            if epoch % EVAL_STEPS == 0:
                val_loss = test_epoch(
                    model,
                    predictor,
                    validation_dataset=val_dataset,
                    batch_size=BATCH_SIZE,
                    device=device,
                    city_class_weights=city_class_weights,
                )
                val_losses[(run, epoch)] = val_loss
                print(f"val_loss={val_loss} after epoch {epoch} of run {run}")
                torch.save(model.state_dict(), f"GNN_model_{epoch:03d}.pt")
                torch.save(predictor.state_dict(), f"GNN_predictor_{epoch:03d}.pt")

    for e, v in train_losses.items():
        print(e)
        print(statistics.mean(v))

    for e, v in val_losses.items():
        print(e)
        print(v)


def train_epoch(model, predictor, dataset, optimizer, batch_size, device, city_class_weights):
    model.train()

    losses = []
    optimizer.zero_grad()

    for data in tqdm(
        torch_geometric.loader.dataloader.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS
        ),
        "train",
        total=len(dataset) // batch_size,
    ):

        data = data.to(device)

        data.x = data.x.nan_to_num(-1)

        h = model(data)
        assert (h.isnan()).sum() == 0, h
        x_i = torch.index_select(h, 0, data.edge_index[0])
        x_j = torch.index_select(h, 0, data.edge_index[1])

        y_hat = predictor(x_i, x_j)

        y = data.y.nan_to_num(-1)
        y = y.long()

        loss_f = torch.nn.CrossEntropyLoss(weight=city_class_weights, ignore_index=-1)
        loss = loss_f(y_hat, y)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.cpu().item())

    return losses


@torch.no_grad()
def test_epoch(model, predictor, validation_dataset, batch_size, device, city_class_weights):
    model.eval()

    y_hat_list = []
    y_list = []
    for data in tqdm(validation_dataset, "test", total=len(validation_dataset)):
        data = data.to(device)

        data.x = data.x.nan_to_num(-1)
        h = model(data)

        x_i = torch.index_select(h, 0, data.edge_index[0])
        x_j = torch.index_select(h, 0, data.edge_index[1])

        y_hat = predictor(x_i, x_j)

        y_hat_list.append(y_hat)
        y_list.append(data.y)

    y_hat = torch.cat(y_hat_list, 0)
    y = torch.cat(y_list, 0)
    y = y.nan_to_num(-1)
    y = y.long()
    loss = torch.nn.CrossEntropyLoss(weight=city_class_weights, ignore_index=-1)
    total_loss = loss(y_hat, y)
    print(f"total losses {total_loss}")
    return total_loss


class Swish(nn.Module):
    def __init__(self, beta=1):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class GNNLayer(MessagePassing):
    """
    Parameters
    ----------
    in_features : int
        Dimensionality of input features.
    out_features : int
        Dimensionality of output features.
    """

    def __init__(self, in_features, out_features, hidden_features):
        super(GNNLayer, self).__init__(node_dim=-2, aggr="mean")

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

    def forward(self, x, edge_index, batch):
        """Propagate messages along edges."""
        x = self.propagate(edge_index, x=x)
        # x = self.norm(x, batch)
        return x

    def message(self, x_i, x_j):
        """Message update."""
        message = self.message_net(torch.cat((x_i, x_j), dim=-1))
        return message

    def update(self, message, x):
        """Node update."""
        x += self.update_net(torch.cat((x, message), dim=-1))
        return x


class CongestionNN(torch.nn.Module):
    def __init__(
        self, in_features=4, out_features=32, hidden_features=32, hidden_layer=1
    ):

        super(CongestionNN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.hidden_layer = hidden_layer

        # in_features have to be of the same size as out_features for the time being
        self.cgnn = torch.nn.ModuleList(
            modules=[
                GNNLayer(self.out_features, self.out_features, self.hidden_features)
                for _ in range(self.hidden_layer)
            ]
        )

        self.head_pre_pool = nn.Sequential(
            nn.Linear(self.out_features, self.hidden_features),
            Swish(),
            nn.Linear(self.hidden_features, self.hidden_features),
        )
        self.head_post_pool = nn.Sequential(
            nn.Linear(self.hidden_features, self.hidden_features),
            Swish(),
            nn.Linear(hidden_features, 1),
        )

        self.embedding_mlp = nn.Sequential(
            nn.Linear(self.in_features, self.out_features)
        )

    def forward(self, data):
        batch = data.batch
        x = data.x
        edge_index = data.edge_index

        x = self.embedding_mlp(x)
        for i in range(self.hidden_layer):
            x = self.cgnn[i](x, edge_index, batch)

        return x


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.swish = Swish()

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = self.swish(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)

        return x


if __name__ == "__main__":
    main()
