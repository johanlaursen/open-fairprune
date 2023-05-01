import typing

import click
import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from mlflow import log_metric, log_params
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from open_fairprune.data_util import DATA_PATH, LoanDataset, load_model, timeit
from open_fairprune.simple_nn import MODEL_NAME, Net

torch.manual_seed(42)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


@click.command()
@click.option("--batch-size", default=128)
@click.option("--lr", default=2.0, type=float)
@click.option("--decay", default=0.0, type=float)
@click.option("--gamma", default=1.0, type=float)
@click.option("--epochs", default=10)
@click.option("--checkpoint", default="", type=str)
def init_cli(
    **kwargs,
):
    main(get_setup(**kwargs))


class ExperimentSetup(typing.NamedTuple):
    """
    Encapsulate all model specific information into a single object.
    May have to extend it with model specific hyperparams?
    """

    model: nn.Module  # Accepts a batch of data and spits out class probabilities
    model_name: str  # Name of model
    dataset_kwargs: dict[str, any]
    params: dict  # Hyperparameters
    dataloader_kwargs: dict[str, any] = {}


def get_setup(**params) -> ExperimentSetup:
    model = load_model(params["checkpoint"]) if params["checkpoint"] else Net()

    return ExperimentSetup(
        model=model,
        model_name=f"{MODEL_NAME}",
        dataset_kwargs={
            "returns": ["data", "label"],
        },
        params=params,
    )


def train(model, device, train_loader, optimizer):
    model.train()
    train_loss = 0
    for data, target in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = metric(output, target)
        loss.backward()
        train_loss += loss
        optimizer.step()

    return train_loss / len(train_loader)


def metric(y_pred, y_true):
    # Adapted from: https://www.kaggle.com/code/rejpalcz/best-loss-function-for-f1-score-metric/notebook#Optimal-loss-function---macro-F1-score
    assert y_pred.ndim == y_true.ndim == 1
    tp = (y_true * y_pred).sum()
    fp = ((1 - y_true) * y_pred).sum()
    fn = (y_true * (1 - y_pred)).sum()

    f1 = 2 * tp / (2 * tp + fp + fn)
    assert 0 <= f1 <= 1, f"{f1 = }"
    return 1 / f1  # Minimise loss, so maximizes F1


def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += metric(output, target)

    print(f"Test set Epoch {epoch}: Average loss: {(test_loss / len(test_loader)):.4f}")
    return test_loss / len(test_loader)


def main(setup: ExperimentSetup):
    device = torch.device("cuda")

    data_kwargs = {
        # "num_workers": 4,
        "pin_memory": True,
        "shuffle": True,  # Only done once for entire run
        "batch_size": setup.params["batch_size"],
        "drop_last": True,  # Drop last batch if it's not full
    }

    train_loader, dev_loader = [
        DataLoader(
            LoanDataset(split, **setup.dataset_kwargs),
            **data_kwargs,
            **setup.dataloader_kwargs,
        )
        for split in ["train", "dev"]
    ]

    model = setup.model.to(device)

    optimizer = optim.Adam(params=model.parameters(), lr=setup.params["lr"], weight_decay=setup.params["decay"])

    scheduler = StepLR(optimizer, step_size=1, gamma=setup.params["gamma"])

    best_test_loss = 999
    with mlflow.start_run():
        try:
            log_params(setup.params)

            for epoch in range(1, setup.params["epochs"] + 1):
                train_loss = train(model, device, train_loader, optimizer)
                if (test_loss := test(model, device, dev_loader, epoch)) < best_test_loss:
                    best_test_loss = test_loss

                scheduler.step()

                log_metric("train loss", train_loss, step=epoch)
                log_metric("val loss", test_loss, step=epoch)
        finally:
            mlflow.pytorch.log_model(model, "model")


if __name__ == "__main__":
    # init_cli()

    kwargs = dict(
        batch_size=int(87040 / 7),  # No greater than this or dev set doesnt work
        lr=1e-5,
        decay=0.0,
        gamma=0.90,
        epochs=10,
        checkpoint="",
    )
    setup = get_setup(**kwargs)
    main(setup)
