import typing

import click
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from mlflow import log_metric, log_params
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from open_fairprune import simple_nn
from open_fairprune.data_util import LoanDataset, get_git_hash, load_model
from open_fairprune.eval_model import get_all_metrics
from open_fairprune.simple_nn import MODEL_NAME

torch.manual_seed(42)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

device = torch.device("cuda")
METRIC = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 11.0]).to(device))


@click.command()
@click.option("--batch-size", default=int(87040 / 7))
@click.option("--lr", default=1e-5, type=float)
@click.option("--decay", default=0.0, type=float)
@click.option("--gamma", default=1.0, type=float)
@click.option("--epochs", default=10)
@click.option("--checkpoint", default="", type=str)
@click.option("--fairness", default=0.0, type=float)
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
    model = load_model(params["checkpoint"]) if params["checkpoint"] else simple_nn.model

    returns = ["data", "group", "label"]
    return ExperimentSetup(
        model=model,
        model_name=f"{MODEL_NAME}",
        dataset_kwargs={
            "returns": returns,
        },
        params=params,
    )


def train(model, device, train_loader, optimizer, metric):
    model.train()
    train_loss = 0
    for data, group, target in train_loader:
        data, group, target = data.to(device), group.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = metric(output, target, group)
        loss.backward()
        train_loss += loss
        optimizer.step()

    return train_loss / len(train_loader)


def test(model, device, test_loader, epoch, metric):
    model.eval()
    test_loss = correct = total = 0
    with torch.no_grad():
        for data, group, target in test_loader:
            data, group, target = data.to(device), group.to(device), target.to(device)
            output = model(data)
            test_loss += metric(output, target, group)
            correct += (output.argmax(axis=1) == target).sum()
            total += len(target)
            get_all_metrics(output, target, group, log_mlflow_w_suffix="_dev")

    print(f"Test Epoch {epoch}: Avg loss: {(test_loss / len(test_loader)):.4f}, Avg accuracy = {correct / total:.4f}")
    return test_loss / len(test_loader)


def metric_fairness_loss(output, target, group):
    g0 = group == 0
    g1 = group == 1
    y_trues = torch.cartesian_prod(target[g0], target[g1])
    y_preds = torch.cartesian_prod(output[g0][:, 1], output[g1][:, 1])

    assert len(y_trues) == len(target[g0]) * len(target[g1]), f"{len(y_trues)} != {len(target[g0]) * len(target[g1])}"

    probability_differences = (y_trues[:, 0] == y_trues[:, 1]) * (y_preds[:, 0] - y_preds[:, 1])
    individual_fairness = (probability_differences**2).mean()
    return individual_fairness


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

    def metric(y_pred, y_true, group):
        λ = setup.params["fairness"]
        return METRIC(y_pred, y_true) + (0 if λ == 0 else λ * metric_fairness_loss(y_pred, y_true, group))

    best_test_loss = 1e16
    with mlflow.start_run():
        mlflow.log_param("git_hash", get_git_hash())
        try:
            log_params(setup.params)

            for epoch in range(1, setup.params["epochs"] + 1):
                train_loss = train(model, device, train_loader, optimizer, metric)
                if (test_loss := test(model, device, dev_loader, epoch, metric)) < best_test_loss and epoch > 3:
                    best_test_loss = test_loss
                    best_model = model

                scheduler.step()

                log_metric("train loss", train_loss, step=epoch)
                log_metric("val loss", test_loss, step=epoch)
        finally:
            mlflow.pytorch.log_model(best_model, "model")


if __name__ == "__main__":
    init_cli()
    exit()

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
