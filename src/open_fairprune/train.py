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
from open_fairprune.data_util import get_dataset, get_git_hash, load_model
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


def train(model, device, train_data, optimizer, metric):
    model.train()
    train_loss = 0

    n_chunks = 4
    chunk_size = len(train_data[0]) // n_chunks
    slices = [slice(i * chunk_size, (i + 1) * chunk_size) for i in range(n_chunks)]

    for s in slices:
        data, group, target = [d[s].to(device) for d in train_data]
        optimizer.zero_grad()
        output = model(data)
        loss = metric(output, target, group)
        loss.backward()
        train_loss += loss
        optimizer.step()
    return train_loss


def test(model, device, test_data, epoch, metric):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        data, group, target = [d.to(device) for d in test_data]
        output = model(data)
        test_loss += metric(output, target, group)
        fairness, general, fairprune = get_all_metrics(output, target, group, log_mlflow_w_suffix="_dev")

    print(
        f"{epoch:02}: E[loss]={test_loss:.4f}, Macro[acc]={general.accuracy.total:.0%} G0[acc]={general.accuracy.group0:.0%} G1[acc]={general.accuracy.group1:.0%} G0[recall]={general.tpr.group0:.0%} G1[recall]={general.tpr.group1:.0%}"
    )
    return test_loss


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
    train_data, dev_data = get_dataset("train"), get_dataset("dev")

    device = torch.device("cuda")
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
                train_loss = train(model, device, train_data, optimizer, metric)
                if (test_loss := test(model, device, dev_data, epoch, metric)) < best_test_loss and epoch > 3:
                    best_test_loss = test_loss
                    best_model = model

                scheduler.step()

                log_metric("train loss", train_loss, step=epoch)
                log_metric("val loss", test_loss, step=epoch)
        finally:
            mlflow.pytorch.log_model(best_model, "model")


"""
# This checkpoint has basically no training
train-prune --checkpoint 46521209a91e46758f2201ca95750a2e --lr 1e-4 --epochs 150 --fairness 0.1 &&
train-prune --checkpoint 46521209a91e46758f2201ca95750a2e --lr 1e-4 --epochs 150 --fairness 0.25 &&
train-prune --checkpoint 46521209a91e46758f2201ca95750a2e --lr 1e-4 --epochs 150 --fairness 0.5 &&
train-prune --checkpoint 46521209a91e46758f2201ca95750a2e --lr 1e-4 --epochs 150 --fairness 1 &&
train-prune --checkpoint 46521209a91e46758f2201ca95750a2e --lr 1e-4 --epochs 150 --fairness 2 &&
train-prune --checkpoint 46521209a91e46758f2201ca95750a2e --lr 1e-4 --epochs 150 --fairness 4 &&
train-prune --checkpoint 46521209a91e46758f2201ca95750a2e --lr 1e-4 --epochs 150 --fairness 8 &&
train-prune --checkpoint 46521209a91e46758f2201ca95750a2e --lr 1e-4 --epochs 150 --fairness 16 &&
train-prune --checkpoint 46521209a91e46758f2201ca95750a2e --lr 1e-4 --epochs 150 --fairness 32 &&
train-prune --checkpoint 46521209a91e46758f2201ca95750a2e --lr 1e-4 --epochs 150 --fairness 64 &&
train-prune --checkpoint 46521209a91e46758f2201ca95750a2e --lr 1e-4 --epochs 150 --fairness 128 &&
train-prune --checkpoint 46521209a91e46758f2201ca95750a2e --lr 1e-4 --epochs 150 --fairness 256 &&
train-prune --checkpoint 46521209a91e46758f2201ca95750a2e --lr 1e-4 --epochs 150 --fairness 512 &&
train-prune --checkpoint 46521209a91e46758f2201ca95750a2e --lr 1e-4 --epochs 150 --fairness 1024 &&
train-prune --checkpoint 46521209a91e46758f2201ca95750a2e --lr 1e-4 --epochs 150 --fairness 2048 &&
train-prune --checkpoint 46521209a91e46758f2201ca95750a2e --lr 1e-4 --epochs 150 --fairness 4096 &&

# This checkpoint has converged
train-prune --checkpoint 7b9c67bcf82b40328baf2294df5bd1a6 --lr 1e-4 --epochs 25 --fairness 0.1 &&
train-prune --checkpoint 7b9c67bcf82b40328baf2294df5bd1a6 --lr 1e-4 --epochs 25 --fairness 0.25 &&
train-prune --checkpoint 7b9c67bcf82b40328baf2294df5bd1a6 --lr 1e-4 --epochs 25 --fairness 0.5 &&
train-prune --checkpoint 7b9c67bcf82b40328baf2294df5bd1a6 --lr 1e-4 --epochs 25 --fairness 1 &&
train-prune --checkpoint 7b9c67bcf82b40328baf2294df5bd1a6 --lr 1e-4 --epochs 25 --fairness 2 &&
train-prune --checkpoint 7b9c67bcf82b40328baf2294df5bd1a6 --lr 1e-4 --epochs 25 --fairness 4 &&
train-prune --checkpoint 7b9c67bcf82b40328baf2294df5bd1a6 --lr 1e-4 --epochs 25 --fairness 8 &&
train-prune --checkpoint 7b9c67bcf82b40328baf2294df5bd1a6 --lr 1e-4 --epochs 25 --fairness 16 &&
train-prune --checkpoint 7b9c67bcf82b40328baf2294df5bd1a6 --lr 1e-4 --epochs 25 --fairness 32 &&
train-prune --checkpoint 7b9c67bcf82b40328baf2294df5bd1a6 --lr 1e-4 --epochs 25 --fairness 64 &&
train-prune --checkpoint 7b9c67bcf82b40328baf2294df5bd1a6 --lr 1e-4 --epochs 25 --fairness 128 &&
train-prune --checkpoint 7b9c67bcf82b40328baf2294df5bd1a6 --lr 1e-4 --epochs 25 --fairness 256 &&
train-prune --checkpoint 7b9c67bcf82b40328baf2294df5bd1a6 --lr 1e-4 --epochs 25 --fairness 512 &&
train-prune --checkpoint 7b9c67bcf82b40328baf2294df5bd1a6 --lr 1e-4 --epochs 25 --fairness 1024 &&
train-prune --checkpoint 7b9c67bcf82b40328baf2294df5bd1a6 --lr 1e-4 --epochs 25 --fairness 2048 &&
train-prune --checkpoint 7b9c67bcf82b40328baf2294df5bd1a6 --lr 1e-4 --epochs 25 --fairness 4096 &&
"""

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
