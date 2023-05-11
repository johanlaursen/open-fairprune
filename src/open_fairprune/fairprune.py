import mlflow
import numpy as np
import torch
import torch.nn as nn
from backpack import backpack, extend
from backpack.extensions import DiagHessian
from torch.utils.data import DataLoader

from open_fairprune.data_util import LoanDataset, get_dataset, load_model, timeit
from open_fairprune.eval_model import (
    FairPruneMetrics,
    get_fairness_metrics,
    get_general_metrics,
)

metric = nn.CrossEntropyLoss()


def fairprune(model, metric, train_dataset, device, prune_ratio, beta, privileged_group, unprivileged_group):
    """
    ARGS:
        model: model to be pruned
        metric: loss function to be used for saliency
        train_dataset: train_dataset
        label: label to be predicted
        device: device to run on
        prune_ratio: ratio of parameters to be pruned on each layer
        beta: ratio between 0 and 1 for weighting of privileged group
        privileged_group: privileged group idx in group tensor
        unprivileged_group: unprivileged group idx in group tensor
    """

    model_extend = extend(model).to(device)
    metric_extend = extend(metric).to(device)

    saliency_0_dict = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
    saliency_1_dict = {name: torch.zeros_like(param) for name, param in model.named_parameters()}

    (data, group, target) = train_dataset
    # Explicitely set unprivileged and privileged group to 0 and 1
    group[group == unprivileged_group] = 0
    group[group == privileged_group] = 1

    data = data.to(device)
    group = group.to(device)
    target = target.to(device)

    one_indices = torch.nonzero(group == 1, as_tuple=False).squeeze(1)
    zero_indices = torch.nonzero(group == 0, as_tuple=False).squeeze(1)
    data_0 = torch.index_select(data, 0, zero_indices).to(device)
    data_1 = torch.index_select(data, 0, one_indices).to(device)
    target_0 = torch.index_select(target, 0, zero_indices).to(device)
    target_1 = torch.index_select(target, 0, one_indices).to(device)

    saliency_0_dict = get_parameter_salience(model_extend, metric_extend, data_0, target_0, saliency_0_dict)
    saliency_1_dict = get_parameter_salience(model_extend, metric_extend, data_1, target_1, saliency_1_dict)

    # get difference in saliency for each parameter
    saliency_diff_dict = {
        name: param_0 - beta * param_1
        for (name, param_0), (_, param_1) in zip(saliency_0_dict.items(), saliency_1_dict.items())
    }

    # Prune
    with torch.no_grad():
        for (name, saliency), (_, param) in zip(saliency_diff_dict.items(), model.named_parameters()):
            k = int(prune_ratio * param.numel())
            saliency_flat = saliency.flatten()
            topk_indices = torch.topk(saliency_flat, k).indices
            param.flatten()[topk_indices] = 0

    return model


def get_parameter_salience(model_extend, metric_extend, data, target, saliency_dict):
    output = model_extend(data)
    loss = metric_extend(output.squeeze(), target)
    with backpack(DiagHessian()):
        loss.backward()

    for name, param in model_extend.named_parameters():
        saliency_dict[name] += param.diag_h * torch.square(param)

    return saliency_dict


def hyperparameter_search_fairprune(model, RUNID, device, lossfunc):
    betas = [0.1, 0.3, 0.5, 0.7, 0.9, 1]
    prune_ratios = np.linspace(0, 1, 11)
    data, group, y_true = get_dataset("dev")
    train = get_dataset("train")

    with mlflow.start_run(run_name=f"fairprune_hyperparameter_search_{RUNID}"):
        for beta in betas:
            for prune_ratio in prune_ratios:
                model_pruned = fairprune(
                    model=model,
                    metric=lossfunc,
                    train_dataset=train,
                    device=device,
                    prune_ratio=prune_ratio,
                    beta=beta,
                    privileged_group=1,
                    unprivileged_group=0,
                )
                with torch.no_grad():
                    y_pred = model_pruned(data.to("cuda")).softmax(dim=1).detach().cpu()

                general_metrics = get_general_metrics(y_pred, y_true, group)
                fairprune_metrics = FairPruneMetrics.from_general_metrics(general_metrics)
                mlflow.log_metrics(
                    {
                        f"{beta}_{prune_ratio}_{key}": value.item()
                        for key, value in zip(fairprune_metrics._fields, fairprune_metrics)
                    }
                )
        # fairness_metrics = get_fairness_metrics(y_pred, y_true, group)


if __name__ == "__main__":
    prune_ratio = 0.5
    beta = 0.3
    hyperparameter_search = True

    device = torch.device("cuda")
    lossfunc = nn.CrossEntropyLoss()

    model, RUN_ID = load_model(return_run_id=True)
    client = mlflow.MlflowClient()
    setup = client.get_run(RUN_ID).to_dictionary()["data"]["params"]

    dev = get_dataset("dev")
    train = get_dataset("train")

    if hyperparameter_search:
        with timeit("hyperparameter_search"):
            hyperparameter_search_fairprune(model, RUNID=RUN_ID, device=device, lossfunc=lossfunc)
        exit()
    with timeit("fairprune"):
        model_pruned = fairprune(
            model=model,
            metric=lossfunc,
            train_dataset=train,
            device=device,
            prune_ratio=prune_ratio,
            beta=beta,
            privileged_group=1,
            unprivileged_group=0,
        )

    with mlflow.start_run(RUN_ID):
        mlflow.pytorch.log_model(model_pruned, f"fairpruned_model_{prune_ratio}_{beta}")

        data, group, y_true = get_dataset("dev")

        for model, suffix in [(model, "pre"), (model_pruned, "post")]:
            with torch.no_grad():
                y_pred = model(data.to("cuda")).softmax(dim=1).detach().cpu()

            general_metrics = get_general_metrics(y_pred, y_true, group)
            fairprune_metrics = FairPruneMetrics.from_general_metrics(general_metrics)

            fairness_metrics = get_fairness_metrics(y_pred, y_true, group)

            mlflow.log_metrics(
                {f"{key}_{suffix}": value.item() for key, value in zip(fairprune_metrics._fields, fairprune_metrics)}
            )
            print(fairness_metrics)
