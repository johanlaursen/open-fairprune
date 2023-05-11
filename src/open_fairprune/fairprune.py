import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
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


def fairprune(
    model, metric, train_dataset, device, prune_ratio, beta, privileged_group, unprivileged_group, verbose=False
):
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

    # Prune by layer
    # with torch.no_grad():
    #     for (name, saliency), (_, param) in zip(saliency_diff_dict.items(), model.named_parameters()):
    #         k = int(prune_ratio * param.numel())
    #         saliency_flat = saliency.flatten()
    #         topk_indices = torch.topk(saliency_flat, k).indices
    #         param.flatten()[topk_indices] = 0

    # Prune
    all_params = torch.cat([param.data.view(-1) for name, param in model.named_parameters() if "bias" not in name])
    k = int(prune_ratio * all_params.numel())
    all_saliency = torch.cat(
        [saliency.data.view(-1) for name, saliency in saliency_diff_dict.items() if "bias" not in name]
    )
    topk_indices = torch.topk(all_saliency, k).indices
    all_params[topk_indices] = 0
    param_index = 0
    for name, param in model.named_parameters():
        # Note: bias is not pruned so explicitly avoiding
        if "bias" not in name:
            num_params = param.numel()
            param.data = all_params[param_index : param_index + num_params].view(param.size())
            param_index += num_params

    if verbose:
        num_zeros = 0
        num_elems = 0
        for param in model.parameters():
            num_zeros += torch.sum(param.data == 0).item()
            num_elems += param.numel()
        print(" --------- Pruning Verification ---------")
        print("number of total zeros: ", num_zeros, " out of ", num_elems, " parameters")
        print(" ----------------------------------------")

        for (name, saliency), (_, param) in zip(saliency_diff_dict.items(), model.named_parameters()):
            mean = round(torch.mean(saliency).item(), 5)
            std = round(torch.std(saliency).item(), 5)
            min_value = torch.min(saliency).item()
            max_value = torch.max(saliency).item()
            print(f"{name}, mean:{mean} std:{std}, number of parameters: {saliency.numel()}")
            print(f"num of zeros: {torch.sum(param == 0).item()} out of {param.numel()}")
            print(f"min: {min_value}, max: {max_value}")
        print(" ----------------------------------------")
    return model


def get_parameter_salience(model_extend, metric_extend, data, target, saliency_dict):
    output = model_extend(data)
    loss = metric_extend(output.squeeze(), target)
    with backpack(DiagHessian()):
        loss.backward()

    for name, param in model_extend.named_parameters():
        saliency_dict[name] += param.diag_h * torch.square(param)

    return saliency_dict


def hyperparameter_search_fairprune(RUNID, device, lossfunc):
    """Hyperparameter search for fairprune to reproduce ablation study from paper"""
    betas = [0.1, 0.3, 0.5, 0.7, 0.9]
    prune_ratios = np.linspace(0, 1, 11)
    prune_ratios = np.insert(prune_ratios, np.searchsorted(prune_ratios, 0.35), 0.35)
    data, group, y_true = get_dataset("dev")
    train = get_dataset("train")
    data_recall = {
        "prune_ratio": prune_ratios,
    }
    data_eodd = {
        "prune_ratio": prune_ratios,
    }
    data_mcc_eopp1 = {"prune_ratio": 0.35}
    with mlflow.start_run(run_name=f"fairprune_hyperparameter_search_{RUNID}"):
        matthews_scores = []
        eopp1 = []
        for beta in betas:
            recall_scores = []
            eodd = []
            for prune_ratio in prune_ratios:
                model = load_model(id=RUN_ID)
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
                recall_scores.append(round(general_metrics.tpr.total.item(), 3))
                eodd.append(round(fairprune_metrics.EOdd.item(), 3))
                if prune_ratio == 0.35:
                    matthews_scores.append(round(general_metrics.matthews.total.item(), 3))
                    eopp1.append(round(fairprune_metrics.EOpp1.item(), 3))

            data_recall[f"Beta={beta}"] = recall_scores
            data_eodd[f"Beta={beta}"] = eodd
        df_data_recall = pd.DataFrame(data_recall)
        df_data_eodd = pd.DataFrame(data_eodd)
        print("df_data_recall: ", df_data_recall)
        print(eopp1)
        print(matthews_scores)
        fig = plt.figure(figsize=(10, 10))
        for key in data_recall.keys():
            if "Beta" in key:
                sns.lineplot(data=df_data_recall, x="prune_ratio", y=key, label=key, marker="o")
        plt.ylabel("Recall")
        mlflow.log_figure(fig, f"fairprune_hyperparameter_search_recall.png")
        fig = plt.figure(figsize=(10, 10))
        for key in data_eodd.keys():
            if "Beta" in key:
                sns.lineplot(data=df_data_eodd, x="prune_ratio", y=key, label=key, marker="o")
        plt.ylabel("EOdd")
        mlflow.log_figure(fig, f"fairprune_hyperparameter_search_eodd.png")

        fig = plt.figure(figsize=(10, 10))
        data_mcc_eopp1[f"eopp1"] = eopp1
        data_mcc_eopp1[f"matthews"] = matthews_scores
        df_f1_eopp1 = pd.DataFrame(data_mcc_eopp1)
        sns.lineplot(data=df_f1_eopp1, x="eopp1", y="matthews", label="Prune Ratio 0.35", marker="o")
        plt.ylabel("Matthews")
        plt.xlabel("EOpp1")
        mlflow.log_figure(fig, f"fairprune_hyperparameter_search_matthews_eopp1.png")


if __name__ == "__main__":
    prune_ratio = 0.001
    beta = 0.3
    hyperparameter_search = False
    RUN_ID = "latest"  # latest
    device = torch.device("cuda")
    lossfunc = nn.CrossEntropyLoss()

    model, RUN_ID = load_model(id=RUN_ID, return_run_id=True)
    print("RUN_ID: ", RUN_ID)
    client = mlflow.MlflowClient()
    setup = client.get_run(RUN_ID).to_dictionary()["data"]["params"]

    dev = get_dataset("dev")
    train = get_dataset("train")

    if hyperparameter_search:
        with timeit("hyperparameter_search"):
            hyperparameter_search_fairprune(RUNID=RUN_ID, device=device, lossfunc=lossfunc)
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
            verbose=True,
        )

    with mlflow.start_run(RUN_ID):
        mlflow.pytorch.log_model(model_pruned, f"fairpruned_model_{prune_ratio}_{beta}")

        data, group, y_true = get_dataset("dev")
        model, RUN_ID = load_model(id=RUN_ID, return_run_id=True)  # reload to avoid overwriting

        for model, suffix in [(model, "pre"), (model_pruned, "post")]:
            with torch.no_grad():
                y_pred = model(data.to("cuda")).softmax(dim=1).detach().cpu()

            general_metrics = get_general_metrics(y_pred, y_true, group)
            fairprune_metrics = FairPruneMetrics.from_general_metrics(general_metrics)

            fairness_metrics = get_fairness_metrics(y_pred, y_true, group)

            mlflow.log_metrics(
                {f"{key}_{suffix}": value.item() for key, value in zip(fairprune_metrics._fields, fairprune_metrics)}
            )
            mlflow.log_metrics(
                {f"{key}_{suffix}": value.total.item() for key, value in zip(general_metrics._fields, general_metrics)}
            )
            print(f" Fairprune metrics for {suffix} pruning: ")
            print(fairness_metrics)
            print(general_metrics)
