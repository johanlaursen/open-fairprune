import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from backpack import backpack, extend
from backpack.extensions import DiagHessian

from open_fairprune.data_util import get_dataset, load_model, timeit
from open_fairprune.eval_model import get_all_metrics

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

    (data, group, target) = [x.to(device) for x in train_dataset]
    # Explicitely set unprivileged and privileged group to 0 and 1
    g0 = group == unprivileged_group
    g1 = group == privileged_group

    h0 = get_parameter_salience(model_extend, metric_extend, data[g0], target[g0])
    h1 = get_parameter_salience(model_extend, metric_extend, data[g1], target[g1])

    θ = torch.cat([param.flatten() for param in model_extend.parameters()])

    saliency = 1 / 2 * θ**2 * (h0 - beta * h1)

    k = int(prune_ratio * len(θ))
    topk_indices = torch.topk(-saliency, k).indices
    θ[topk_indices] = 0

    param_index = n_pruned = n_param = 0
    for name, param in model.named_parameters():
        # Note: bias is not pruned so explicitly avoiding
        if "bias" in name:
            continue
        num_params = param.numel()
        layer_saliency = θ[param_index : param_index + num_params].view(param.size())
        param.data = layer_saliency
        param_index += num_params

        if verbose:
            n_pruned += torch.sum(param.data == 0).item()
            n_param += num_params
            mean = round(torch.mean(layer_saliency).item(), 5)
            std = round(torch.std(layer_saliency).item(), 5)
            min_value = torch.min(layer_saliency).item()
            max_value = torch.max(layer_saliency).item()
            n_positive_predictions = model(data[g0]).softmax(dim=1).argmax(axis=1).sum()
            print(f"{name = } {mean = } {std = } {num_params = }")
            print(f"num of zeros: {torch.sum(param == 0).item()} / {num_params}")
            print(f"{min_value = } {max_value = } {n_positive_predictions = }")
            print(" ----------------------------------------")

    if verbose:
        print(" --------- Pruning Verification ---------")
        print("number of total zeros: ", n_pruned, " out of ", n_param, " parameters")
        print(" ----------------------------------------")

    return model


def get_parameter_salience(model_extend, metric_extend, data, target):
    output = model_extend(data)
    loss = metric_extend(output, target)
    with backpack(DiagHessian()):
        loss.backward()

    return torch.cat([param.diag_h.flatten() for param in model_extend.parameters()])


def hyperparameter_search_fairprune(RUN_ID, device, lossfunc):
    """Hyperparameter search for fairprune to reproduce ablation study from paper"""
    betas = [0.1, 0.3, 0.5, 0.7, 0.9]
    # prune_ratios = np.linspace(0, 0.001, 11)
    prune_ratios = np.linspace(0, 0.1, 11)
    target_tradeoff = 0.05
    # prune_ratios = np.insert(prune_ratios, np.searchsorted(prune_ratios, target_tradeoff), target_tradeoff)
    data, group, y_true = get_dataset("dev")
    train = get_dataset("train")
    data_accuracy = {"prune_ratio": prune_ratios}
    data_eodd = {"prune_ratio": prune_ratios}
    data_mcc_eopp1 = {"prune_ratio": target_tradeoff}
    accuracy_metric = "accuracy"
    fairness_metric = "parity"
    with mlflow.start_run(run_id=RUN_ID):
        matthews_scores = []
        eopp1 = []
        for beta in betas:
            accuracy_scores = []
            fairness_scores = []
            for prune_ratio in prune_ratios:
                model = load_model(id=RUN_ID)
                model.eval()

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
                model_pruned.eval()
                with torch.no_grad():
                    y_pred = model_pruned(data.to("cuda")).softmax(dim=1).detach().cpu()

                fairness_metrics, general_metrics, fairprune_metrics = get_all_metrics(y_pred, y_true, group)
                accuracy_scores.append(round(general_metrics.accuracy.total.item(), 3))
                fairness_scores.append(round(fairprune_metrics.EOdd.item(), 3))
                if prune_ratio == target_tradeoff:
                    matthews_scores.append(round(general_metrics.matthews.total.item(), 3))
                    eopp1.append(round(fairprune_metrics.EOpp1.item(), 3))

            data_accuracy[f"Beta={beta}"] = accuracy_scores
            data_eodd[f"Beta={beta}"] = fairness_scores
        df_data_accuracy = pd.DataFrame(data_accuracy)
        df_data_eodd = pd.DataFrame(data_eodd)
        print(f"df_data_{accuracy_metric}: ", df_data_accuracy)
        print(eopp1)
        print(matthews_scores)
        fig = plt.figure(figsize=(6, 6))
        for key in data_accuracy.keys():
            if "Beta" in key:
                sns.lineplot(data=df_data_accuracy, x="prune_ratio", y=key, label=key, marker="o")
        plt.ylabel(accuracy_metric)
        mlflow.log_figure(fig, f"fairprune_hyperparameter_search_{accuracy_metric}.png")
        fig = plt.figure(figsize=(6, 6))
        for key in data_eodd.keys():
            if "Beta" in key:
                sns.lineplot(data=df_data_eodd, x="prune_ratio", y=key, label=key, marker="o")
        plt.ylabel(fairness_metric)
        mlflow.log_figure(fig, f"fairprune_hyperparameter_search_{fairness_metric}.png")

        fig = plt.figure(figsize=(6, 6))
        data_mcc_eopp1[f"eopp1"] = eopp1
        data_mcc_eopp1[f"matthews"] = matthews_scores
        df_f1_eopp1 = pd.DataFrame(data_mcc_eopp1)
        sns.lineplot(data=df_f1_eopp1, x="eopp1", y="matthews", label=f"Prune Ratio {target_tradeoff}", marker="o")
        plt.ylabel("Matthews")
        plt.xlabel("EOpp1")
        plt.title(f"Matthews vs EOpp1 at {target_tradeoff} Prune Ratio")
        mlflow.log_figure(fig, f"fairprune_hyperparameter_search_matthews_eopp1.png")


if __name__ == "__main__":
    prune_ratio = 0.001
    beta = 0.3
    hyperparameter_search = True
    RUN_ID = "latest"
    device = torch.device("cuda")
    lossfunc = nn.CrossEntropyLoss()

    model, RUN_ID = load_model(id=RUN_ID, return_run_id=True)
    print("RUN_ID: ", RUN_ID)
    client = mlflow.MlflowClient()
    setup = client.get_run(RUN_ID).to_dictionary()["data"]["params"]

    train = get_dataset("train")

    if hyperparameter_search:
        with timeit("hyperparameter_search"):
            hyperparameter_search_fairprune(RUN_ID=RUN_ID, device=device, lossfunc=lossfunc)
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

        for model, suffix in [(model, "pre_"), (model_pruned, "post_")]:
            with torch.no_grad():
                model.eval()
                y_pred = model(data.to("cuda")).softmax(dim=1).detach().cpu()

            fairness_metrics, general_metrics, fairprune_metrics = get_all_metrics(
                y_pred, y_true, group, log_mlflow_w_suffix=suffix
            )
            print(f" Fairprune metrics for {suffix} pruning: ", fairness_metrics, general_metrics, sep="\n")
