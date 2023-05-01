import mlflow
import torch
from backpack import backpack, extend
from backpack.extensions import DiagHessian
from torch.utils.data import DataLoader

from open_fairprune.data_util import LoanDataset, load_model, timeit
from open_fairprune.train import metric


def fairprune(
    model, metric, data_loader, device, prune_ratio, beta, privileged_group, unprivileged_group, num_of_batches=5
):
    """
    ARGS:
        model: model to be pruned
        metric: loss function to be used for saliency
        data_loader: data loader for the dataset
        label: label to be predicted
        device: device to run on
        prune_ratio: ratio of parameters to be pruned on each layer
        beta: ratio between 0 and 1 for weighting of privileged group
        privileged_group: privileged group idx in group tensor
        unprivileged_group: unprivileged group idx in group tensor
        num_of_batches: number of batches to be used for saliency calculation

    """

    model_extend = extend(model).to(device)
    metric_extend = extend(metric).to(device)

    saliency_0_dict = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
    saliency_1_dict = {name: torch.zeros_like(param) for name, param in model.named_parameters()}

    for batch_idx, (data, group, target) in enumerate(data_loader):

        # Excplicitely set unprivileged and privileged group to 0 and 1
        group[group == unprivileged_group] = 0
        group[group == privileged_group] = 1
        # TODO make sure masked select preserves device

        target = target.to(device)

        one_indices = torch.nonzero(group == 1, as_tuple=False).squeeze(1)
        zero_indices = torch.nonzero(group == 0, as_tuple=False).squeeze(1)
        data_0 = torch.index_select(data, 0, zero_indices).to(device)
        data_1 = torch.index_select(data, 0, one_indices).to(device)
        target_0 = torch.index_select(target, 0, zero_indices).to(device)
        target_1 = torch.index_select(target, 0, one_indices).to(device)

        saliency_0_dict = get_parameter_salience(model_extend, metric_extend, data_0, target_0, saliency_0_dict)
        saliency_1_dict = get_parameter_salience(model_extend, metric_extend, data_1, target_1, saliency_1_dict)

        # number of batches to average over
        if batch_idx + 1 == num_of_batches:
            break

    # average saliency over batches
    for name, param in saliency_0_dict.items():
        saliency_0_dict[name] = param / num_of_batches

    for name, param in saliency_1_dict.items():
        saliency_1_dict[name] = param / num_of_batches

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
        # TODO add paramters**2 to saliency_dict
        saliency_dict[name] += param.diag_h * torch.square(param)

    return saliency_dict


if __name__ == "__main__":
    prune_ratio = 0.5
    beta = 0.3

    device = torch.device("cpu")
    lossfunc = metric

    model, RUN_ID = load_model(return_run_id=True)
    client = mlflow.MlflowClient()
    setup = client.get_run(RUN_ID).to_dictionary()["data"]["params"]
    data_kwargs = {
        # "num_workers": 4,
        "pin_memory": True,
        "shuffle": True,  # Only done once for entire run
        "batch_size": int(setup["batch_size"]),
        "drop_last": True,  # Drop last batch if it's not full
    }

    train_loader, dev_loader = [
        DataLoader(
            LoanDataset(
                split="train",
                returns=["data", "group", "label"],
            ),
            **data_kwargs,
        )
        for split in ["train", "dev"]
    ]

    with timeit("fairprune"):
        model_pruned = fairprune(
            model=model,
            metric=lossfunc,
            data_loader=train_loader,
            device=device,
            prune_ratio=prune_ratio,
            beta=beta,
            privileged_group=1,
            unprivileged_group=0,
            num_of_batches=5,
        )

    with mlflow.start_run(RUN_ID):
        mlflow.pytorch.log_model(model_pruned, f"fairpruned_model_{prune_ratio}_{beta}")
