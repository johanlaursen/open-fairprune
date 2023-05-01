from backpack import backpack, extend
from backpack.extensions import DiagHessian, BatchDiagHessian
import torch

from open_fairprune.data_util import load_model
from open_fairprune import metric
RUN_ID = ""
model = load_model(RUN_ID)
num_of_batches = 5
def fairprune(
            model,
            metric,
            data_loader,
            device,
            prune_ratio,
            beta,
            privileged_group,
            unprivileged_group,
            num_of_batches=5):
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

    model_extend = extend(model)
    metric_extend = extend(metric)

    saliency_0_dict = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
    saliency_1_dict = {name: torch.zeros_like(param) for name, param in model.named_parameters()}

    for batch_idx, (*data, group, target) in enumerate(data_loader):

        # Excplicitely set unprivileged and privileged group to 0 and 1
        group[group == unprivileged_group] = 0
        group[group == privileged_group] = 1
        # TODO make sure masked select preserves device

        data = [d.to(device) for d in data]
        target = target.to(device)

        data_0, data_1 = [torch.masked_select(data, group == i) for i in range(2)]
        target_0, target_1 = [torch.masked_select(target, group == i) for i in range(2)]
        
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
    saliency_diff_dict = {name: param_0 - beta*param_1 for (name, param_0), (_, param_1) in zip(saliency_0_dict.items(), saliency_1_dict.items())}

    # Prune
    with torch.no_grad():
        for (name, saliency), (_, param)  in zip(saliency_diff_dict.items(), model.named_parameters()):
            k = int(prune_ratio * param.numel())
            topk_indices = torch.topk(saliency,k).indices
            param[topk_indices] = 0
        
    
    # TODO check that the pruning is actually done in place

    return model
         


def get_parameter_salience(model_extend, metric_extend, data, target, saliency_dict):
    output = model_extend(*data)
    loss = metric_extend(output.squeeze(), target)
    with backpack(DiagHessian()):
        loss.backward()
    
    for name, param_0 in model.named_parameters():
            saliency_dict[name] += param_0
    
    return saliency_dict



    # for name, param in model.named_parameters():
    #     print(name)
    #     print(".grad.shape:             ", param.grad.shape)
    #     print(".diag_h.shape:           ", param.diag_h.shape)
    #     print(".diag_h_batch.shape:     ", param.diag_h_batch.shape)


    

    
    # get idx of maximum saliency for each group
