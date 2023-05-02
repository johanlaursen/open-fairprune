import typing
from contextlib import suppress

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, F1Score
from torchmetrics.classification import BinaryAccuracy

from open_fairprune.data_util import LoanDataset, load_model
from open_fairprune.train import metric


class FairnessMetrics(typing.NamedTuple):
    Δ_parity: float
    Δ_eq_odds_t0: float
    Δ_eq_odds_t1: float
    Δ_eq_out_s0: float
    Δ_eq_out_s1: float

    def __repr__(self):
        return f"""
        {self.Δ_parity     = :.2f}
        {self.Δ_eq_odds_t0 = :.2f}
        {self.Δ_eq_odds_t1 = :.2f}
        {self.Δ_eq_out_s0  = :.2f}
        {self.Δ_eq_out_s1  = :.2f}"""


def get_fairness_metrics(y_pred, y_true, group, *, thresh=0.5, verbose=False) -> FairnessMetrics:
    assert y_pred.ndim == 2
    assert y_true.ndim == group.ndim == 1

    is_probs = y_pred.sum(axis=1).isclose(torch.ones_like(y_pred.sum(axis=1))).all()
    if not is_probs:
        raise ValueError("y_pred isn't probabilities. Try running y_pred.softmax(dim=1) first")

    df = pd.DataFrame(
        {
            "G": group.numpy().astype(bool),
            "T": y_true.cpu().numpy().astype(bool),
            "S": y_pred[:, 1].cpu().numpy() > thresh,
        }
    )
    # If error, return nan
    Δ_parity = Δ_eq_odds_t0 = Δ_eq_odds_t1 = Δ_eq_out_s0 = Δ_eq_out_s1 = np.nan

    pr_s_cond_g = df.groupby("G").S.mean()
    Δ_parity = abs(pr_s_cond_g.diff()[-1])

    pr_s_cond_gt = df.groupby(["G", "T"]).S.mean()
    # .xs just selects the rows with the given value for the given index level
    with suppress(KeyError):
        Δ_eq_odds_t0 = abs(pr_s_cond_gt.xs(False, level="T").diff()[-1])
    with suppress(KeyError):
        Δ_eq_odds_t1 = abs(pr_s_cond_gt.xs(True, level="T").diff()[-1])
    pr_s_cond_gt

    pr_t_cond_gs = df.groupby(["G", "S"])["T"].mean()
    with suppress(KeyError):
        Δ_eq_out_s0 = abs(pr_t_cond_gs.xs(False, level="S").diff()[-1])
    with suppress(KeyError):
        Δ_eq_out_s1 = abs(pr_t_cond_gs.xs(True, level="S").diff()[-1])

    if verbose:
        print(
            f"{pr_s_cond_g}",
            f"{pr_s_cond_gt = }",
            f"{pr_t_cond_gs = }",
            sep="\n",
        )

    return FairnessMetrics(Δ_parity, Δ_eq_odds_t0, Δ_eq_odds_t1, Δ_eq_out_s0, Δ_eq_out_s1)


if __name__ == "__main__":
    favors_minority_100_to_one = "020ffd14ba2f44458ab0b435fabc6bab"
    favors_minority_10_to_one = "368701f1065f4cd18fad9b51e7ec961f"

    model = load_model(favors_minority_10_to_one)

    dataset = LoanDataset("dev", returns=["data", "group", "label"])

    data_kwargs = {
        # "num_workers": 4,
        "shuffle": True,  # Only done once for entire run
        "batch_size": len(dataset),
        "drop_last": True,  # Drop last batch if it's not full
    }

    dev_loader = DataLoader(
        dataset,
        **data_kwargs,
    )

    data, group, y_true = next(iter(dev_loader))

    with torch.no_grad():
        y_pred = model(data.to("cuda")).softmax(1).detach().cpu()

    thresh = 0.5
    fairness_metrics = get_fairness_metrics(y_pred, y_true, group, thresh)
    print(fairness_metrics)

    BinaryAccuracy()(y_pred[:, 1], y_true)

    # accuracy = Accuracy("multiclass", num_classes=2)(y_pred, y_true)
    # accuracy
    # y_pred
