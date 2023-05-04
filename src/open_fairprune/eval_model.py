import typing
from contextlib import suppress
from lib2to3.pgen2.pgen import generate_grammar

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchmetrics import AUROC, Accuracy, F1Score, Recall, Specificity
from torchmetrics.classification import BinaryAccuracy

from open_fairprune.data_util import get_dataset, load_model
from open_fairprune.train import metric


class GroupScore(typing.NamedTuple):
    total: float
    group0: float
    group1: float

    def __repr__(self):
        total, group0, group1 = self.total, self.group0, self.group1
        return f"{total = :.2f} ({group0 = :.2f}, {group1 = :.2f})"


class GeneralMetrics(typing.NamedTuple):
    accuracy: GroupScore
    f1: GroupScore
    tpr: GroupScore  # recall, sensitivity
    fpr: GroupScore  # 1 - specificity
    tnr: GroupScore  # Specificity

    def __repr__(self):
        out = []
        for field in self._fields:
            out.append(f"{field:<10}: {getattr(self, field)}")
        return "\n".join(out)


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


class FairPruneMetrics(typing.NamedTuple):
    # Paper cites [8] https://papers.nips.cc/paper_files/paper/2016/file/9d2682367c3935defcb1f9e247a97c0d-Paper.pdf
    EOpp0: float
    EOpp1: float
    EOdd: float
    EOdd_sep_abs: float

    @classmethod
    def from_general_metrics(cls, metrics: GeneralMetrics) -> "FairPruneMetrics":
        # For EOpp: The paper sums this for y=1 and y=0, but in binary classification we can just do y=1?
        EOpp0 = abs(metrics.tnr.group0 - metrics.tnr.group1)
        EOpp1 = abs(metrics.tpr.group0 - metrics.fpr.group1)
        EOdd = abs(metrics.tpr.group1 - metrics.tpr.group0 + metrics.fpr.group1 - metrics.fpr.group0)
        EOdd_sep_abs = abs(metrics.tpr.group1 - metrics.tpr.group0) + abs(metrics.fpr.group1 - metrics.fpr.group0)
        return cls(EOpp0, EOpp1, EOdd, EOdd_sep_abs)

    def __repr__(self):
        EOpp0, EOpp1, EOdd, EOdd_sep_abs = self.EOpp0, self.EOpp1, self.EOdd, self.EOdd_sep_abs
        return f"{EOpp0 = :.2f} {EOpp1 = :.2f} {EOdd = :.2f} {EOdd_sep_abs = :.2f}"


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

    pr_t_cond_gs = df.groupby(["G", "S"])["T"].mean()
    with suppress(KeyError):
        Δ_eq_out_s0 = abs(pr_t_cond_gs.xs(False, level="S").diff()[-1])
    with suppress(KeyError):
        Δ_eq_out_s1 = abs(pr_t_cond_gs.xs(True, level="S").diff()[-1])

    if verbose:
        print(
            f"{pr_s_cond_g  = }",
            f"{pr_s_cond_gt = }",
            f"{pr_t_cond_gs = }",
            sep="\n",
        )

    return FairnessMetrics(Δ_parity, Δ_eq_odds_t0, Δ_eq_odds_t1, Δ_eq_out_s0, Δ_eq_out_s1)


def get_general_metrics(y_pred, y_true, group, thresh=0.5) -> FairnessMetrics:
    df = pd.DataFrame(
        {
            "G": group,
            "y_true": y_true,
            "y_pred": y_pred[:, 1],
        }
    )
    gb = df.groupby("G")
    g0_df, g1_df = [gb.get_group(group) for group in sorted(gb.groups)]

    kwargs = dict(task="binary", threshold=thresh)
    metrics = dict(  # order is important
        accuracy=Accuracy(**kwargs),
        f1=F1Score(**kwargs),
        tpr=Recall(**kwargs),
        fpr=lambda x, y: 1 - Specificity(**kwargs)(x, y),
        tnr=Specificity(**kwargs),
    )

    groups = [df, g0_df, g1_df]  # order is important

    results = []
    for metric_func in metrics.values():
        group_metrics = []
        for subset_df in groups:
            subset_y_pred, subset_y_true = torch.tensor(subset_df.y_pred.values), torch.tensor(subset_df.y_true.values)
            group_metrics.append(metric_func(subset_y_pred, subset_y_true))
        results.append(GroupScore(*group_metrics))

    return GeneralMetrics(*results)


def ROC_curve(y_pred, y_true, group):
    import holoviews as hv
    import hvplot.pandas

    hv.extension("bokeh")
    data = []
    for threshold in range(1, 100, 1):
        metrics = get_general_metrics(y_pred, y_true, group, thresh=threshold / 100)
        data.append((0, metrics.fpr.group0, metrics.tpr.group0, threshold / 100))
        data.append((1, metrics.fpr.group1, metrics.tpr.group1, threshold / 100))

    df = pd.DataFrame(data, columns=["G", "fpr", "tpr", "thresh"])
    mid_df = df.query("thresh==0.5")

    kwargs = dict(
        by="G",
        x="fpr",
        y="tpr",
        hover_cols=["thresh"],
        xlabel="False Positive Rate = Pr(S=1 | G=g, T=0)",
        ylabel="True Positive Rate = Pr(S=1 | G=g, T=1)",
    )
    ROC_plot = df.hvplot(**kwargs) * df.hvplot.scatter(**kwargs)
    mids = mid_df.hvplot.scatter(**kwargs, marker="square")
    ROC_plot * mids

    mids
    return ROC_plot * mids


if __name__ == "__main__":
    favors_minority_100_to_one = "020ffd14ba2f44458ab0b435fabc6bab"
    favors_minority_10_to_one = "368701f1065f4cd18fad9b51e7ec961f"

    model = load_model()

    data, group, y_true = get_dataset("dev")

    with torch.no_grad():
        y_pred = model(data.to("cuda")).softmax(dim=1).detach().cpu()

    fairness_metrics = get_fairness_metrics(y_pred, y_true, group, thresh=0.5)
    print()
    print(fairness_metrics)

    general_metrics = get_general_metrics(y_pred, y_true, group)
    print()
    print(general_metrics)

    fairprune_metrics = FairPruneMetrics.from_general_metrics(general_metrics)
    print()
    print(fairprune_metrics)
