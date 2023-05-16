import typing
import warnings
from contextlib import suppress
from dataclasses import dataclass

import holoviews as hv
import hvplot.pandas
import mlflow
import numpy as np
import pandas as pd
import panel as pn
import torch
from bokeh.models import FixedTicker
from torchmetrics import Accuracy, MatthewsCorrCoef, Recall, Specificity

from open_fairprune.data_util import filter_mlflow_data, get_dataset, load_model

warnings.filterwarnings("ignore", category=UserWarning)


hv.extension("bokeh")

BASE_RUN_ID = "7b9c67bcf82b40328baf2294df5bd1a6"


@dataclass
class GroupScore:
    group0: float
    group1: float
    total: float = None

    def __init__(self, group0, group1):
        self.group0 = torch.tensor(group0)
        self.group1 = torch.tensor(group1)
        self.total = torch.tensor((group0 + group1) / 2)

    def __repr__(self):
        total, group0, group1 = self.total, self.group0, self.group1
        return f"{total = :.2f} ({group0 = :.2f}, {group1 = :.2f})"

    def _fields(self):
        return ("total", "group0", "group1")


class GeneralMetrics(typing.NamedTuple):
    accuracy: GroupScore
    matthews: GroupScore
    tpr: GroupScore  # Recall, Sensitivity
    fpr: GroupScore  # 1 - Specificity
    tnr: GroupScore  # Specificity
    fnr: GroupScore  # 1 - Recall

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
        EOpp1 = abs(metrics.tpr.group0 - metrics.tpr.group1)
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
        matthews=MatthewsCorrCoef(**kwargs),
        tpr=Recall(**kwargs),
        fpr=lambda x, y: 1 - Specificity(**kwargs)(x, y),
        tnr=Specificity(**kwargs),
        fnr=lambda x, y: 1 - Recall(**kwargs)(x, y),
    )

    groups = [g0_df, g1_df]  # order is important

    results = []
    for metric_func in metrics.values():
        group_metrics = []
        for subset_df in groups:
            subset_y_pred, subset_y_true = torch.tensor(subset_df.y_pred.values), torch.tensor(subset_df.y_true.values)
            group_metrics.append(metric_func(subset_y_pred, subset_y_true))
        results.append(GroupScore(*group_metrics))

    return GeneralMetrics(*results)


def get_MCC(y_pred, y_true, thresh=0.5) -> FairnessMetrics:
    df = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred[:, 1],
        }
    )
    kwargs = dict(task="binary", threshold=thresh)
    matthews = MatthewsCorrCoef(**kwargs)

    y_pred, y_true = torch.tensor(df.y_pred.values), torch.tensor(df.y_true.values)

    return GeneralMetrics(matthews(y_pred, y_true))


def ROC_curve(y_pred, y_true, group):
    data = []
    for threshold in range(0, 101, 2):
        metrics = get_general_metrics(y_pred, y_true, group, thresh=threshold / 100)
        data.append(("g = 0", metrics.fpr.group0, metrics.tpr.group0, threshold / 100))
        data.append(("g = 1", metrics.fpr.group1, metrics.tpr.group1, threshold / 100))

    df = pd.DataFrame(data, columns=["G", "fpr", "tpr", "thresh"])
    mid_df = df.query("thresh==0.5")

    kwargs = dict(
        by="G",
        x="fpr",
        y="tpr",
        s=5,
        hover_cols=["thresh"],
        xlabel="FPR = Pr(S=1 | G=g, T=0)",
        ylabel="TPR = Pr(S=1 | G=g, T=1)",
        width=250,
        height=250,
        legend="bottom_right",
        xticks=2,
        yticks=2,
        title="ROC Curve per group",
    )
    ROC_plot = df.hvplot(**kwargs) * df.hvplot.scatter(**kwargs)
    mids = mid_df.hvplot.scatter(**kwargs, size=100, marker="square")
    plot = ROC_plot * mids
    plot

    return plot


def MCC_Odds(y_pred, y_true, group):

    data = []
    for threshold in range(0, 101, 2):
        matthews = MatthewsCorrCoef(num_classes=2, task="binary", threshold=threshold / 100)
        metrics = get_general_metrics(y_pred, y_true, group, thresh=threshold / 100)
        mcc = matthews(torch.tensor(y_pred[:, 1]), torch.tensor(y_true))
        data.append(("g = 0", metrics.fpr.group0, metrics.tpr.group0, mcc, threshold / 100))
        data.append(("g = 1", metrics.fpr.group1, metrics.tpr.group1, mcc, threshold / 100))

    df = pd.DataFrame(data, columns=["G", "fpr", "tpr", "mcc", "thresh"])

    # Convert tensor values to floats
    df[["fpr", "tpr", "mcc"]] = df[["fpr", "tpr", "mcc"]].applymap(lambda x: x.item())

    # Split the dataframe into two groups
    df0 = df[df["G"] == "g = 0"]
    df1 = df[df["G"] == "g = 1"]

    # Rename columns for merging
    df0 = df0.rename(columns={"fpr": "fpr0", "tpr": "tpr0"})
    df1 = df1.rename(columns={"fpr": "fpr1", "tpr": "tpr1"})

    # Merge dataframes on thresh
    df_merged = pd.merge(df0, df1, on=["thresh", "mcc"])

    # Compute delta equalized odds
    df_merged["DEOdds"] = abs(df_merged["tpr0"] - df_merged["tpr1"]) + abs(df_merged["fpr0"] - df_merged["fpr1"])

    # Print df
    df_merged.to_csv("../../data/mcc_odds.csv", index=False)
    return df_merged


def get_all_metrics(model_output, true, group, log_mlflow_w_suffix=None):
    with torch.no_grad():
        y_prob = model_output.softmax(dim=1).detach().cpu()

    fairness_metrics = get_fairness_metrics(y_prob, true.cpu(), group.cpu())
    general_metrics = get_general_metrics(y_prob, true.cpu(), group.cpu())
    fairprune_metrics = FairPruneMetrics.from_general_metrics(general_metrics)
    if log_mlflow_w_suffix is not None:
        mlflow.log_metrics(
            {
                f"{key}{log_mlflow_w_suffix}": value.item()
                if getattr(value, "total", None) is None
                else value.total.item()
                for metrics in [fairness_metrics, general_metrics, fairprune_metrics]
                for key, value in zip(metrics._fields, metrics)
            }
        )
    return fairness_metrics, general_metrics, fairprune_metrics


def fairness_constraint_parato_curve():
    filters = {"params.checkpoint": BASE_RUN_ID}
    run_df = filter_mlflow_data(**filters)
    # df_metrics = df[[c for c in df.columns if "metrics" in c]]
    # df_metrics.columns = df_metrics.columns.str.replace("metrics.", "")
    # df_metrics

    client = mlflow.tracking.MlflowClient()
    metrics = [
        "val loss",
        "Δ_eq_odds_t0_dev",
        "Δ_eq_odds_t1_dev",
        "tpr_dev",
        "fpr_dev",
        "tnr_dev",
        "matthews_dev",
        "EOdd_sep_abs_dev",
    ]

    def get_metrics(run_id):
        return (
            pd.DataFrame({metric: client.get_metric_history(run_id, metric) for metric in metrics})
            .applymap(lambda x: x.value)
            .assign(run_id=run_id)
            .set_index("run_id", append=True)
            .stack()
            .rename_axis(["epoch", "run_id", "metric"])
            .rename("value")
            .reset_index()
        )

    metrics_history = pd.concat(map(get_metrics, run_df.run_id))

    run_id_to_fairness = run_df.set_index("run_id")["params.fairness"].to_dict()
    metrics_history["fairness"] = metrics_history.run_id.map(run_id_to_fairness)

    fair_col = "EOdd_sep_abs_dev"
    loss_col = "matthews_dev"
    pareto_frontier = metrics_df.sort_values(fair_col)
    x = pareto_frontier[loss_col]
    while not all(increasing := x > x.shift(1).fillna(0)):
        pareto_frontier = pareto_frontier[increasing]
        x = pareto_frontier[loss_col]
    next_point_same_value = pareto_frontier.assign(**{fair_col: pareto_frontier[fair_col].shift(-1)})
    pareto_frontier = pd.concat([pareto_frontier, next_point_same_value]).sort_values([loss_col, fair_col])

    kw = dict(x=fair_col, hover_cols=["fairness"], title="Pareto frontier", xlabel="Equalized odds", ylabel="MCC")
    pareto_line = pareto_frontier.hvplot(y=loss_col, c="red", **kw)

    metrics_df = metrics_history.pivot(index=["fairness", "epoch"], columns="metric", values="value").reset_index()
    possible_values_plot = metrics_df.hvplot.scatter(y="matthews_dev", s=5, **kw)
    return possible_values_plot * pareto_line


def fairness_plots(metrics_history):
    colorbar_kw = dict(
        colorbar=True,
        colorbar_position="bottom",
        clabel="λ (fairness multiplier)",
        cnorm="log",
        colorbar_opts={
            "height": 5,
            "ticker": FixedTicker(ticks=sorted(metrics_history.fairness.astype(float).unique())),
        },
    )

    minimums = (  # selected rows minimizing Δ_eq_odds_t0_dev
        metrics_history.pivot(index=["fairness", "epoch"], columns="metric", values="value")
        .reset_index()
        .groupby(["fairness"])
        .apply(lambda x: x[x["Δ_eq_odds_t0_dev"] == x["Δ_eq_odds_t0_dev"].min()])
        .reset_index(drop=True)
    )

    # sort as categorical
    minimums = minimums.astype({"fairness": float}).sort_values("fairness").astype({"fairness": str})
    minimums.columns = minimums.columns.str.replace("_dev", "")

    minimums["color"] = minimums.fairness.astype(float)
    kwargs = dict(x="Δ_eq_odds_t0", y="Δ_eq_odds_t1", hover_cols=["fairness"], width=250, height=200)
    fairness_metrics_and_multiplier = minimums.hvplot.line(**kwargs) * minimums.hvplot.scatter(
        **kwargs, c="color", title="λ fairness effect", yticks=3, xticks=3
    ).opts(
        clabel="λ (fairness multiplier)",
        cnorm="log",
        ylabel="Δ equalized odds (y=1)",
        xlabel="Δ equalized odds (y=0)",
        colorbar_opts={
            "ticker": FixedTicker(ticks=sorted(metrics_history.fairness.astype(float).unique())),
            "width": 5,
        },
    )
    # PLOT 1
    fairness_metrics_and_multiplier

    H = 75

    def plot_timeline_w_optimum(metric, optimum=np.argmin):
        assert any(metric == metrics_history.metric)
        loss_history = metrics_history.query("metric == @metric")
        epoch_mins = loss_history.groupby("fairness").apply(lambda df: df.iloc[df.agg(optimum)["value"]])
        kw = dict(x="epoch", y="value", ylabel=metric, height=H, yticks=FixedTicker(ticks=[0.12, 0.18]), legend=False)
        grey_lines = loss_history.hvplot(**kw, by="fairness", c="grey", line_width=0.5)
        # colored_scatter = loss_history.hvplot.scatter(**kw, c="color", cnorm="log", s=50, marker="dash")
        colored_minimums = epoch_mins.hvplot.scatter(**kw, c="color", cnorm="log", marker="x", s=150)
        return grey_lines * colored_minimums

    timeline_plots = (
        plot_timeline_w_optimum("val loss").opts(ylabel="ℒ (val)").opts(ylim=(0.6, 1), yticks=2)
        + plot_timeline_w_optimum("Δ_eq_odds_t0_dev").opts(ylabel="Δodds0")
        + plot_timeline_w_optimum("Δ_eq_odds_t1_dev").opts(ylabel="Δodds1")
    )
    timeline_plots.cols(1).opts(hv.opts.Scatter(colorbar=False, xaxis=None)).opts(
        shared_axes=False, title="Minimums for values of λ (fairness loss multipliers)", toolbar=None
    )

    timeline_plots[-1].opts(hv.opts.Scatter(**colorbar_kw, xaxis=True)).opts(height=H + 120)
    # PLOT 2
    timeline_plots


if __name__ == "__main__":
    model = load_model(BASE_RUN_ID)

    data, group, y_true = get_dataset("dev")

    with torch.no_grad():
        model.eval()
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

    ROC_curve(y_pred, y_true, group)
    MCC_Odds(y_pred, y_true, group)
