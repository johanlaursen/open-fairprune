from sre_parse import CATEGORIES
from unicodedata import numeric

import holoviews as hv
import hvplot.pandas
import numpy as np
import pandas as pd
from click import FLOAT

from open_fairprune.data_util import (
    CATEGORICAL,
    FLOAT_COLUMNS,
    get_dataset,
    get_split_df,
)

hv.extension("bokeh")

FLOAT_COLUMNS = list(set(FLOAT_COLUMNS) - {"Default", "Age_Days"} | {"G", "T"})

df = get_split_df("dev")

# minority_size = min(df["G"].value_counts())
# (_, df0), (_, df1) = df.groupby("G")
# df = pd.concat([df0.sample(minority_size), df1.sample(minority_size)])

non_numeric_columns = list(set(df.columns) - set(FLOAT_COLUMNS))
cat_df = df[non_numeric_columns].astype("category")
categorical_ids_df = pd.concat([series.cat.codes.rename(series.name) for _, series in cat_df.items()], axis=1)

numeric_df = pd.concat([df[FLOAT_COLUMNS], categorical_ids_df], axis=1).fillna(-1)

# plot = hvplot.scatter_matrix(df, c="G", alpha=0.1)
# hv.save(plot, "correlation.html")


def get_correlation_scatters():
    plots = {}
    for col in numeric_df:
        if col == "G":
            continue
        scatter = numeric_df.hvplot.scatter(x=col, y="G", width=200, height=200, yaxis=None).opts(
            jitter=0.2, alpha=0.01
        )
        slope = hv.Slope.from_scatter(scatter)

        plots[col] = scatter * slope

    # hv.save(hv.Layout(plots.values()).cols(1), "group_correlation.html")
    return plots


def get_group_conditional_prob_plots():
    target = "G"
    densities = {}
    for col in numeric_df[list(set(FLOAT_COLUMNS) - set(CATEGORICAL))]:
        if col == target:
            continue
        densities[col] = numeric_df.hvplot.kde(y=col, by=target, width=400, height=200, yaxis=None)

    for col in numeric_df[CATEGORICAL]:
        if col == target:
            continue
        if numeric_df[col].nunique() > 25:
            continue
        counts = numeric_df[[col, target]].value_counts()

        densities[col] = (counts / counts.groupby(target).sum()).hvplot.bar(
            ylabel=f"Pr({col} | {target})", title="", width=400, height=200
        )

    return densities


def ROC_curve(clf):
    prob = pd.Series(clf.predict_proba(X_train[group])[:, 1], name="prob")
    T = pd.Series(y_train[group], name="T")
    G = pd.Series([bool(group_no)] * len(T), name="G")

    df = pd.concat(
        [get_group_df(group, group_no) for group_no, group in enumerate(train_group_masks)],
    )

    ROC_g = []
    ROC_G = []
    for threshold in range(5, 100, 5):
        df["S"] = df["prob"] * 100 > threshold
        # TP_G = Count(S, G, T) / Count(G, T)
        # FP_G = Count(S, G, ~T) / Count(G, ~T)
        TP_g = len(df.query("S and ~G and T")) / len(df.query("~G and T"))
        FP_g = len(df.query("S and ~G and ~T")) / len(df.query("~G and ~T"))
        ROC_g.append([TP_g, FP_g, threshold])
        TP_G = len(df.query("S and G and T")) / len(df.query("G and T"))
        FP_G = len(df.query("S and G and ~T")) / len(df.query("G and ~T"))
        ROC_G.append([TP_G, FP_G, threshold])

        if threshold == 50:
            g_mid = [[TP_g, FP_g, 50]]
            G_mid = [[TP_G, FP_G, 50]]

    kwargs = dict(
        y="0",
        x="1",
        hover_cols=["2"],
        xlabel="False Positive Rate = Pr(S=1 | G=g, T=0)",
        ylabel="True Positive Rate = Pr(S=1 | G=g, T=1)",
    )
    ROC_plot = pd.DataFrame(ROC_g).hvplot(**kwargs).relabel("G=1") * pd.DataFrame(ROC_G).hvplot(**kwargs).relabel("G=2")
    mids = pd.DataFrame(g_mid).hvplot.scatter(**kwargs) * pd.DataFrame(G_mid).hvplot.scatter(**kwargs)
    return ROC_plot * mids


plots = ROC_curve(clfs[0]).opts(title=str(clfs[0])) + ROC_curve(clfs[1]).opts(title=str(clfs[1]))
plots.cols(1)

densities = get_group_conditional_prob_plots()
plot = hv.Layout(densities.values()).cols(1)


# hv.save(hv.Layout(plots.values()).cols(1), "group_densities.html")
(
    densities["G"]
    # + densities["T"]
    + densities["Client_Gender"]  # Not so big
    + densities["Client_Marital_Status"]  # Group 1 has way more NaNs, yet higher scores on average
)

(
    densities["Score_Source_3"]
    + densities["Score_Source_2"]  # Not so big
    + densities["Score_Source_1"]  # Group 1 has way more NaNs, yet higher scores on average
    + densities["Phone_Change"]
    + densities["Homephone_Tag"]
    + densities["Client_Family_Members"]
    + densities["Credit_Amount"]
    + densities["House_Own"]
)

# Proxies
(densities["ID_Days"] + densities["Registration_Days"] + densities["Employed_Days"])
