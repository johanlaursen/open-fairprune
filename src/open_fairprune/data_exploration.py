import itertools
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

FILLNA = -0.2
df = get_split_df("dev")

minority_size = min(df["G"].value_counts())
(_, df0), (_, df1) = df.groupby("G")
df = pd.concat([df0.sample(minority_size), df1.sample(minority_size)])

non_numeric_columns = list(set(df.columns) - set(FLOAT_COLUMNS))
cat_df = df[non_numeric_columns].astype("category")
categorical_ids_df = pd.concat([series.cat.codes.rename(series.name) for _, series in cat_df.items()], axis=1)

numeric_df = pd.concat([df[FLOAT_COLUMNS].astype(float), categorical_ids_df], axis=1).fillna(FILLNA)

# plot = hvplot.scatter_matrix(df, c="G", alpha=0.1)
# hv.save(plot, "correlation.html")


def get_correlation_scatters(columns=[c for c in numeric_df if "Source" in c]):
    kwargs = {
        "width": 300,
        "height": 300,
        "xlim": (FILLNA - 0.1, 1),
        "ylim": (FILLNA - 0.1, 1),
        "legend_position": "bottom_left",
        "xticks": 1,
        "yticks": 1,
    }
    t_plots = []
    g_plots = []
    df = numeric_df.assign(**{x: f"{x} = " + numeric_df[x].astype(int).astype(str) for x in ["G", "T"]})
    for col1, col2 in itertools.combinations(columns, r=2):
        t_plots.append(df.hvplot.scatter(x=col1, y=col2, c="T").opts(**kwargs, alpha=0.3))
        g_plots.append(df.hvplot.scatter(x=col1, y=col2, c="G").opts(**kwargs, alpha=0.3))
    plot = hv.Layout(t_plots + g_plots).cols(3)
    plot
    return plot


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


nonlinear_correlations = get_correlation_scatters()

densities = get_group_conditional_prob_plots()
plot = hv.Layout(densities.values()).cols(1)
numeric_df


# hv.save(hv.Layout(plots.values()).cols(1), "group_densities.html")
(
    # densities["G"]
    densities["T"]
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
