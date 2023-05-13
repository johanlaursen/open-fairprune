import itertools

import holoviews as hv
import hvplot.pandas
import pandas as pd

from open_fairprune.data_util import CATEGORICAL, ORDINAL_COLUMNS, get_split_df

hv.extension("bokeh")

ORDINAL_COLUMNS = list(set(ORDINAL_COLUMNS) - {"Default", "Age_Days"} | {"G", "T"})

df = get_split_df("dev")

non_numeric_columns = list(set(df.columns) - set(ORDINAL_COLUMNS))
cat_df = df[non_numeric_columns].astype("category")
categorical_ids_df = pd.concat([series.cat.codes.rename(series.name) for _, series in cat_df.items()], axis=1)

numeric_df = pd.concat([df[ORDINAL_COLUMNS].astype(float), categorical_ids_df], axis=1)


def get_correlation_scatters(numeric_df, cols1, cols2):
    kwargs = {
        "width": 300,
        "height": 300,
        "xlim": (-0.1, 1),
        "ylim": (-0.1, 1),
        "legend_position": "bottom_left",
        "xticks": 1,
        "yticks": 1,
    }
    t_plots = []
    g_plots = []
    df_full = numeric_df.assign(**{x: f"{x} = " + numeric_df[x].astype(int).astype(str) for x in ["G", "T"]})
    for col1, col2 in itertools.product(cols1, cols2):
        df = df_full.dropna(subset=[col1, col2])
        t_plots.append(df.hvplot.scatter(x=col1, y=col2, c="T").opts(**kwargs, alpha=0.3))
        g_plots.append(df.hvplot.scatter(x=col1, y=col2, c="G").opts(**kwargs, alpha=0.3))
    plot = hv.Layout(t_plots + g_plots)
    plot
    return plot.cols(len(plot) // 2)


def get_group_conditional_prob_plots(target="G"):
    densities = {}
    for col in numeric_df[list(set(ORDINAL_COLUMNS) - set(CATEGORICAL))]:
        df = numeric_df.dropna(subset=[col])
        if col == target:
            continue
        densities[col] = df.hvplot.kde(y=col, by=target, bandwidth=0.01, width=400, height=200)

    for col in df[CATEGORICAL]:
        if col == target:
            continue
        if df[col].nunique() > 25:
            continue
        counts = df[[col, target]].value_counts()

        densities[col] = (counts / counts.groupby(target).sum()).hvplot.bar(
            ylabel=f"Pr({col} | {target})", title="", width=400, height=200
        )

    return densities


nonlinear_correlations = get_correlation_scatters(numeric_df, ["House_Own"], ["Score_Source_1"])
nonlinear_correlations


densities = get_group_conditional_prob_plots(target="G")
plot = hv.Layout(densities.values()).cols(1)

(
    # densities["G"]
    densities["T"]
    + densities["Client_Gender"]  # Not so big
    + densities["Client_Marital_Status"]  # Group 1 has way more NaNs, yet higher scores on average
)

(
    densities["Score_Source_3"]
    + densities["Score_Source_2"].opts(xlim=(0, 1))  # Not so big
    + densities["Score_Source_1"]  # Group 1 has way more NaNs, yet higher scores on average
    + densities["Phone_Change"]
    + densities["Homephone_Tag"]
    + densities["Client_Family_Members"]
    + densities["Credit_Amount"]
    + densities["House_Own"]
)

# Proxies
(densities["ID_Days"] + densities["Registration_Days"] + densities["Employed_Days"])

plot = get_group_conditional_prob_plots(target="G")["Score_Source_1"].opts(
    width=250, legend_position="top_left", ylabel="Pr(Score_Source_1 | G)"
) + get_group_conditional_prob_plots(target="T")["Score_Source_1"].opts(
    width=250, legend_position="top_right", ylabel="Pr(Score_Source_1 | T)"
)
plot
