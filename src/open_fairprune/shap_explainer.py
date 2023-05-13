import numpy as np
import pandas as pd
import shap
import torch

from open_fairprune.data_util import DATA_PATH, get_dataset, load_model

shap.initjs()

BASE_RUN_ID = "7b9c67bcf82b40328baf2294df5bd1a6"
# FAIR_RUN_ID = "35be9bf261e5470584f44e386441411e"
model = load_model(BASE_RUN_ID)
X_train, g_train, y_train = [np.array(x) for x in get_dataset("train")]
X_dev, g_dev, y_dev = [np.array(x) for x in get_dataset("dev")]

shap_train = shap.kmeans(X_train, 100)

f = lambda x: model(torch.autograd.Variable(torch.from_numpy(x).to("cuda"))).softmax(dim=1).detach().cpu().numpy()


explainer = shap.KernelExplainer(f, shap_train)

feature_names = torch.load(open(DATA_PATH / f"data.columns.pt", "rb")).to_list()

kw = dict(feature_names=feature_names)
to_explain = X_dev[:500, :]
shap_values = explainer.shap_values(to_explain, nsamples=500)

age_proxy = ["ID_Days", "Registration_Days", "Employed_Days", "Score_Source_1"]
proxy_mask = pd.Series(feature_names).isin(age_proxy)

age_proxy_plot = shap.summary_plot(
    shap_values[1][:, proxy_mask], to_explain[:, proxy_mask], feature_names=age_proxy, plot_size=(5, 5)
)
age_proxy_plot

# shap.force_plot(explainer.expected_value[0], shap_values[0], **kw)

# shap.dependence_plot("Score_Source_1", shap_values[1], to_explain, feature_names)
# shap.decision_plot(explainer.expected_value[0], shap_values[0], to_explain)
