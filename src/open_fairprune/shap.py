import numpy as np
import shap
import torch

from open_fairprune.data_util import get_dataset, load_model

shap.initjs()

model = load_model()
X_train, g_train, y_train = [np.array(x) for x in get_dataset("train")]
X_dev, g_dev, y_dev = [np.array(x) for x in get_dataset("dev")]

f = lambda x: model(torch.autograd.Variable(torch.from_numpy(x).to("cuda"))).softmax(dim=1).detach().cpu().numpy()

shap_train = shap.kmeans(X_train, 100)

explainer = shap.KernelExplainer(f, shap_train)

shap_values = explainer.shap_values(X_dev[:500, :], nsamples=500)

shap.summary_plot(shap_values[0], X_dev[:500, :])
shap.summary_plot(shap_values[1], X_dev[:500, :])


shap.force_plot(explainer.expected_value[0], shap_values[0])
shap.dependence_plot(shap_values[0])

shap.dependence_plot(0, shap_values[0], X_dev[:50, :])
shap.decision_plot(explainer.expected_value[0], shap_values[0], X_dev[:50, :])

shap_values[0]

# shap.__version__

# shap.KernelExplainer
