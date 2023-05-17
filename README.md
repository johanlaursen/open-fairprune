# open-fairprune
An open implementation of [FairPrune](https://arxiv.org/abs/2203.02110), a method for pruning parameters as to increase the fairness between two groups

# Install
Run the install.bat script. For Linux, examine the script and run the equivalent commands. This creates a venv folder, which you should use to run the files.

You'll get 2 additional commands
* train-prune --help (Train a model, possibly using --fairness LAMBDA_VALUE)
* mlflow ui (Run in the [data](./data/) folder)

# Applying fairprune
Examine the main section in [fairprune.py](.\src\open_fairprune\fairprune.py), and modify hyperparameters, model loading etc. to your needs.

# Running our models
The data directory already contains some runs. Upon loading these, you'll get an error stating it cannot find the path C/USERNAME. You therefore need to change the [artifact_uri](.\data\mlruns\0\08a5ecfcb09b4ee9a9eaf8a1065198e0\meta.yaml)

# Running our plots
Run the [eval_model.py](.\src\open_fairprune\eval_model.py)

# Running shap
Examine and run the [shap_explainer.py](.\src\open_fairprune\shap_explainer.py) (takes a while)