import datetime as dt
import subprocess
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data.dataloader import DataLoader, Dataset

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

import open_fairprune as this

root = Path(this.__path__[0]).parents[1]
DATA_PATH = root / "data"

INPUT_SIZE = 186


ORDINAL_COLUMNS = [
    "Age_Days",
    "Child_Count",
    "Cleint_City_Rating",
    "Client_Family_Members",
    "Client_Income",
    "Credit_Amount",
    "Employed_Days",
    "ID",
    "ID_Days",
    "Loan_Annuity",
    "Own_House_Age",
    "Phone_Change",
    "Population_Region_Relative",
    "Registration_Days",
    "Score_Source_1",
    "Score_Source_2",
    "Score_Source_3",
    "Social_Circle_Default",
]


CATEGORICAL = [
    "Accompany_Client",
    "Active_Loan",
    "Application_Process_Day",
    "Application_Process_Hour",
    "Bike_Owned",
    "Car_Owned",
    "Child_Count",
    "Cleint_City_Rating",
    "Client_Contact_Work_Tag",
    "Client_Education",
    "Client_Gender",
    "Client_Housing_Type",
    "Client_Income_Type",
    "Client_Marital_Status",
    "Client_Occupation",
    "Client_Permanent_Match_Tag",
    "Credit_Bureau",
    "G",
    "Homephone_Tag",
    "House_Own",
    "Loan_Contract_Type",
    "Mobile_Tag",
    "T",
    "Type_Organization",
    "Workphone_Working",
]


@contextmanager
def timeit(msg: str) -> float:
    start = time.perf_counter()
    start_date = f"{dt.datetime.now():%H:%M:%S}"
    yield
    print(f"{start_date} Time: {msg} {time.perf_counter() - start:.3f} seconds")


def filter_mlflow_data(**filters) -> pd.DataFrame:
    runs = mlflow.search_runs()
    for k, v in filters.items():
        runs = runs[runs[k] == v]
    return runs


def load_model(id: str = "latest", model_name: str = "model", return_run_id=False):
    """E.g.: model = load_model("aa246d9d2106472492442ff362b1b143")"""
    if id == "latest":
        runs = mlflow.search_runs()
        latest_model = runs.iloc[runs.end_time.argmax()]
        print("Getting model from: UTC", latest_model.end_time)
        id = latest_model.run_id

    if return_run_id:
        return mlflow.pytorch.load_model(model_uri=f"file://{DATA_PATH}/mlruns/0/{id}/artifacts/{model_name}"), id
    else:
        return mlflow.pytorch.load_model(model_uri=f"file://{DATA_PATH}/mlruns/0/{id}/artifacts/{model_name}")


def get_split_df(split="train"):
    df = pd.read_csv(DATA_PATH / "Train_Dataset.csv", low_memory=False)
    splits = {
        "train": df.ID % 7 <= 4,
        "dev": df.ID % 7 == 5,
        "test": df.ID % 7 == 6,  # Around 15%
    }
    df = df[splits[split]]

    def isfloat(x):
        try:
            float(x)
            return True
        except:
            return False

    df.loc[:, ORDINAL_COLUMNS] = df[ORDINAL_COLUMNS][df[ORDINAL_COLUMNS].applymap(isfloat)]
    df = df.astype({c: "float" for c in ORDINAL_COLUMNS})

    df = df.dropna(subset="Age_Days")  # NOTE: Drops where we dont have label!
    df["T"] = df.Default.astype(bool)
    df["G"] = df.Age_Days.astype(int) // 365 > 43
    df = df.drop(columns=["Default", "Age_Days"])
    return df


def export_data():
    train_df = get_split_df("train")
    train_target = torch.tensor(train_df["T"].to_numpy(), dtype=torch.long)
    train_group = torch.tensor(train_df.G.to_numpy(), dtype=torch.long)
    train_df = train_df.drop(columns=["T", "G"])

    # Dropping columns with more than 50% missing values (except score sources)
    train_df = train_df.drop(["Own_House_Age", "Social_Circle_Default"], axis=1)

    # train imputers
    ordinal = list(set(ORDINAL_COLUMNS) - {"Age_Days", "Own_House_Age", "Social_Circle_Default"})
    number_imputer = SimpleImputer(strategy="median").set_output(transform="pandas").fit(train_df[ordinal])

    categorical = list(set(CATEGORICAL) - {"G", "T"})
    category_imputer = SimpleImputer(strategy="most_frequent").set_output(transform="pandas").fit(train_df[categorical])
    category_1hot = (
        OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop="first")
        .set_output(transform="pandas")
        .fit(category_imputer.transform(train_df[categorical]))
    )

    def impute(df):
        return pd.concat(
            [
                number_imputer.transform(df[ordinal]),
                category_1hot.transform(category_imputer.transform(df[categorical])),
            ],
            axis=1,
        ).astype("float32")

    train_df_1hot = impute(train_df)
    scaler = StandardScaler().set_output(transform="pandas").fit(train_df_1hot)
    scaled_train_df = scaler.transform(train_df_1hot)

    dfs = [(scaled_train_df, train_target, train_group)]
    for df in [get_split_df("dev"), get_split_df("test")]:
        target = torch.tensor(df["T"].to_numpy(), dtype=torch.long)
        group = torch.tensor(df.G.to_numpy(), dtype=torch.long)

        df = df.drop(["Own_House_Age", "Social_Circle_Default"], axis=1)
        df = df.drop(columns=["T", "G"])

        df_1hot = impute(df)
        scaled_df = scaler.transform(df_1hot)

        assert len(scaled_df.columns) == INPUT_SIZE, f"{INPUT_SIZE = } != {len(scaled_df.columns) = }"
        dfs.append((scaled_df, target, group))

    torch.save(scaled_train_df.columns, DATA_PATH / f"data.columns.pt")
    for split, (data, target, group) in zip(["train", "dev", "test"], dfs):
        torch.save(torch.from_numpy(data.to_numpy()), DATA_PATH / f"{split}_data.pt")
        torch.save(group, DATA_PATH / f"{split}_group.pt")
        torch.save(target, DATA_PATH / f"{split}_label.pt")


def get_dataset(split: str, returns=["data", "group", "label"]):
    return [torch.load(open(DATA_PATH / f"{split}_{return_}.pt", "rb")) for return_ in returns]


def get_git_hash():
    return (
        subprocess.run(["git", "log", "-1", "--pretty=format:%H"], check=True, stdout=subprocess.PIPE)
        .stdout.decode("utf-8")
        .strip()
    )


if __name__ == "__main__":
    with timeit("get_data"):
        features, group, labels = get_dataset("train")

    print(features.shape)
    print(group.shape)
    print(labels.shape)

    def sample_df(df):
        return df.sample(10_000, replace=True, random_state=42)

    equal_df = pd.DataFrame(np.hstack([labels[:, None], features])).groupby(0).apply(sample_df)
    equal_X, equal_y = equal_df.iloc[:, 1:].values, equal_df.iloc[:, 0].values
    assert equal_X.shape == (20_000, INPUT_SIZE)
    assert equal_y.shape == (20_000,)
    assert sum(equal_y) == 10_000
