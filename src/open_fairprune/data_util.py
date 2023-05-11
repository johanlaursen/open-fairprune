import datetime as dt
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import mlflow
import numpy as np
import pandas as pd
import torch
from click import FLOAT
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data.dataloader import DataLoader, Dataset

import open_fairprune as this

root = Path(this.__path__[0]).parents[1]
DATA_PATH = root / "data"

INPUT_SIZE = 29


FLOAT_COLUMNS = [
    "Active_Loan",
    "Age_Days",
    "Application_Process_Day",
    "Application_Process_Hour",
    "Bike_Owned",
    "Car_Owned",
    "Child_Count",
    "Cleint_City_Rating",
    "Client_Family_Members",
    "Client_Income",
    "Credit_Amount",
    "Credit_Bureau",
    "Default",
    "Employed_Days",
    "Homephone_Tag",
    "House_Own",
    "ID",
    "ID_Days",
    "Loan_Annuity",
    "Mobile_Tag",
    "Own_House_Age",
    "Phone_Change",
    "Population_Region_Relative",
    "Registration_Days",
    "Score_Source_1",
    "Score_Source_2",
    "Score_Source_3",
    "Social_Circle_Default",
    "Workphone_Working",
]

CATEGORICAL = [
    "Accompany_Client",
    "Active_Loan",
    "Bike_Owned",
    "Car_Owned",
    "Child_Count",
    "Cleint_City_Rating",
    "Client_Contact_Work_Tag",
    "Client_Education",
    "Client_Family_Members",
    "Client_Gender",
    "Client_Housing_Type",
    "Client_Income_Type",
    "Client_Marital_Status",
    "Client_Occupation",
    "Client_Permanent_Match_Tag",
    "Homephone_Tag",
    "Homephone_Tag",
    "House_Own",
    "Loan_Contract_Type",
    "Population_Region_Relative",
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
    df = pd.read_csv(DATA_PATH / "Train_Dataset.csv")
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

    df.loc[:, FLOAT_COLUMNS] = df[FLOAT_COLUMNS][df[FLOAT_COLUMNS].applymap(isfloat)]
    df = df.astype({c: "float" for c in FLOAT_COLUMNS})

    df = df.dropna(subset="Age_Days")  # NOTE: Drops where we dont have label!
    df["T"] = df.Default.astype(bool)
    df["G"] = df.Age_Days.astype(int) // 365 > 43
    df = df.drop(columns=["Default", "Age_Days"])
    return df


class LoanDataset(Dataset):
    def __init__(
        self,
        split: Literal["train"] | Literal["dev"] | Literal["test"],
        returns=["data", "group", "label"],
        transform=None,
        *,
        fuck_your_ram: int = 0,
    ):
        self.transform = transform
        self.returns = returns
        df = get_split_df(split)

        # df.Default.value_counts()  # 0=80009, 1=7031, 11:1 ratio
        # df.Client_Gender.value_counts()  # Male=56070, Female=29263, XNA=2, M:F ratio=2:1

        # gb = df.groupby("Default")
        # g0, g1 = gb.get_group(0), gb.get_group(1)
        # g1 = g1.sample(len(g0), replace=True)
        # df = pd.concat([g0.iloc[:8000], g1])

        self.target = torch.tensor(df["T"].to_numpy(), dtype=torch.long)
        self.group = torch.tensor(df.G.to_numpy(), dtype=torch.long)

        cols_b4 = len(df.columns)
        df = df.select_dtypes(include=[np.number, bool]).astype(float)
        cols_after = len(df.columns)
        print(f"Dropped {cols_b4 - cols_after} columns: {df.shape = }")

        pipe = make_pipeline(SimpleImputer(), StandardScaler()).set_output(transform="pandas")

        df = pipe.fit_transform(df)
        df = df.astype("float32")
        self.df = df

        self.fuck_your_ram = fuck_your_ram  # Load this many samples into memory
        self._gigacache = {}
        assert len(self.df.columns) == INPUT_SIZE, f"Please update training data feature size: {INPUT_SIZE = }"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if (cached := self._gigacache.get(idx)) is not None:
            return cached

        row = self.df.iloc[idx]

        # Do callables, such that we do lazy loading
        returns = {
            "data": lambda row: row.values,
            "group": lambda row: self.group[idx],
            "label": lambda row: self.target[idx],
        }

        output = [returns[key](row) for key in self.returns]
        if self.transform:
            output = self.transform(output)
        if idx < len(self._gigacache):
            self._gigacache[idx] = output
        return output


def get_dataset(split: str, returns=["data", "group", "label"]):
    dataset = LoanDataset(split, returns=returns)
    loader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=len(dataset),
        drop_last=True,
    )
    return next(iter(loader))


if __name__ == "__main__":
    with timeit("get data"):
        features, group, labels = get_dataset("train")

    print(features.shape)
    print(group.shape)
    print(labels.shape)
