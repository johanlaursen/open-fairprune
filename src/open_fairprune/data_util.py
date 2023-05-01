import open_fairprune as this
import mlflow
from contextlib import contextmanager
import time
import numpy as np
import pandas as pd
from typing import Literal
from pathlib import Path
import datetime as dt
from torch.utils.data.dataloader import Dataset, DataLoader

root = Path(this.__path__[0]).parents[1]
data_dir = root / "data"


@contextmanager
def timeit(msg: str) -> float:
    start = time.perf_counter()
    start_date = f"{dt.datetime.now():%H:%M:%S}"
    yield
    print(f"{start_date} Time: {msg} {time.perf_counter() - start:.3f} seconds")


def load_model(id: str):
    """E.g.: model = load_model("aa246d9d2106472492442ff362b1b143")"""
    return mlflow.pytorch.load_model(model_uri=f"file://{data_dir}/mlruns/0/{id}/artifacts/model")


class DriverDataset(Dataset):
    def __init__(
        self,
        split: Literal["train"] | Literal["dev"] | Literal["test"],
        returns=["data", "label"],
        transform=None,
        *,
        fuck_your_ram: int = 0,
    ):
        self.transform = transform
        self.returns = returns
        df = pd.read_csv(data_dir / "Train_Dataset.csv")
        df["Client_Gender"] = df.Client_Gender == "Male"

        cols_b4 = len(df.columns)
        df = df.select_dtypes(include=[np.number, bool]).astype(float)
        cols_after = len(df.columns)
        print(f"Dropped {cols_b4 - cols_after} columns")

        self.df = df

        # .query(f"subject in @subjects[@split]").drop(columns="img")

        self.fuck_your_ram = fuck_your_ram  # Load this many samples into memory
        self._gigacache = {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if (cached := self._gigacache.get(idx)) is not None:
            return cached

        row = self.df.iloc[idx]

        # Do callables, such that we do lazy loading
        returns = {
            "data": lambda row: row.values,
            "label": lambda row: row.Client_Gender,
        }

        output = [returns[key](row) for key in self.returns]
        if self.transform:
            output = self.transform(output)
        if idx < len(self._gigacache):
            self._gigacache[idx] = output
        return output


if __name__ == "__main__":
    from tqdm import tqdm

    BATCH_SIZE = 128
    with timeit("total"):
        with timeit("dataset"):
            segment_train_dataset = DriverDataset(
                split="train",
                returns=["data", "label"],
                fuck_your_ram=1_000_000,
            )

        with timeit("cache"):
            segment_train_dataset[0]

        with timeit("dataloader"):
            segment_train_dataloader = DataLoader(
                segment_train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                drop_last=True,
            )

        with timeit("iter dataloader"):
            iterator = iter(segment_train_dataloader)

            for *data, labels in tqdm(iterator):
                pass
