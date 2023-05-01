from open_fairprune.data_util import DATA_PATH, LoanDataset, load_model, timeit

if __name__ == "__main__":
    LoanDataset("dev", returns=["data", "group", "label"])
    model = load_model("")
