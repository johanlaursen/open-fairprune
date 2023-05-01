from torch.utils.data import DataLoader

from open_fairprune.data_util import DATA_PATH, LoanDataset, load_model, timeit

if __name__ == "__main__":
    model = load_model("c9c6956dba1b483a888798b2eb37a0b2")

    dataset = LoanDataset("dev", returns=["data", "group", "label"])

    data_kwargs = {
        # "num_workers": 4,
        "pin_memory": True,
        "shuffle": True,  # Only done once for entire run
        "batch_size": len(dataset),
        "drop_last": True,  # Drop last batch if it's not full
    }

    dev_loader = DataLoader(
        dataset,
        **data_kwargs,
    )

    data, group, label = next(iter(dev_loader))
    pred = model(data.to("cuda"))
    accuracy_score = (pred.argmax(dim=1) == label.to("cuda")).to(float).mean()
    print(f"Accuracy: {accuracy_score:.4f}")
