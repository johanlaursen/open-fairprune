from torch.utils.data import DataLoader

from open_fairprune.data_util import LoanDataset, load_model
from open_fairprune.train import metric

if __name__ == "__main__":
    model = load_model()

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

    data, group, y_true = next(iter(dev_loader))
    y_true = y_true.to("cuda")
    print(f"{y_true.unique(return_counts=True) = }")

    y_pred = model(data.to("cuda"))
    accuracy_score = (y_pred.round() == y_true).to(float).mean()
    print(f"Accuracy: {accuracy_score:.4f}")

    print(f'F1-score: {1 / metric(model(data.to("cuda")).squeeze(), y_true)}')
