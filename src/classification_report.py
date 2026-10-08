import torch
from sklearn.metrics import classification_report

from src.config import CFG
from src.dataloader import create_dataloaders
from src.model import MDCNet


def main():

    print("Loading test data...")

    _, _, test_loader, _ = create_dataloaders()

    model = MDCNet().to(CFG.device)

    checkpoint = torch.load(
        "best_mdc_net.pth",
        map_location=CFG.device
    )

    model.load_state_dict(checkpoint)
    model.eval()

    print("Model loaded successfully.")
    print("Running classification evaluation...")

    true_labels = []
    predicted_labels = []

    with torch.no_grad():

        for batch in test_loader:

            images = batch["images"].to(
                CFG.device
            )

            bboxes = batch["bboxes"].to(
                CFG.device
            )

            _, class_logits, _ = model.encoder(
                images,
                bboxes
            )

            predictions = (
                class_logits
                .argmax(dim=1)
                .cpu()
                .tolist()
            )

            actual = [
                CFG.label_to_id[label]
                for label in batch["labels"]
            ]

            true_labels.extend(actual)
            predicted_labels.extend(predictions)

    class_names = list(
        CFG.label_to_id.keys()
    )

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=list(range(CFG.num_classes)),
        target_names=class_names,
        digits=4
    )

    print("\n" + "=" * 70)
    print("DEFECT CLASSIFICATION REPORT")
    print("=" * 70)

    print(report)

    # Save report
    with open(
        "results/classification_report.txt",
        "w"
    ) as f:

        f.write(
            "MDC-Net Defect Classification Report\n"
        )

        f.write("=" * 70 + "\n\n")

        f.write(report)

    print(
        "Saved to: results/classification_report.txt"
    )


if __name__ == "__main__":
    main()