import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

    true_labels = []
    predicted_labels = []

    print("Running test evaluation...")

    with torch.no_grad():

        for batch in test_loader:

            images = batch["images"].to(CFG.device)
            bboxes = batch["bboxes"].to(CFG.device)

            _, class_logits, _ = model.encoder(
                images,
                bboxes
            )

            predictions = class_logits.argmax(
                dim=1
            ).cpu().tolist()

            actual = [
                CFG.label_to_id[label]
                for label in batch["labels"]
            ]

            true_labels.extend(actual)
            predicted_labels.extend(predictions)

    # --------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------

    class_names = list(
        CFG.label_to_id.keys()
    )

    cm = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=list(range(CFG.num_classes))
    )

    print("\nConfusion Matrix:")
    print(cm)

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    display.plot(
        ax=ax,
        xticks_rotation=45
    )

    plt.title(
        "MDC-Net Defect Classification Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        "confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(
        "\nSaved as: confusion_matrix.png"
    )


if __name__ == "__main__":
    main()