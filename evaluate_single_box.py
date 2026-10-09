import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

from src.config import CFG
from src.dataloader import create_dataloaders
from src.model import MDCNet


CHECKPOINT_PATH = Path("best_mdc_net_single_box.pth")
OUTPUT_DIR = Path("evaluation_results")
OUTPUT_DIR.mkdir(exist_ok=True)


def decode_tokens(tokens, vocab):
    words = []
    for token in tokens:
        token = int(token)
        if token == CFG.eos_idx:
            break
        if token in (CFG.pad_idx, CFG.sos_idx):
            continue
        words.append(vocab.itos.get(token, "<UNK>"))
    return " ".join(words)


def generate_caption(model, image, vocab):
    """Generate a caption using image features only; do not pass the true box."""
    with torch.no_grad():
        memory, _, _, _ = model.encoder(image)
        generated = [CFG.sos_idx]

        for _ in range(CFG.max_len):
            current_tokens = torch.tensor(
                [generated], dtype=torch.long, device=CFG.device
            )
            decoder_output = model.decoder(current_tokens, memory)
            next_token = int(decoder_output[0, -1].argmax().item())
            generated.append(next_token)
            if next_token == CFG.eos_idx:
                break

    return decode_tokens(generated, vocab)


def box_iou_xyxy(box_a, box_b):
    """IoU for normalized xyxy boxes."""
    ax1, ay1, ax2, ay2 = [float(x) for x in box_a]
    bx1, by1, bx2, by2 = [float(x) for x in box_b]

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def main():
    print("Device:", CFG.device)
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH.resolve()}\n"
            "Place this script in the project root and check the checkpoint filename."
        )

    _, _, test_loader, vocab = create_dataloaders()
    print("Test images:", len(test_loader.dataset))
    print("Vocabulary size:", len(vocab))

    checkpoint = torch.load(
        CHECKPOINT_PATH, map_location=CFG.device, weights_only=False
    )
    model = MDCNet(vocab_size=checkpoint.get("vocab_size", len(vocab))).to(CFG.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("Loaded checkpoint:", CHECKPOINT_PATH)
    print("Checkpoint epoch:", checkpoint.get("epoch"))
    print("Checkpoint validation loss:", checkpoint.get("val_loss"))

    id_to_label = {idx: label for label, idx in CFG.label_to_id.items()}
    class_names = [id_to_label[i] for i in range(len(id_to_label))]

    y_true, y_pred, y_score = [], [], []
    ious, captions = [], []
    total_locations = 0
    correct_locations = 0

    with torch.no_grad():
        for batch in test_loader:
            images = batch["images"].to(CFG.device)
            gt_boxes = batch["bboxes"].to(CFG.device)
            location_ids = batch["location_ids"].to(CFG.device)
            labels = batch["labels"]

            # The model predicts from images alone. Ground-truth boxes are
            # used only after prediction, for IoU calculation.
            _, class_logits, location_logits, pred_boxes = model.encoder(images)
            probabilities = torch.softmax(class_logits, dim=1)
            predicted_ids = class_logits.argmax(dim=1).cpu().tolist()

            for i, predicted_id in enumerate(predicted_ids):
                actual_label = labels[i]
                actual_id = CFG.label_to_id[actual_label]
                y_true.append(actual_id)
                y_pred.append(predicted_id)
                y_score.append(probabilities[i].cpu().numpy())

                ious.append(
                    box_iou_xyxy(
                        pred_boxes[i].detach().cpu().tolist(),
                        gt_boxes[i].detach().cpu().tolist(),
                    )
                )

                actual_loc = int(location_ids[i].item())
                predicted_loc = int(location_logits[i].argmax(dim=0).item())
                correct_locations += int(actual_loc == predicted_loc)
                total_locations += 1

                if len(captions) < 20:
                    caption = generate_caption(model, images[i:i+1], vocab)
                    captions.append(
                        {
                            "image_id": str(batch["image_ids"][i]),
                            "actual_class": actual_label,
                            "predicted_class": id_to_label[predicted_id],
                            "actual_bbox_xyxy_normalized": [
                                round(float(x), 4) for x in gt_boxes[i].cpu().tolist()
                            ],
                            "predicted_bbox_xyxy_normalized": [
                                round(float(x), 4)
                                for x in pred_boxes[i].detach().cpu().tolist()
                            ],
                            "IoU": round(ious[-1], 4),
                            "generated_caption": caption,
                        }
                    )

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_score = np.asarray(y_score)

    if len(y_true) == 0:
        raise RuntimeError("The test loader returned no samples.")

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    print("\n" + "=" * 65)
    print("TEST SET RESULTS")
    print("=" * 65)
    print(f"Test images: {len(y_true)}")
    print(f"Classification accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Weighted precision:       {precision:.4f}")
    print(f"Weighted recall:          {recall:.4f}")
    print(f"Weighted F1-score:        {f1:.4f}")
    print(f"Location accuracy:        {correct_locations / total_locations:.4f}")
    print(f"Mean bounding-box IoU:    {float(np.mean(ious)):.4f}")
    print("\nPer-class report:")
    print(
        classification_report(
            y_true, y_pred, labels=list(range(len(class_names))),
            target_names=class_names, zero_division=0
        )
    )

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(9, 7))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    display.plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title("MDC-Net Test Confusion Matrix")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    # One-vs-rest ROC/AUC for multiclass defect classification.
    # Skip classes absent from the test labels, since their ROC is undefined.
    y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(8, 6))
    auc_values = {}
    plotted = 0
    for class_id, class_name in enumerate(class_names):
        positives = y_true_bin[:, class_id]
        if positives.min() == positives.max():
            continue
        fpr, tpr, _ = roc_curve(positives, y_score[:, class_id])
        class_auc = auc(fpr, tpr)
        auc_values[class_name] = float(class_auc)
        ax.plot(fpr, tpr, label=f"{class_name} (AUC={class_auc:.3f})")
        plotted += 1

    if plotted:
        ax.plot([0, 1], [0, 1], linestyle="--", label="Chance")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("One-vs-Rest ROC Curves — Test Set")
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "roc_curves.png", dpi=200)
        try:
            print(f"Macro ROC-AUC: {roc_auc_score(y_true, y_score, multi_class='ovr', average='macro', labels=list(range(len(class_names)))):.4f}")
        except ValueError:
            print("Macro ROC-AUC unavailable for this test-label distribution.")
    else:
        print("ROC curves unavailable: test set does not contain both positive and negative samples for any class.")
    plt.close(fig)

    results = {
        "checkpoint": str(CHECKPOINT_PATH),
        "test_images": int(len(y_true)),
        "classification_accuracy": float(accuracy),
        "weighted_precision": float(precision),
        "weighted_recall": float(recall),
        "weighted_f1": float(f1),
        "location_accuracy": float(correct_locations / total_locations),
        "mean_bbox_iou": float(np.mean(ious)),
        "class_names": class_names,
        "confusion_matrix": cm.tolist(),
        "per_class_roc_auc": auc_values,
        "sample_predictions": captions,
    }
    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("\nSaved outputs in:", OUTPUT_DIR.resolve())
    print("- metrics.json")
    print("- confusion_matrix.png")
    if (OUTPUT_DIR / "roc_curves.png").exists():
        print("- roc_curves.png")
    print("- sample_predictions are included in metrics.json")


if __name__ == "__main__":
    main()
