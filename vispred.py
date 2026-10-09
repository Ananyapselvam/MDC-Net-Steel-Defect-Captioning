
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from PIL import Image

from src.config import CFG
from src.dataloader import create_dataloaders
from src.model import MDCNet

CHECKPOINT = "best_mdc_net_single_box.pth"


def main():
    _, _, test_loader, vocab = create_dataloaders()

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=CFG.device,
        weights_only=False,
    )

    model = MDCNet(
        vocab_size=checkpoint["vocab_size"]
    ).to(CFG.device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    batch = next(iter(test_loader))

    image = batch["images"][0:1].to(CFG.device)
    true_box = batch["bboxes"][0].cpu().tolist()
    true_class = batch["labels"][0]
    image_id = batch["image_ids"][0]

    with torch.no_grad():
        _, class_logits, location_logits, pred_box = model.encoder(image)

    predicted_id = class_logits.argmax(dim=1).item()
    id_to_label = {
        value: key for key, value in CFG.label_to_id.items()
    }
    predicted_class = id_to_label[predicted_id]
    predicted_location = location_logits.argmax(dim=1).item()
    predicted_box = pred_box[0].cpu().tolist()

    # Convert normalized xyxy coordinates into display coordinates.
    display_image = image[0].cpu().permute(1, 2, 0).numpy()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    display_image = image[0].cpu() * std + mean
    display_image = display_image.permute(1, 2, 0).clamp(0, 1).numpy()

    height, width = display_image.shape[:2]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.imshow(display_image)

    # Ground-truth box: green
    x1, y1, x2, y2 = true_box
    ax.add_patch(patches.Rectangle(
        (x1 * width, y1 * height),
        (x2 - x1) * width,
        (y2 - y1) * height,
        linewidth=2, edgecolor="lime", facecolor="none",
        label="Ground truth",
    ))

    # Predicted box: red
    x1, y1, x2, y2 = predicted_box
    ax.add_patch(patches.Rectangle(
        (x1 * width, y1 * height),
        (x2 - x1) * width,
        (y2 - y1) * height,
        linewidth=2, edgecolor="red", facecolor="none",
        label="Model prediction",
    ))

    ax.set_title(
        f"Image: {image_id}\n"
        f"Actual: {true_class} | Predicted: {predicted_class}"
    )
    ax.legend()
    ax.axis("off")
    plt.tight_layout()

    output_path = Path("prediction_visualization.png")
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.show()

    print("Image ID:", image_id)
    print("Actual class:", true_class)
    print("Predicted class:", predicted_class)
    print("Predicted location ID:", predicted_location)
    print("Ground-truth box:", [round(v, 3) for v in true_box])
    print("Predicted box:", [round(v, 3) for v in predicted_box])
    print("Saved visualization:", output_path.resolve())


if __name__ == "__main__":
    main()
