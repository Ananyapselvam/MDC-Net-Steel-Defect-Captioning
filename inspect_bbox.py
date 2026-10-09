import torch

from src.config import CFG
from src.dataloader import create_dataloaders
from src.model import MDCNet


def iou_xyxy(a, b, eps=1e-7):
    ix1 = max(float(a[0]), float(b[0]))
    iy1 = max(float(a[1]), float(b[1]))
    ix2 = min(float(a[2]), float(b[2]))
    iy2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(
        0.0, float(a[3]) - float(a[1])
    )
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(
        0.0, float(b[3]) - float(b[1])
    )
    return inter / (area_a + area_b - inter + eps)


def main():
    CFG.batch_size = 1
    _train_loader, val_loader, _test_loader, vocab = create_dataloaders()

    checkpoint_path = "best_mdc_net_bbox.pth"
    checkpoint = torch.load(
        checkpoint_path, map_location=CFG.device, weights_only=False
    )

    model = MDCNet(vocab_size=len(vocab)).to(CFG.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Checkpoint:", checkpoint_path)
    print("Saved epoch:", checkpoint.get("epoch", "unknown"))
    print("Validation samples inspected: up to 12")
    print("-" * 90)

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 12:
                break

            images = batch["images"].to(CFG.device)
            captions = batch["captions"].to(CFG.device)
            gt_box = batch["bboxes"][0].cpu().tolist()

            _caption_logits, class_logits, location_logits, pred_boxes = model(
                images, captions[:, :-1]
            )

            pred_box = pred_boxes[0].cpu().tolist()
            iou = iou_xyxy(pred_box, gt_box)
            pred_class_id = int(class_logits.argmax(dim=1).item())
            pred_location_id = int(location_logits.argmax(dim=1).item())
            true_label = str(batch["labels"][0])
            image_id = str(batch["image_ids"][0])

            print(f"Sample {i + 1} | image_id={image_id} | true class={true_label}")
            print(
                "  GT box   [x1,y1,x2,y2]: "
                + str([round(v, 3) for v in gt_box])
            )
            print(
                "  Pred box [x1,y1,x2,y2]: "
                + str([round(v, 3) for v in pred_box])
            )
            print(
                f"  IoU={iou:.4f} | predicted class ID={pred_class_id} "
                f"| predicted location ID={pred_location_id}"
            )

    print("-" * 90)
    print("This is a diagnostic sample, not a full evaluation metric.")


if __name__ == "__main__":
    main()
