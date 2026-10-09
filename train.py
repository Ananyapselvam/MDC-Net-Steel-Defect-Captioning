import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from src.config import CFG
from src.dataloader import create_dataloaders
from src.model import MDCNet


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def class_targets_from_labels(labels):
    """Dataset returns class names as strings; map them to class IDs."""
    unknown = sorted({str(label) for label in labels} - set(CFG.label_to_id))
    if unknown:
        raise ValueError(
            f"Unknown labels in batch: {unknown}. "
            f"Expected one of: {list(CFG.label_to_id)}"
        )
    return torch.tensor(
        [CFG.label_to_id[str(label)] for label in labels],
        dtype=torch.long,
        device=CFG.device,
    )


def box_iou(pred, target, eps=1e-7):
    """Mean IoU inputs are normalized xyxy boxes: [xmin, ymin, xmax, ymax]."""
    ix1 = torch.maximum(pred[:, 0], target[:, 0])
    iy1 = torch.maximum(pred[:, 1], target[:, 1])
    ix2 = torch.minimum(pred[:, 2], target[:, 2])
    iy2 = torch.minimum(pred[:, 3], target[:, 3])
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)

    area_p = (pred[:, 2] - pred[:, 0]).clamp(min=0) * (
        pred[:, 3] - pred[:, 1]
    ).clamp(min=0)
    area_t = (target[:, 2] - target[:, 0]).clamp(min=0) * (
        target[:, 3] - target[:, 1]
    ).clamp(min=0)
    return inter / (area_p + area_t - inter + eps)


def run_epoch(model, loader, optimizer=None):
    training = optimizer is not None
    model.train(training)

    ce_caption = nn.CrossEntropyLoss(ignore_index=CFG.pad_idx)
    ce_class = nn.CrossEntropyLoss()
    ce_location = nn.CrossEntropyLoss()
    bbox_loss_fn = nn.SmoothL1Loss()

    totals = {
        "loss": 0.0, "caption": 0.0, "class": 0.0, "location": 0.0,
        "bbox": 0.0, "class_correct": 0, "location_correct": 0,
        "samples": 0, "iou": 0.0,
    }
    start = time.perf_counter()

    grad_context = torch.enable_grad() if training else torch.no_grad()
    with grad_context:
        for batch in loader:
            if training and totals["samples"] % 50 == 0:
                print(
                    f"Training progress: {totals['samples']}/{len(loader.dataset)}",
                    flush=True
                )
            images = batch["images"].to(CFG.device)
            captions = batch["captions"].to(CFG.device)
            bboxes = batch["bboxes"].to(CFG.device).clamp(0.0, 1.0)
            location_ids = batch["location_ids"].to(CFG.device)
            class_targets = class_targets_from_labels(batch["labels"])

            # Teacher forcing: decoder receives caption tokens except the last;
            # targets are the same captions shifted by one token.
            caption_in = captions[:, :-1]
            caption_target = captions[:, 1:]

            if training:
                optimizer.zero_grad(set_to_none=True)

            caption_logits, class_logits, location_logits, bbox_pred = model(
                images, caption_in
            )

            caption_loss = ce_caption(
                caption_logits.reshape(-1, caption_logits.size(-1)),
                caption_target.reshape(-1),
            )
            class_loss = ce_class(class_logits, class_targets)
            location_loss = ce_location(location_logits, location_ids)
            bbox_loss = bbox_loss_fn(bbox_pred, bboxes)

            loss = (
                caption_loss
                + 0.5 * class_loss
                + 0.1 * location_loss
                + 5.0 * bbox_loss
            )

            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            batch_size = images.size(0)
            totals["loss"] += loss.item() * batch_size
            totals["caption"] += caption_loss.item() * batch_size
            totals["class"] += class_loss.item() * batch_size
            totals["location"] += location_loss.item() * batch_size
            totals["bbox"] += bbox_loss.item() * batch_size
            totals["class_correct"] += (
                class_logits.argmax(1) == class_targets
            ).sum().item()
            totals["location_correct"] += (
                location_logits.argmax(1) == location_ids
            ).sum().item()
            totals["samples"] += batch_size
            totals["iou"] += box_iou(bbox_pred.detach(), bboxes).sum().item()

    n = max(totals["samples"], 1)
    return {
        "loss": totals["loss"] / n,
        "caption_loss": totals["caption"] / n,
        "class_loss": totals["class"] / n,
        "location_loss": totals["location"] / n,
        "bbox_loss": totals["bbox"] / n,
        "class_accuracy": totals["class_correct"] / n,
        "location_accuracy": totals["location_correct"] / n,
        "mean_iou": totals["iou"] / n,
        "seconds": time.perf_counter() - start,
    }


def main():
    seed_everything(CFG.seed)

    # Start conservatively for a 4 GB RTX 3050. This is set before loaders
    # are created because the dataloader reads CFG.batch_size.
    CFG.batch_size = 1

    train_loader, val_loader, _test_loader, vocab = create_dataloaders()

    print(f"Device: {CFG.device}")
    print(f"Vocabulary size built from train captions: {len(vocab)}")
    print(
        f"Train samples: {len(train_loader.dataset)} | "
        f"Validation samples: {len(val_loader.dataset)}"
    )

    #model = MDCNet().to(CFG.device)
    model = MDCNet(vocab_size=len(vocab)).to(CFG.device)
    optimizer = AdamW(
        model.parameters(),
        lr=CFG.learning_rate,
        weight_decay=CFG.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    # Use 2 epochs for the first smoke test; increase after it runs cleanly.
    epochs = 1
    best_val = float("inf")
    best_path = "best_mdc_net_single_box.pth"
    start_all = time.perf_counter()

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer)
        val_metrics = run_epoch(model, val_loader)
        scheduler.step(val_metrics["loss"])

        print(f"\nEpoch {epoch}/{epochs}")
        for prefix, metrics in (("Train", train_metrics), ("Val", val_metrics)):
            print(
                f"{prefix}: total_loss={metrics['loss']:.4f}, "
                f"class_acc={metrics['class_accuracy']:.4f}, "
                f"location_acc={metrics['location_accuracy']:.4f}, "
                f"bbox_loss={metrics['bbox_loss']:.4f}, "
                f"mean_IoU={metrics['mean_iou']:.4f}, "
                f"time={metrics['seconds']:.1f}s"
            )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": best_val,
                    "label_to_id": CFG.label_to_id,
                    "vocab_size": len(vocab),
                    "model_name": CFG.model_name,
                    "img_size": CFG.img_size,
                },
                best_path,
            )
            print(f"Saved best checkpoint: {best_path}")

    print(f"Total training time: {(time.perf_counter() - start_all) / 60:.1f} minutes")
    print("Training smoke test finished. The old checkpoint was not overwritten.")


if __name__ == "__main__":
    main()
