import torch
import torch.nn as nn
from torch.optim import AdamW

from src.config import CFG
from src.dataloader import create_dataloaders
from src.model import MDCNet


def train_one_epoch(
    model,
    loader,
    caption_criterion,
    class_criterion,
    location_criterion,
    optimizer
):

    model.train()

    total_loss = 0.0
    total_caption_loss = 0.0
    total_class_loss = 0.0
    total_location_loss = 0.0

    for batch_idx, batch in enumerate(loader):

        images = batch["images"].to(CFG.device)
        captions = batch["captions"].to(CFG.device)
        bboxes = batch["bboxes"].to(CFG.device)

        location_ids = batch["location_ids"].to(CFG.device)

        # Convert string labels to numerical class IDs
        class_targets = torch.tensor(
            [
                CFG.label_to_id[label]
                for label in batch["labels"]
            ],
            dtype=torch.long,
            device=CFG.device
        )

        # Teacher forcing
        inputs = captions[:, :-1]
        targets = captions[:, 1:]

        optimizer.zero_grad()

        # Model outputs
        (
            caption_outputs,
            class_logits,
            location_logits
        ) = model(
            images,
            inputs,
            bboxes
        )

        # Caption loss
        caption_loss = caption_criterion(
            caption_outputs.reshape(
                -1,
                CFG.vocab_size
            ),
            targets.reshape(-1)
        )

        # Defect classification loss
        classification_loss = class_criterion(
            class_logits,
            class_targets
        )

        # Location classification loss
        location_loss = location_criterion(
            location_logits,
            location_ids
        )

        # Combined loss
        loss = (
            caption_loss
            + 0.5 * classification_loss
            + CFG.location_loss_weight * location_loss
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()
        total_caption_loss += caption_loss.item()
        total_class_loss += classification_loss.item()
        total_location_loss += location_loss.item()

        if (batch_idx + 1) % 100 == 0:

            print(
                f"Batch [{batch_idx + 1}/{len(loader)}] "
                f"Total: {loss.item():.4f} | "
                f"Caption: {caption_loss.item():.4f} | "
                f"Class: {classification_loss.item():.4f} | "
                f"Location: {location_loss.item():.4f}"
            )

    n = len(loader)

    return (
        total_loss / n,
        total_caption_loss / n,
        total_class_loss / n,
        total_location_loss / n
    )


def validate(
    model,
    loader,
    caption_criterion,
    class_criterion,
    location_criterion
):

    model.eval()

    total_loss = 0.0
    total_caption_loss = 0.0
    total_class_loss = 0.0
    total_location_loss = 0.0

    with torch.no_grad():

        for batch in loader:

            images = batch["images"].to(CFG.device)
            captions = batch["captions"].to(CFG.device)
            bboxes = batch["bboxes"].to(CFG.device)

            location_ids = batch["location_ids"].to(
                CFG.device
            )

            class_targets = torch.tensor(
                [
                    CFG.label_to_id[label]
                    for label in batch["labels"]
                ],
                dtype=torch.long,
                device=CFG.device
            )

            inputs = captions[:, :-1]
            targets = captions[:, 1:]

            (
                caption_outputs,
                class_logits,
                location_logits
            ) = model(
                images,
                inputs,
                bboxes
            )

            caption_loss = caption_criterion(
                caption_outputs.reshape(
                    -1,
                    CFG.vocab_size
                ),
                targets.reshape(-1)
            )

            classification_loss = class_criterion(
                class_logits,
                class_targets
            )

            location_loss = location_criterion(
                location_logits,
                location_ids
            )

            loss = (
                caption_loss
                + 0.5 * classification_loss
                + CFG.location_loss_weight * location_loss
            )

            total_loss += loss.item()
            total_caption_loss += caption_loss.item()
            total_class_loss += classification_loss.item()
            total_location_loss += location_loss.item()

    n = len(loader)

    return (
        total_loss / n,
        total_caption_loss / n,
        total_class_loss / n,
        total_location_loss / n
    )


def main():

    print("Device:", CFG.device)

    train_loader, val_loader, _, vocab = create_dataloaders()

    print(
        "Training samples:",
        len(train_loader.dataset)
    )

    print(
        "Vocabulary size:",
        len(vocab)
    )

    model = MDCNet().to(CFG.device)

    # Caption loss
    caption_criterion = nn.CrossEntropyLoss(
        ignore_index=CFG.pad_idx
    )

    # Defect classification loss
    class_criterion = nn.CrossEntropyLoss()

    # Location classification loss
    location_criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=CFG.learning_rate,
        weight_decay=CFG.weight_decay
    )

    print("\nStarting training...\n")

    best_val_loss = float("inf")

    for epoch in range(1, CFG.epochs + 1):

        print(
            f"\n{'=' * 70}"
        )

        print(
            f"Epoch {epoch}/{CFG.epochs}"
        )

        print(
            f"{'=' * 70}"
        )

        (
            train_loss,
            train_caption_loss,
            train_class_loss,
            train_location_loss
        ) = train_one_epoch(
            model,
            train_loader,
            caption_criterion,
            class_criterion,
            location_criterion,
            optimizer
        )

        (
            val_loss,
            val_caption_loss,
            val_class_loss,
            val_location_loss
        ) = validate(
            model,
            val_loader,
            caption_criterion,
            class_criterion,
            location_criterion
        )

        print(
            f"\nEpoch {epoch} Results:"
        )

        print(
            f"Train Total:    {train_loss:.4f}"
        )

        print(
            f"Train Caption:  {train_caption_loss:.4f}"
        )

        print(
            f"Train Class:    {train_class_loss:.4f}"
        )

        print(
            f"Train Location: {train_location_loss:.4f}"
        )

        print(
            f"\nVal Total:      {val_loss:.4f}"
        )

        print(
            f"Val Caption:    {val_caption_loss:.4f}"
        )

        print(
            f"Val Class:      {val_class_loss:.4f}"
        )

        print(
            f"Val Location:   {val_location_loss:.4f}"
        )

        # Save best model
        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                "best_mdc_net.pth"
            )

            print(
                "\n✓ Best model saved."
            )


if __name__ == "__main__":
    main()