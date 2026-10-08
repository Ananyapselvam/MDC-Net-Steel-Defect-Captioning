import os
import torch
import torch.nn as nn
from torch.optim import AdamW

from src.config import CFG
from src.dataloader import create_dataloaders
from src.model import MDCNet


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer
):

    model.train()

    total_loss = 0.0
    total_caption_loss = 0.0
    total_classification_loss = 0.0
    total_location_loss = 0.0

    for batch_idx, batch in enumerate(loader):

        images = batch["images"].to(CFG.device)
        captions = batch["captions"].to(CFG.device)
        bboxes = batch["bboxes"].to(CFG.device)
        location_ids = batch["location_ids"].to(CFG.device)

        # Teacher forcing
        inputs = captions[:, :-1]
        targets = captions[:, 1:]

        optimizer.zero_grad()

        # --------------------------------------------------
        # Forward pass
        # --------------------------------------------------
        (
            caption_outputs,
            class_logits,
            location_logits
        ) = model(
            images,
            inputs,
            bboxes
        )

        # --------------------------------------------------
        # Caption loss
        # --------------------------------------------------
        caption_loss = criterion(
            caption_outputs.reshape(
                -1,
                CFG.vocab_size
            ),
            targets.reshape(-1)
        )

        # --------------------------------------------------
        # Defect classification target
        # --------------------------------------------------
        class_targets = torch.tensor(
            [
                CFG.label_to_id[label]
                for label in batch["labels"]
            ],
            dtype=torch.long,
            device=CFG.device
        )

        # --------------------------------------------------
        # Defect classification loss
        # --------------------------------------------------
        classification_loss = nn.CrossEntropyLoss()(
            class_logits,
            class_targets
        )

        # --------------------------------------------------
        # Spatial location loss
        # --------------------------------------------------
        location_loss = nn.CrossEntropyLoss()(
            location_logits,
            location_ids
        )

        # --------------------------------------------------
        # Combined loss
        #
        # Caption       = main task
        # Classification = auxiliary task
        # Location       = auxiliary spatial task
        # --------------------------------------------------
        loss = (
            caption_loss
            + 0.5 * classification_loss
            + CFG.location_loss_weight * location_loss
        )

        # --------------------------------------------------
        # Backpropagation
        # --------------------------------------------------
        loss.backward()

        optimizer.step()

        # --------------------------------------------------
        # Track losses
        # --------------------------------------------------
        total_loss += loss.item()

        total_caption_loss += (
            caption_loss.item()
        )

        total_classification_loss += (
            classification_loss.item()
        )

        total_location_loss += (
            location_loss.item()
        )

        if (batch_idx + 1) % 100 == 0:

            print(
                f"Batch [{batch_idx + 1}/{len(loader)}] "
                f"Total: {loss.item():.4f} | "
                f"Caption: {caption_loss.item():.4f} | "
                f"Class: {classification_loss.item():.4f} | "
                f"Location: {location_loss.item():.4f}"
            )

    num_batches = len(loader)

    return (
        total_loss / num_batches,
        total_caption_loss / num_batches,
        total_classification_loss / num_batches,
        total_location_loss / num_batches
    )


def validate(
    model,
    loader,
    criterion
):

    model.eval()

    total_loss = 0.0
    total_caption_loss = 0.0
    total_classification_loss = 0.0
    total_location_loss = 0.0

    with torch.no_grad():

        for batch in loader:

            images = batch["images"].to(CFG.device)
            captions = batch["captions"].to(CFG.device)
            bboxes = batch["bboxes"].to(CFG.device)
            location_ids = batch["location_ids"].to(CFG.device)

            # Teacher forcing
            inputs = captions[:, :-1]
            targets = captions[:, 1:]

            # --------------------------------------------------
            # Forward pass
            # --------------------------------------------------
            (
                caption_outputs,
                class_logits,
                location_logits
            ) = model(
                images,
                inputs,
                bboxes
            )

            # --------------------------------------------------
            # Caption loss
            # --------------------------------------------------
            caption_loss = criterion(
                caption_outputs.reshape(
                    -1,
                    CFG.vocab_size
                ),
                targets.reshape(-1)
            )

            # --------------------------------------------------
            # Defect classification target
            # --------------------------------------------------
            class_targets = torch.tensor(
                [
                    CFG.label_to_id[label]
                    for label in batch["labels"]
                ],
                dtype=torch.long,
                device=CFG.device
            )

            # --------------------------------------------------
            # Defect classification loss
            # --------------------------------------------------
            classification_loss = nn.CrossEntropyLoss()(
                class_logits,
                class_targets
            )

            # --------------------------------------------------
            # Location loss
            # --------------------------------------------------
            location_loss = nn.CrossEntropyLoss()(
                location_logits,
                location_ids
            )

            # --------------------------------------------------
            # Combined loss
            # --------------------------------------------------
            loss = (
                caption_loss
                + 0.5 * classification_loss
                + CFG.location_loss_weight * location_loss
            )

            total_loss += loss.item()

            total_caption_loss += (
                caption_loss.item()
            )

            total_classification_loss += (
                classification_loss.item()
            )

            total_location_loss += (
                location_loss.item()
            )

    num_batches = len(loader)

    return (
        total_loss / num_batches,
        total_caption_loss / num_batches,
        total_classification_loss / num_batches,
        total_location_loss / num_batches
    )


def main():

    print("Device:", CFG.device)

    # --------------------------------------------------
    # Data
    # --------------------------------------------------
    (
        train_loader,
        val_loader,
        _,
        vocab
    ) = create_dataloaders()

    print(
        "Training samples:",
        len(train_loader.dataset)
    )

    print(
        "Validation samples:",
        len(val_loader.dataset)
    )

    print(
        "Vocabulary size:",
        len(vocab)
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    model = MDCNet().to(
        CFG.device
    )

    # --------------------------------------------------
    # Caption criterion
    # --------------------------------------------------
    criterion = nn.CrossEntropyLoss(
        ignore_index=CFG.pad_idx
    )

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------
    optimizer = AdamW(
        model.parameters(),
        lr=CFG.learning_rate,
        weight_decay=CFG.weight_decay
    )

    # --------------------------------------------------
    # Learning-rate scheduler
    # --------------------------------------------------
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    best_val_loss = float("inf")

    print("\nStarting training...\n")

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------
    for epoch in range(CFG.epochs):

        print("=" * 60)

        print(
            f"Epoch {epoch + 1}/{CFG.epochs}"
        )

        print("=" * 60)

        (
            train_loss,
            train_caption_loss,
            train_classification_loss,
            train_location_loss
        ) = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        (
            val_loss,
            val_caption_loss,
            val_classification_loss,
            val_location_loss
        ) = validate(
            model,
            val_loader,
            criterion
        )

        scheduler.step(val_loss)

        # --------------------------------------------------
        # Print metrics
        # --------------------------------------------------
        print(
            f"\nTrain Total Loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Caption Loss: "
            f"{train_caption_loss:.4f}"
        )

        print(
            f"Train Classification Loss: "
            f"{train_classification_loss:.4f}"
        )

        print(
            f"Train Location Loss: "
            f"{train_location_loss:.4f}"
        )

        print(
            f"Val Total Loss: "
            f"{val_loss:.4f}"
        )

        print(
            f"Val Caption Loss: "
            f"{val_caption_loss:.4f}"
        )

        print(
            f"Val Classification Loss: "
            f"{val_classification_loss:.4f}"
        )

        print(
            f"Val Location Loss: "
            f"{val_location_loss:.4f}"
        )

        # --------------------------------------------------
        # Save best model safely
        # --------------------------------------------------
        if val_loss < best_val_loss:

            best_val_loss = val_loss

            temp_path = "best_mdc_net_temp.pth"
            final_path = "best_mdc_net.pth"

            torch.save(
                model.state_dict(),
                temp_path
            )

            checkpoint = torch.load(
                temp_path,
                map_location="cpu"
            )

            if len(checkpoint) > 0:

                os.replace(
                    temp_path,
                    final_path
                )

                print(
                    "✓ Best model saved and verified!"
                )

                print(
                    f"✓ Best validation loss: "
                    f"{best_val_loss:.4f}"
                )

            else:

                print(
                    "WARNING: "
                    "Checkpoint verification failed!"
                )

        print()


if __name__ == "__main__":
    main()