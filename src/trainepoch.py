import torch
import torch.nn as nn
from torch.optim import AdamW

from config import CFG
from dataloader import create_dataloaders
from model import MDCNet


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0

    for batch_idx, batch in enumerate(loader):

        images = batch["images"].to(CFG.device)
        captions = batch["captions"].to(CFG.device)

        # Teacher forcing
        inputs = captions[:, :-1]
        targets = captions[:, 1:]

        optimizer.zero_grad()

        outputs = model(images, inputs)

        loss = criterion(
            outputs.reshape(-1, CFG.vocab_size),
            targets.reshape(-1)
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 100 == 0:
            print(
                f"Batch [{batch_idx + 1}/{len(loader)}] "
                f"Loss: {loss.item():.4f}"
            )

    return total_loss / len(loader)


def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():

        for batch in loader:

            images = batch["images"].to(CFG.device)
            captions = batch["captions"].to(CFG.device)

            inputs = captions[:, :-1]
            targets = captions[:, 1:]

            outputs = model(images, inputs)

            loss = criterion(
                outputs.reshape(-1, CFG.vocab_size),
                targets.reshape(-1)
            )

            total_loss += loss.item()

    return total_loss / len(loader)


def main():

    print("Device:", CFG.device)

    # Load data
    train_loader, val_loader, vocab = create_dataloaders()

    print("Training samples:", len(train_loader.dataset))
    print("Validation samples:", len(val_loader.dataset))
    print("Vocabulary size:", len(vocab))

    # Model
    model = MDCNet().to(CFG.device)

    # Loss
    criterion = nn.CrossEntropyLoss(
        ignore_index=CFG.pad_idx
    )

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=CFG.learning_rate,
        weight_decay=CFG.weight_decay
    )

    best_val_loss = float("inf")

    print("\nStarting training...\n")

    # Multi-epoch training
    for epoch in range(CFG.epochs):

        print("=" * 50)
        print(f"Epoch {epoch + 1}/{CFG.epochs}")
        print("=" * 50)

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss = validate(
            model,
            val_loader,
            criterion
        )

        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                "best_mdc_net.pth"
            )

            print("✓ Best model saved!")

        print()


if __name__ == "__main__":
    main()