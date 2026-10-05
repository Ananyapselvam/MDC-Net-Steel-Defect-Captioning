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

        # Teacher forcing:
        # input  = <SOS> ... last word
        # target = first word ... <EOS>
        inputs = captions[:, :-1]
        targets = captions[:, 1:]

        optimizer.zero_grad()

        outputs = model(images, inputs)

        # [batch, sequence, vocab] -> [batch*sequence, vocab]
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

    train_loader, val_loader, vocab = create_dataloaders()

    print("Training samples:", len(train_loader.dataset))
    print("Vocabulary size:", len(vocab))

    model = MDCNet().to(CFG.device)

    criterion = nn.CrossEntropyLoss(
        ignore_index=CFG.pad_idx
    )

    optimizer = AdamW(
        model.parameters(),
        lr=CFG.learning_rate,
        weight_decay=CFG.weight_decay
    )

    print("\nStarting training...\n")

    loss = train_one_epoch(
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

    print(f"\nTraining loss: {loss:.4f}")
    print(f"Validation loss: {val_loss:.4f}")

    torch.save(
        model.state_dict(),
        "mdc_net_epoch1.pth"
    )

    print("Model saved as mdc_net_epoch1.pth")


if __name__ == "__main__":
    main()