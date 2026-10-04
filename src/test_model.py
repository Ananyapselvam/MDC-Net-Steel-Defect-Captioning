import torch

from config import CFG
from dataloader import create_dataloaders
from model import MDCNet


# Create DataLoader
train_loader, _, _ = create_dataloaders()

# Get one batch
batch = next(iter(train_loader))

images = batch["images"].to(CFG.device)
captions = batch["captions"].to(CFG.device)

print("Images:", images.shape)
print("Captions:", captions.shape)


# Create model
model = MDCNet().to(CFG.device)

print("\nModel created successfully.")


# Forward pass
with torch.no_grad():
    output = model(images, captions)

print("\nOutput shape:", output.shape)