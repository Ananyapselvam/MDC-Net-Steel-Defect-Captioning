from dataloader import create_dataloaders


train_loader, val_loader, vocab = create_dataloaders()

print("Training batches:", len(train_loader))
print("Validation batches:", len(val_loader))
print("Vocabulary size:", len(vocab))


# Load one batch
batch = next(iter(train_loader))

print("\nBatch information:")
print("Images shape:", batch["images"].shape)
print("Captions shape:", batch["captions"].shape)
print("Bounding boxes shape:", batch["bboxes"].shape)

print("\nLabels:")
print(batch["labels"])

print("\nImage IDs:")
print(batch["image_ids"])

print("\nCaption IDs:")
print(batch["captions"])