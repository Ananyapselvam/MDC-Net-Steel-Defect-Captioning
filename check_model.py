import torch

from src.config import CFG
from src.dataloader import create_dataloaders
from src.model import MDCNet

def main():
    print("Device:", CFG.device)

    # Use one sample to avoid unnecessary VRAM pressure during this check.
    CFG.batch_size = 1
    train_loader, val_loader, test_loader, vocab = create_dataloaders()
    print("Vocabulary size:", len(vocab))
    print("Train batches:", len(train_loader))
    print("Validation batches:", len(val_loader))
    print("Test batches:", len(test_loader))

    batch = next(iter(train_loader))
    images = batch["images"].to(CFG.device)
    captions = batch["captions"].to(CFG.device)
    bboxes = batch["bboxes"].to(CFG.device)
    labels = batch["labels"]
    locations = batch["location_ids"].to(CFG.device)

    print("Images:", tuple(images.shape))
    print("Captions:", tuple(captions.shape))
    print("Boxes:", tuple(bboxes.shape))
    print("Example class label:", labels[0])
    print("Locations:", tuple(locations.shape))
    print("BBox min/max:", bboxes.min().item(), bboxes.max().item())

    model = MDCNet(vocab_size=len(vocab)).to(CFG.device)
    model.eval()

    with torch.no_grad():
        caption_logits, class_logits, location_logits, bbox_pred = model(
            images, captions[:, :-1]
        )

    print("Caption logits:", tuple(caption_logits.shape))
    print("Class logits:", tuple(class_logits.shape))
    print("Location logits:", tuple(location_logits.shape))
    print("Predicted boxes:", tuple(bbox_pred.shape))

    assert images.ndim == 4 and images.shape[1] == 3
    assert captions.ndim == 2
    assert bboxes.shape == (images.shape[0], 4)
    assert caption_logits.shape[:2] == captions[:, :-1].shape
    assert caption_logits.shape[-1] == len(vocab)
    assert class_logits.shape == (images.shape[0], CFG.num_classes)
    assert location_logits.shape == (images.shape[0], CFG.num_locations)
    assert bbox_pred.shape == (images.shape[0], 4)
    assert torch.isfinite(bbox_pred).all()
    assert (bbox_pred >= 0).all() and (bbox_pred <= 1).all()

    print("\nPASS: dataloader and model output shapes are compatible.")

if __name__ == "__main__":
    main()
