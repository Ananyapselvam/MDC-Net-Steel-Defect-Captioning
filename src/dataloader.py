import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from dataset import NEUDataset
from vocabulary import Vocabulary
from config import CFG


TRAIN_CSV = r"D:\MDC Net Project\data\processed\train.csv"
VAL_CSV = r"D:\MDC Net Project\data\processed\val.csv"


def collate_fn(batch):

    # Stack images into one batch
    images = torch.stack([
        item["image"] for item in batch
    ])

    # Get captions
    captions = [
        item["caption"] for item in batch
    ]

    # Pad captions to the same length
    captions = pad_sequence(
        captions,
        batch_first=True,
        padding_value=CFG.pad_idx
    )

    labels = [
        item["label"] for item in batch
    ]

    bboxes = torch.stack([
        item["bbox"] for item in batch
    ])

    image_ids = [
        item["image_id"] for item in batch
    ]

    return {
        "images": images,
        "captions": captions,
        "labels": labels,
        "bboxes": bboxes,
        "image_ids": image_ids
    }


def create_dataloaders():

    # Build vocabulary ONLY from training captions
    train_df = pd.read_csv(TRAIN_CSV)

    vocab = Vocabulary(freq_threshold=1)
    vocab.build_vocab(train_df["caption"].tolist())

    # Create datasets
    train_dataset = NEUDataset(
        TRAIN_CSV,
        vocab,
        train=True
    )

    val_dataset = NEUDataset(
        VAL_CSV,
        vocab,
        train=False
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    return train_loader, val_loader, vocab