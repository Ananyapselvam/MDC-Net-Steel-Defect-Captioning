import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from src.datasetwbox import NEUDataset
from src.vocabulary import Vocabulary
from src.config import CFG


TRAIN_CSV = "data/processed/train.csv"
VAL_CSV = "data/processed/val.csv"
TEST_CSV = "data/processed/test.csv"


def collate_fn(batch):

    images = torch.stack(
        [item["image"] for item in batch]
    )

    captions = [
        item["caption"] for item in batch
    ]

    captions = pad_sequence(
        captions,
        batch_first=True,
        padding_value=CFG.pad_idx
    )

    labels = [
        item["label"] for item in batch
    ]

    bboxes = torch.stack(
        [item["bbox"] for item in batch]
    )

    location_ids = torch.stack(
        [item["location_id"] for item in batch]
    )

    image_ids = [
        item["image_id"] for item in batch
    ]

    return {
        "images": images,
        "captions": captions,
        "labels": labels,
        "bboxes": bboxes,
        "location_ids": location_ids,
        "image_ids": image_ids
    }


def create_dataloaders():

    # Read CSV files
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)

    # Build vocabulary ONLY from training captions
    vocab = Vocabulary(freq_threshold=1)
    vocab.build_vocab(
        train_df["caption"].tolist()
    )

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

    test_dataset = NEUDataset(
        TEST_CSV,
        vocab,
        train=False
    )

    # Create dataloaders
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

    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        vocab
    )