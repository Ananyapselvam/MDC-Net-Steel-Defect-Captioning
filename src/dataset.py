import os
import torch
import pandas as pd
import cv2
import albumentations as A
from torch.utils.data import Dataset

from augmentation import get_train_transforms, get_val_transforms
from vocabulary import Vocabulary
from config import CFG


class NEUDataset(Dataset):

    def __init__(self, csv_file, vocab, train=True):
        self.df = pd.read_csv(csv_file)
        self.vocab = vocab
        self.train = train

        # Original MDC-Net preprocessing uses 224x224 images
        # and normalization after augmentation.
        base_transform = (
            get_train_transforms()
            if train
            else get_val_transforms()
        )

        self.transform = A.Compose(
            base_transform.transforms + [
                A.Resize(CFG.img_size, CFG.img_size),
                A.Normalize()
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["labels"]
            )
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        # -------------------------
        # Load image
        # -------------------------
        image_path = row["image_path"]

        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # -------------------------
        # Bounding box
        # -------------------------
        bbox = [
            float(row["xmin"]),
            float(row["ymin"]),
            float(row["xmax"]),
            float(row["ymax"])
        ]

        label = row["label"]

        # -------------------------
        # Apply augmentation
        # -------------------------
        if self.train:
            transformed = self.transform(
                image=image,
                bboxes=[bbox],
                labels=[label]
            )

            image = transformed["image"]
            bbox = transformed["bboxes"][0]
        #label = transformed["labels"][0]
        else:
            image = cv2.resize(image,
            (CFG.img_size, CFG.img_size))

        image = image.astype("float32") / 255.0

        # HWC → CHW
        image = torch.tensor(
            image,
            dtype=torch.float32
        ).permute(2, 0, 1)

        # -------------------------
        # Caption → token IDs
        # -------------------------
        caption = str(row["caption"])

        tokens = self.vocab.numericalize(caption)

        tokens = (
            [CFG.sos_idx]
            + tokens
            + [CFG.eos_idx]
        )

        # Limit caption length
        tokens = tokens[:CFG.max_len]

        caption_tensor = torch.tensor(
            tokens,
            dtype=torch.long
        )

        # -------------------------
        # Return sample
        # -------------------------
        return {
            "image": image,
            "caption": caption_tensor,
            "label": label,
            "bbox": torch.tensor(
                bbox,
                dtype=torch.float32
            ),
            "image_id": row["image_id"]
        }