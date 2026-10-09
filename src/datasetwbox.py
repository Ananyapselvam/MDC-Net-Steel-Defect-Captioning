import cv2
import numpy as np
import pandas as pd
import torch
import albumentations as A
from torch.utils.data import Dataset

from src.config import CFG
from src.vocabulary import Vocabulary


class NEUDataset(Dataset):
    """
    NEU-DET dataset for full-image defect classification, captioning,
    3x3 location classification, and bounding-box regression.

    Important:
    - The full image is retained; we do NOT crop to the ground-truth box.
    - Only photometric augmentation is used so spatial labels/captions remain valid.
    - Bboxes are returned normalized to [0, 1] as [xmin, ymin, xmax, ymax].
    """

    def __init__(self, csv_file, vocab, train=True):
        self.df = pd.read_csv(csv_file)
        self.vocab = vocab
        self.train = train

        # Keep geometry unchanged: flips/rotations would require updating the
        # location label and the spatial wording in the caption too.
        self.train_transform = A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.4
            ),
        ])

        self.resize = A.Resize(CFG.img_size, CFG.img_size)
        self.normalize = A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )

    def __len__(self):
        return len(self.df)

    @staticmethod
    def get_location_id(bbox_norm):
        xmin, ymin, xmax, ymax = bbox_norm
        center_x = (xmin + xmax) / 2.0
        center_y = (ymin + ymax) / 2.0

        col = 0 if center_x < 1 / 3 else (1 if center_x < 2 / 3 else 2)
        row = 0 if center_y < 1 / 3 else (1 if center_y < 2 / 3 else 2)
        return row * 3 + col

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_path = str(row["image_path"])

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        height, width = image.shape[:2]

        # Clamp annotation coordinates to the actual image bounds.
        xmin = float(np.clip(row["xmin"], 0, width - 1))
        ymin = float(np.clip(row["ymin"], 0, height - 1))
        xmax = float(np.clip(row["xmax"], xmin + 1, width))
        ymax = float(np.clip(row["ymax"], ymin + 1, height))

        # Normalize coordinates. Relative coordinates remain the same after
        # resizing the full image to a square.
        bbox = torch.tensor([
            xmin / width,
            ymin / height,
            xmax / width,
            ymax / height
        ], dtype=torch.float32)

        # Photometric augmentation only; no crop or geometric transform.
        if self.train:
            image = self.train_transform(image=image)["image"]

        image = self.resize(image=image)["image"]
        image = self.normalize(image=image)["image"]

        image_tensor = torch.from_numpy(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        ).float()

        caption_text = str(row["caption"])
        token_ids = self.vocab.numericalize(caption_text)
        token_ids = [CFG.sos_idx] + token_ids + [CFG.eos_idx]
        token_ids = token_ids[:CFG.max_len]

        caption_tensor = torch.tensor(token_ids, dtype=torch.long)
        location_id = self.get_location_id(bbox.tolist())

        return {
            "image": image_tensor,
            "caption": caption_tensor,
            "label": str(row["label"]),
            "bbox": bbox,
            "location_id": torch.tensor(location_id, dtype=torch.long),
            "image_id": str(row["image_id"])
        }
