import torch
import pandas as pd
import cv2
import albumentations as A
from torch.utils.data import Dataset

from src.augmentation import get_train_transforms, get_val_transforms
from src.config import CFG


class NEUDataset(Dataset):

    def __init__(self, csv_file, vocab, train=True):
        self.df = pd.read_csv(csv_file)
        self.vocab = vocab
        self.train = train

        # --------------------------------------------------
        # Augmentation only
        # --------------------------------------------------
        base_transform = (
            get_train_transforms()
            if train
            else get_val_transforms()
        )

        self.transform = A.Compose(
            base_transform.transforms,
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["labels"],
                min_visibility=0.1
            )
        )

        self.resize = A.Resize(
            CFG.img_size,
            CFG.img_size
        )

        self.normalize = A.Normalize()

    def __len__(self):
        return len(self.df)

    # ------------------------------------------------------
    # Convert bbox center into 3x3 spatial location
    # ------------------------------------------------------
    @staticmethod
    def get_location_id(
        bbox,
        image_width,
        image_height
    ):

        xmin, ymin, xmax, ymax = bbox

        center_x = (
            (xmin + xmax) / 2
        )

        center_y = (
            (ymin + ymax) / 2
        )

        # Normalize center coordinates
        center_x /= image_width
        center_y /= image_height

        # Horizontal position
        if center_x < 1 / 3:
            col = 0
        elif center_x < 2 / 3:
            col = 1
        else:
            col = 2

        # Vertical position
        if center_y < 1 / 3:
            row = 0
        elif center_y < 2 / 3:
            row = 1
        else:
            row = 2

        # Convert 3x3 position to class ID
        location_id = (
            row * 3 + col
        )

        return location_id

    def __getitem__(self, index):

        row = self.df.iloc[index]

        # --------------------------------------------------
        # Load image
        # --------------------------------------------------
        image_path = row["image_path"]

        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        original_h, original_w = image.shape[:2]

        # --------------------------------------------------
        # Original bounding box
        # --------------------------------------------------
        bbox = [
            float(row["xmin"]),
            float(row["ymin"]),
            float(row["xmax"]),
            float(row["ymax"])
        ]

        label = row["label"]

        # --------------------------------------------------
        # Location target from ORIGINAL bbox
        # --------------------------------------------------
        location_id = self.get_location_id(
            bbox,
            original_w,
            original_h
        )

        # --------------------------------------------------
        # Normalized ORIGINAL bbox
        # --------------------------------------------------
        original_bbox = torch.tensor(
            [
                bbox[0] / original_w,
                bbox[1] / original_h,
                bbox[2] / original_w,
                bbox[3] / original_h
            ],
            dtype=torch.float32
        )

        # --------------------------------------------------
        # Apply augmentation
        # --------------------------------------------------
        transformed = self.transform(
            image=image,
            bboxes=[bbox],
            labels=[label]
        )

        image = transformed["image"]

        transformed_bboxes = transformed["bboxes"]

        if len(transformed_bboxes) == 0:
            bbox = bbox
        else:
            bbox = transformed_bboxes[0]

        # --------------------------------------------------
        # Crop defect region
        # --------------------------------------------------
        h, w = image.shape[:2]

        xmin, ymin, xmax, ymax = bbox

        xmin = max(
            0,
            min(int(xmin), w - 1)
        )

        ymin = max(
            0,
            min(int(ymin), h - 1)
        )

        xmax = max(
            xmin + 1,
            min(int(xmax), w)
        )

        ymax = max(
            ymin + 1,
            min(int(ymax), h)
        )

        # Small context margin
        margin_x = int(
            (xmax - xmin) * 0.15
        )

        margin_y = int(
            (ymax - ymin) * 0.15
        )

        crop_xmin = max(
            0,
            xmin - margin_x
        )

        crop_ymin = max(
            0,
            ymin - margin_y
        )

        crop_xmax = min(
            w,
            xmax + margin_x
        )

        crop_ymax = min(
            h,
            ymax + margin_y
        )

        image = image[
            crop_ymin:crop_ymax,
            crop_xmin:crop_xmax
        ]

        if image.size == 0:
            raise RuntimeError(
                f"Empty crop generated for image: "
                f"{image_path}"
            )

        # --------------------------------------------------
        # Resize
        # --------------------------------------------------
        image = self.resize(
            image=image
        )["image"]

        # --------------------------------------------------
        # Normalize
        # --------------------------------------------------
        image = self.normalize(
            image=image
        )["image"]

        # --------------------------------------------------
        # HWC -> CHW
        # --------------------------------------------------
        image = torch.tensor(
            image,
            dtype=torch.float32
        ).permute(2, 0, 1)

        # --------------------------------------------------
        # Caption -> token IDs
        # --------------------------------------------------
        caption = str(
            row["caption"]
        )

        tokens = self.vocab.numericalize(
            caption
        )

        tokens = (
            [CFG.sos_idx]
            + tokens
            + [CFG.eos_idx]
        )

        tokens = tokens[:CFG.max_len]

        caption_tensor = torch.tensor(
            tokens,
            dtype=torch.long
        )

        # --------------------------------------------------
        # Return
        # --------------------------------------------------
        return {
            "image": image,

            "caption": caption_tensor,

            "label": label,

            "bbox": original_bbox,

            "location_id": torch.tensor(
                location_id,
                dtype=torch.long
            ),

            "image_id": row["image_id"]
        }