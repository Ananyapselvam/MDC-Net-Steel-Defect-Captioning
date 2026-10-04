import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import matplotlib.pyplot as plt
import pandas as pd

from augmentation import get_train_transforms


CSV_PATH = r"D:\MDC Net Project\data\processed\train.csv"


# Load one training sample
df = pd.read_csv(CSV_PATH)
row = df.iloc[0]

image = cv2.imread(row["image_path"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

bbox = [
    [
        row["xmin"],
        row["ymin"],
        row["xmax"],
        row["ymax"]
    ]
]

label = [row["label"]]

# Apply augmentation
transform = get_train_transforms()

augmented = transform(
    image=image,
    bboxes=bbox,
    labels=label
)

aug_image = augmented["image"]
aug_bbox = augmented["bboxes"][0]


# Draw bounding boxes
def draw_box(img, box, title):

    x1, y1, x2, y2 = map(int, box)

    img = img.copy()

    cv2.rectangle(
        img,
        (x1, y1),
        (x2, y2),
        (255, 0, 0),
        2
    )

    plt.imshow(img)
    plt.title(title)
    plt.axis("off")


# Display original and augmented image
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
draw_box(
    image,
    bbox[0],
    "Original"
)

plt.subplot(1, 2, 2)
draw_box(
    aug_image,
    aug_bbox,
    "Augmented"
)

plt.tight_layout()
plt.show()