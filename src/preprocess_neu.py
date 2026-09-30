import os
import xml.etree.ElementTree as ET
import pandas as pd

# -----------------------------
# Paths
# -----------------------------
ANNOTATION_DIR = r"D:\MDC Net Project\data\raw\NEU-DET\ANNOTATIONS"
IMAGE_DIR = r"D:\MDC Net Project\data\raw\NEU-DET\IMAGES"

OUTPUT_DIR = r"D:\MDC Net Project\data\processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "neu_annotations.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------
# NEU-DET class mapping
# -----------------------------
LABEL_MAP = {
    "crazing": 0,
    "scratches": 1,
    "rolled-in_scale": 2,
    "pitted_surface": 3,
    "patches": 4,
    "inclusion": 5
}


# -----------------------------
# Determine defect location
# -----------------------------
def get_location(xmin, ymin, xmax, ymax, width, height):

    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2

    if center_x < width / 3:
        horizontal = "left"
    elif center_x < 2 * width / 3:
        horizontal = "center"
    else:
        horizontal = "right"

    if center_y < height / 3:
        vertical = "top"
    elif center_y < 2 * height / 3:
        vertical = "middle"
    else:
        vertical = "bottom"

    if vertical == "middle" and horizontal == "center":
        return "center"

    return f"{vertical}-{horizontal}"


# -----------------------------
# Generate caption
# -----------------------------
def create_caption(label, location):

    return (
        f"A {label.replace('_', ' ')} defect "
        f"is detected in the {location.replace('-', ' ')} region."
    )


# -----------------------------
# Parse XML annotations
# -----------------------------
records = []

xml_files = [
    file for file in os.listdir(ANNOTATION_DIR)
    if file.lower().endswith(".xml")
]

print(f"Found {len(xml_files)} XML annotation files.")


for xml_file in xml_files:

    xml_path = os.path.join(ANNOTATION_DIR, xml_file)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.findtext("filename")

    width = int(root.findtext("size/width"))
    height = int(root.findtext("size/height"))

    if not filename.lower().endswith(".jpg"):
        filename = filename + ".jpg"

    image_path = os.path.join(IMAGE_DIR, filename)

    # Check that image exists
    if not os.path.exists(image_path):
        print(f"Warning: image not found → {filename}")
        continue

    for obj in root.findall("object"):

        label = obj.findtext("name")

        bbox = obj.find("bndbox")

        xmin = int(bbox.findtext("xmin"))
        ymin = int(bbox.findtext("ymin"))
        xmax = int(bbox.findtext("xmax"))
        ymax = int(bbox.findtext("ymax"))

        location = get_location(
            xmin, ymin, xmax, ymax,
            width, height
        )

        caption = create_caption(
            label,
            location
        )

        records.append({
            "image_id": filename,
            "image_path": image_path,
            "label": label,
            "label_id": LABEL_MAP[label],
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
            "location": location,
            "caption": caption
        })


# -----------------------------
# Create dataframe
# -----------------------------
df = pd.DataFrame(records)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nPreprocessing completed!")
print(f"Total records: {len(df)}")
print(f"Saved to: {OUTPUT_FILE}")

print("\nClass distribution:")
print(df["label"].value_counts())

print("\nFirst 5 records:")
print(df.head())