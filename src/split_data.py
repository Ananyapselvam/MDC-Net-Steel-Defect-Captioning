import pandas as pd
from sklearn.model_selection import train_test_split

CSV_PATH = r"D:\MDC Net Project\data\processed\neu_annotations.csv"
OUTPUT_DIR = r"D:\MDC Net Project\data\processed"

# Load annotations
df = pd.read_csv(CSV_PATH)

# Get unique images and their class labels
images = df[["image_id", "label"]].drop_duplicates()

# 70% Train, 30% temporary
train_images, temp_images = train_test_split(
    images,
    test_size=0.30,
    random_state=42,
    stratify=images["label"]
)

# Split remaining 30% into 20% validation and 10% test
val_images, test_images = train_test_split(
    temp_images,
    test_size=1/3,
    random_state=42,
    stratify=temp_images["label"]
)

# Create annotation datasets
train_df = df[df["image_id"].isin(train_images["image_id"])]
val_df = df[df["image_id"].isin(val_images["image_id"])]
test_df = df[df["image_id"].isin(test_images["image_id"])]

# Save
train_df.to_csv(f"{OUTPUT_DIR}\\train.csv", index=False)
val_df.to_csv(f"{OUTPUT_DIR}\\val.csv", index=False)
test_df.to_csv(f"{OUTPUT_DIR}\\test.csv", index=False)

# Display results
print("Data splitting completed!\n")

print("Unique images:")
print("Train:", len(train_images))
print("Validation:", len(val_images))
print("Test:", len(test_images))

print("\nAnnotation records:")
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))

print("\nFiles saved:")
print("train.csv")
print("val.csv")
print("test.csv")