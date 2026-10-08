import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_CSV = "data/processed/neu_annotations.csv"

TRAIN_CSV = "data/processed/train.csv"
VAL_CSV = "data/processed/val.csv"
TEST_CSV = "data/processed/test.csv"

RANDOM_STATE = 42


df = pd.read_csv(INPUT_CSV)

# ---------------------------------------------------------
# 1. Get exactly ONE row per image for splitting
# ---------------------------------------------------------

image_df = df[["image_id", "label"]].drop_duplicates("image_id")

print("Total unique images:", len(image_df))

# ---------------------------------------------------------
# 2. Split images: 70% train, 30% temporary
# ---------------------------------------------------------

train_images, temp_images = train_test_split(
    image_df,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=image_df["label"]
)

# ---------------------------------------------------------
# 3. Split temporary: 20% validation, 10% test
# ---------------------------------------------------------

val_images, test_images = train_test_split(
    temp_images,
    test_size=1 / 3,
    random_state=RANDOM_STATE,
    stratify=temp_images["label"]
)

train_ids = set(train_images["image_id"])
val_ids = set(val_images["image_id"])
test_ids = set(test_images["image_id"])

# ---------------------------------------------------------
# 4. Create annotation-level CSVs using image IDs
# ---------------------------------------------------------

train_df = df[df["image_id"].isin(train_ids)].copy()
val_df = df[df["image_id"].isin(val_ids)].copy()
test_df = df[df["image_id"].isin(test_ids)].copy()

# ---------------------------------------------------------
# 5. Verify there is NO image leakage
# ---------------------------------------------------------

print("\nImage overlap check:")
print("Train ∩ Val :", len(train_ids & val_ids))
print("Train ∩ Test:", len(train_ids & test_ids))
print("Val ∩ Test  :", len(val_ids & test_ids))

# ---------------------------------------------------------
# 6. Save
# ---------------------------------------------------------

train_df.to_csv(TRAIN_CSV, index=False)
val_df.to_csv(VAL_CSV, index=False)
test_df.to_csv(TEST_CSV, index=False)

print("\nData splitting completed!")

print("\nUnique images:")
print("Train:", len(train_ids))
print("Val  :", len(val_ids))
print("Test :", len(test_ids))

print("\nAnnotation records:")
print("Train:", len(train_df))
print("Val  :", len(val_df))
print("Test :", len(test_df))

print("\nFiles saved:")
print(TRAIN_CSV)
print(VAL_CSV)
print(TEST_CSV)