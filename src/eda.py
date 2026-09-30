import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = r"D:\MDC Net Project\data\processed\neu_annotations.csv"

# Load data
df = pd.read_csv(CSV_PATH)

# Basic information
print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

# Missing values
print("\nMissing values:")
print(df.isnull().sum())

# Class distribution
print("\nClass distribution:")
print(df["label"].value_counts())

# Plot class distribution
df["label"].value_counts().plot(kind="bar")
plt.title("NEU-DET Defect Class Distribution")
plt.xlabel("Defect Class")
plt.ylabel("Number of Annotations")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Display sample records
print("\nSample records:")
print(df[[
    "image_id",
    "label",
    "xmin",
    "ymin",
    "xmax",
    "ymax",
    "location",
    "caption"
]].head(10))