import pandas as pd

from src.datasetwbox import NEUDataset
from vocabulary import Vocabulary


CSV_PATH = r"D:\MDC Net Project\data\processed\train.csv"


# Build vocabulary from training captions
df = pd.read_csv(CSV_PATH)

vocab = Vocabulary(freq_threshold=1)
vocab.build_vocab(df["caption"].tolist())


# Create dataset
dataset = NEUDataset(
    csv_file=CSV_PATH,
    vocab=vocab,
    train=True
)


print("Dataset size:", len(dataset))


# Load one sample
sample = dataset[0]

print("\nSample information:")
print("Image ID:", sample["image_id"])
print("Label:", sample["label"])
print("Image shape:", sample["image"].shape)
print("Caption IDs:", sample["caption"].tolist())
print("Bounding box:", sample["bbox"].tolist())

print("\nDecoded caption:")
print(vocab.decode(sample["caption"].tolist()))