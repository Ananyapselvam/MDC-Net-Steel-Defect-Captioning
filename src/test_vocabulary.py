import pandas as pd
from vocabulary import Vocabulary

CSV_PATH = r"D:\MDC Net Project\data\processed\train.csv"

df = pd.read_csv(CSV_PATH)

vocab = Vocabulary(freq_threshold=1)
vocab.build_vocab(df["caption"].tolist())

print("Vocabulary size:", len(vocab))

print("\nWord → ID:")
for word, idx in vocab.stoi.items():
    print(f"{word:15} -> {idx}")

sample_caption = df["caption"].iloc[0]

print("\nSample caption:")
print(sample_caption)

numericalized = vocab.numericalize(sample_caption)

print("\nNumericalized:")
print(numericalized)

print("\nDecoded:")
print(vocab.decode(numericalized))