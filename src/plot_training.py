import os
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# Training history from the final 5-epoch run
# --------------------------------------------------

history = {
    "epoch": [1, 2, 3, 4, 5],

    "train_total": [
        1.3786,
        0.4850,
        0.4189,
        0.3893,
        0.3675
    ],

    "val_total": [
        0.5474,
        0.4244,
        0.3925,
        0.3790,
        0.3794
    ],

    "train_caption": [
        0.9213,
        0.2474,
        0.2057,
        0.1892,
        0.1771
    ],

    "val_caption": [
        0.2717,
        0.2050,
        0.1848,
        0.1773,
        0.1751
    ],

    "train_class": [
        0.4903,
        0.0824,
        0.0520,
        0.0385,
        0.0318
    ],

    "val_class": [
        0.1530,
        0.0574,
        0.0469,
        0.0463,
        0.0621
    ],

    "train_location": [
        2.1217,
        1.9643,
        1.8722,
        1.8090,
        1.7448
    ],

    "val_location": [
        1.9924,
        1.9069,
        1.8422,
        1.7860,
        1.7331
    ]
}


df = pd.DataFrame(history)

os.makedirs(
    "results",
    exist_ok=True
)

# Save numerical history
df.to_csv(
    "results/training_history.csv",
    index=False
)


# --------------------------------------------------
# 1. Total loss
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    df["epoch"],
    df["train_total"],
    marker="o",
    label="Training Loss"
)

plt.plot(
    df["epoch"],
    df["val_total"],
    marker="o",
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("MDC-Net Training and Validation Loss")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/total_loss.png",
    dpi=300
)

plt.show()


# --------------------------------------------------
# 2. Caption loss
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    df["epoch"],
    df["train_caption"],
    marker="o",
    label="Training Caption Loss"
)

plt.plot(
    df["epoch"],
    df["val_caption"],
    marker="o",
    label="Validation Caption Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Caption Generation Loss")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/caption_loss.png",
    dpi=300
)

plt.show()


# --------------------------------------------------
# 3. Classification loss
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    df["epoch"],
    df["train_class"],
    marker="o",
    label="Training Classification Loss"
)

plt.plot(
    df["epoch"],
    df["val_class"],
    marker="o",
    label="Validation Classification Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Defect Classification Loss")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/classification_loss.png",
    dpi=300
)

plt.show()


# --------------------------------------------------
# 4. Location loss
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    df["epoch"],
    df["train_location"],
    marker="o",
    label="Training Location Loss"
)

plt.plot(
    df["epoch"],
    df["val_location"],
    marker="o",
    label="Validation Location Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Defect Location Loss")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/location_loss.png",
    dpi=300
)

plt.show()


print("\nTraining history saved successfully.")

print("\nGenerated files:")

print("results/training_history.csv")
print("results/total_loss.png")
print("results/caption_loss.png")
print("results/classification_loss.png")
print("results/location_loss.png")