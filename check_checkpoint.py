
import torch
from pathlib import Path

checkpoint_path = Path("best_mdc_net_single_box.pth")

print("Checkpoint exists:", checkpoint_path.exists())
print("Checkpoint size (MB):", round(checkpoint_path.stat().st_size / (1024 * 1024), 2))

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=False
)

print("Checkpoint keys:", list(checkpoint.keys()))
print("Saved epoch:", checkpoint.get("epoch"))
print("Validation loss:", checkpoint.get("val_loss"))
print("Vocabulary size:", checkpoint.get("vocab_size"))
print("\nPASS: Checkpoint loaded successfully.")
