import torch


class CFG:

    # -------------------------
    # Device
    # -------------------------
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # -------------------------
    # Image settings
    # -------------------------
    img_size = 224

    # -------------------------
    # Caption settings
    # -------------------------
    max_len = 100

    # -------------------------
    # Defect classes
    # -------------------------
    label_to_id = {
        "crazing": 0,
        "scratches": 1,
        "rolled-in_scale": 2,
        "pitted_surface": 3,
        "patches": 4,
        "inclusion": 5
    }

    num_classes = 6

    # -------------------------
    # Spatial location classes
    # -------------------------
    location_to_id = {
        "top-left": 0,
        "top-center": 1,
        "top-right": 2,
        "middle-left": 3,
        "center": 4,
        "middle-right": 5,
        "bottom-left": 6,
        "bottom-center": 7,
        "bottom-right": 8
    }

    num_locations = 9

    # Weight of location auxiliary loss
    location_loss_weight = 0.1

    # -------------------------
    # Special token IDs
    # -------------------------
    pad_idx = 0
    sos_idx = 1
    eos_idx = 2
    unk_idx = 3

    # -------------------------
    # Vocabulary
    # -------------------------
    vocab_size = 25

    # -------------------------
    # Model
    # -------------------------
    model_name = (
        "deit3_medium_patch16_224.fb_in22k_ft_in1k"
    )

    # DeiT3 224x224 with 16x16 patches
    num_patches = 196

    # -------------------------
    # Training
    # -------------------------
    batch_size = 8

    # Keep this at 2 for the next smoke test
    epochs = 5

    learning_rate = 1e-5
    weight_decay = 1e-4

    # -------------------------
    # Reproducibility
    # -------------------------
    seed = 42