import torch


class CFG:

    # -------------------------
    # Device
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # Image settings
    # -------------------------
    img_size = 224

    # -------------------------
    # Caption settings
    # -------------------------
    max_len = 100

    # Special token IDs
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
    model_name = "deit3_medium_patch16_224.fb_in22k_ft_in1k"

    # DeiT3 224x224 with 16x16 patches
    num_patches = 196

    # -------------------------
    # Training
    # -------------------------
    batch_size = 4
    epochs = 20

    learning_rate = 1e-5
    weight_decay = 1e-4

    # -------------------------
    # Reproducibility
    # -------------------------
    seed = 42