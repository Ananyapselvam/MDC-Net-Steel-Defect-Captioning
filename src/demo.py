import torch
import cv2
import matplotlib.pyplot as plt
import pandas as pd

from src.config import CFG
from src.dataloader import create_dataloaders
from src.model import MDCNet


LOCATION_NAMES = {
    0: "top-left",
    1: "top-center",
    2: "top-right",
    3: "middle-left",
    4: "center",
    5: "middle-right",
    6: "bottom-left",
    7: "bottom-center",
    8: "bottom-right"
}


def decode_tokens(tokens, vocab):

    words = []

    for token in tokens:

        token = int(token)

        if token == CFG.eos_idx:
            break

        if token in (
            CFG.pad_idx,
            CFG.sos_idx
        ):
            continue

        words.append(
            vocab.itos.get(
                token,
                "<UNK>"
            )
        )

    return " ".join(words)


def generate_caption(
    model,
    image,
    bbox,
    vocab
):

    model.eval()

    with torch.no_grad():

        (
            memory,
            class_logits,
            location_logits
        ) = model.encoder(
            image,
            bbox
        )

        generated = [
            CFG.sos_idx
        ]

        for _ in range(CFG.max_len):

            current_tokens = torch.tensor(
                [generated],
                dtype=torch.long,
                device=CFG.device
            )

            output = model.decoder(
                current_tokens,
                memory
            )

            next_token = (
                output[
                    0,
                    -1
                ]
                .argmax()
                .item()
            )

            generated.append(
                next_token
            )

            if next_token == CFG.eos_idx:
                break

        predicted_class_id = (
            class_logits
            .argmax(dim=1)
            .item()
        )

        predicted_location_id = (
            location_logits
            .argmax(dim=1)
            .item()
        )

    predicted_class = None

    for label, class_id in CFG.label_to_id.items():

        if class_id == predicted_class_id:
            predicted_class = label
            break

    predicted_location = LOCATION_NAMES[
        predicted_location_id
    ]

    predicted_caption = decode_tokens(
        generated,
        vocab
    )

    return (
        predicted_class,
        predicted_location,
        predicted_caption
    )


def main():

    print("Device:", CFG.device)

    # --------------------------------------------------
    # Load dataloaders and vocabulary
    # --------------------------------------------------

    _, _, test_loader, vocab = create_dataloaders()

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------

    model = MDCNet().to(CFG.device)

    checkpoint = torch.load(
        "best_mdc_net.pth",
        map_location=CFG.device
    )

    model.load_state_dict(checkpoint)

    model.eval()

    print("Best model loaded successfully.")

    # --------------------------------------------------
    # Select one test sample
    # --------------------------------------------------

    test_dataset = test_loader.dataset

    sample_index = 10

    sample = test_dataset[
        sample_index
    ]

    # --------------------------------------------------
    # Model input
    # --------------------------------------------------

    image_tensor = sample["image"].unsqueeze(0).to(
        CFG.device
    )

    bbox_tensor = sample["bbox"].unsqueeze(0).to(
        CFG.device
    )

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    (
        predicted_class,
        predicted_location,
        predicted_caption
    ) = generate_caption(
        model,
        image_tensor,
        bbox_tensor,
        vocab
    )

    # --------------------------------------------------
    # Ground truth
    # --------------------------------------------------

    actual_class = sample["label"]

    actual_location = LOCATION_NAMES[
        sample["location_id"].item()
    ]

    actual_caption = decode_tokens(
        sample["caption"],
        vocab
    )

    image_id = sample["image_id"]

    # --------------------------------------------------
    # Load original image
    # --------------------------------------------------

    df = pd.read_csv(
        "data/processed/test.csv"
    )

    row = df[
        df["image_id"] == image_id
    ].iloc[0]

    image_path = row["image_path"]

    image = cv2.imread(
        image_path
    )

    if image is None:

        raise FileNotFoundError(
            f"Could not load image: {image_path}"
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------
    # Draw ground-truth bounding box
    # --------------------------------------------------

    xmin = int(row["xmin"])
    ymin = int(row["ymin"])
    xmax = int(row["xmax"])
    ymax = int(row["ymax"])

    image_with_box = image.copy()

    cv2.rectangle(
        image_with_box,
        (xmin, ymin),
        (xmax, ymax),
        (255, 0, 0),
        2
    )

    # --------------------------------------------------
    # Display image
    # --------------------------------------------------

    plt.figure(
        figsize=(8, 8)
    )

    plt.imshow(
        image_with_box
    )

    plt.axis("off")

    plt.title(
        f"Steel Defect: {predicted_class}\n"
        f"Location: {predicted_location}"
    )

    plt.tight_layout()

    plt.savefig(
        "demo_prediction.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    # --------------------------------------------------
    # Print results
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("MDC-NET VISUAL DEMO")
    print("=" * 70)

    print(
        f"Image: {image_id}"
    )

    print("\nGROUND TRUTH")

    print(
        f"Defect:   {actual_class}"
    )

    print(
        f"Location: {actual_location}"
    )

    print(
        f"Caption:  {actual_caption}"
    )

    print("\nMODEL PREDICTION")

    print(
        f"Defect:   {predicted_class}"
    )

    print(
        f"Location: {predicted_location}"
    )

    print(
        f"Caption:  {predicted_caption}"
    )

    print("\nSaved visualization:")
    print(
        "demo_prediction.png"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()