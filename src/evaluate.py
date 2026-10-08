import torch

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


def get_caption_location(caption):
    """
    Extract the location phrase from a generated caption.
    """

    location_phrases = [
        "top left",
        "top center",
        "top right",
        "middle left",
        "center",
        "middle right",
        "bottom left",
        "bottom center",
        "bottom right"
    ]

    caption = caption.lower()

    for location in location_phrases:

        if location in caption:

            return location.replace(
                " ",
                "-"
            )

    return None


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

        # --------------------------------------------------
        # Defect prediction
        # --------------------------------------------------
        predicted_class_id = (
            class_logits
            .argmax(dim=1)
            .item()
        )

        predicted_class = None

        for label, class_id in (
            CFG.label_to_id.items()
        ):

            if class_id == predicted_class_id:
                predicted_class = label
                break

        # --------------------------------------------------
        # Location prediction
        # --------------------------------------------------
        predicted_location_id = (
            location_logits
            .argmax(dim=1)
            .item()
        )

        predicted_location = (
            LOCATION_NAMES[
                predicted_location_id
            ]
        )

    return (
        generated,
        predicted_class,
        predicted_location
    )


def main():

    print("Device:", CFG.device)

    # --------------------------------------------------
    # Load test data
    # --------------------------------------------------
    (
        _,
        _,
        test_loader,
        vocab
    ) = create_dataloaders()

    print(
        "Test samples:",
        len(test_loader.dataset)
    )

    print(
        "Vocabulary size:",
        len(vocab)
    )

    # --------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------
    model = MDCNet().to(
        CFG.device
    )

    checkpoint = torch.load(
        "best_mdc_net.pth",
        map_location=CFG.device
    )

    model.load_state_dict(
        checkpoint
    )

    model.eval()

    print(
        "\nBest model loaded successfully."
    )

    # --------------------------------------------------
    # Metric counters
    # --------------------------------------------------
    total_samples = 0

    correct_class = 0
    correct_location = 0
    correct_caption = 0
    correct_caption_location = 0

    # --------------------------------------------------
    # Sample display limit
    # --------------------------------------------------
    samples_to_display = 20

    displayed = 0

    print("\n" + "=" * 75)
    print("FULL TEST SET EVALUATION")
    print("=" * 75)

    # --------------------------------------------------
    # Test loop
    # --------------------------------------------------
    for batch in test_loader:

        images = batch["images"].to(
            CFG.device
        )

        bboxes = batch["bboxes"].to(
            CFG.device
        )

        captions = batch["captions"]

        labels = batch["labels"]

        location_ids = batch[
            "location_ids"
        ]

        image_ids = batch[
            "image_ids"
        ]

        for i in range(
            images.size(0)
        ):

            image = images[
                i:i + 1
            ]

            bbox = bboxes[
                i:i + 1
            ]

            # --------------------------------------------------
            # Actual values
            # --------------------------------------------------
            actual_class = labels[i]

            actual_location = (
                LOCATION_NAMES[
                    int(location_ids[i])
                ]
            )

            actual_caption = (
                decode_tokens(
                    captions[i],
                    vocab
                )
            )

            # --------------------------------------------------
            # Predictions
            # --------------------------------------------------
            (
                generated_tokens,
                predicted_class,
                predicted_location
            ) = generate_caption(
                model,
                image,
                bbox,
                vocab
            )

            predicted_caption = (
                decode_tokens(
                    generated_tokens,
                    vocab
                )
            )

            # --------------------------------------------------
            # Classification accuracy
            # --------------------------------------------------
            if (
                predicted_class
                == actual_class
            ):
                correct_class += 1

            # --------------------------------------------------
            # Location accuracy
            # --------------------------------------------------
            if (
                predicted_location
                == actual_location
            ):
                correct_location += 1

            # --------------------------------------------------
            # Exact caption accuracy
            # --------------------------------------------------
            if (
                predicted_caption
                == actual_caption
            ):
                correct_caption += 1

            # --------------------------------------------------
            # Caption location accuracy
            # --------------------------------------------------
            predicted_caption_location = (
                get_caption_location(
                    predicted_caption
                )
            )

            if (
                predicted_caption_location
                == actual_location
            ):
                correct_caption_location += 1

            total_samples += 1

            # --------------------------------------------------
            # Display first 20
            # --------------------------------------------------
            if displayed < samples_to_display:

                print("\n" + "-" * 75)

                print(
                    f"Image ID: {image_ids[i]}"
                )

                print(
                    f"Actual class:       {actual_class}"
                )

                print(
                    f"Predicted class:    {predicted_class} "
                    f"{'✓' if predicted_class == actual_class else '✗'}"
                )

                print(
                    f"Actual location:    {actual_location}"
                )

                print(
                    f"Predicted location: {predicted_location} "
                    f"{'✓' if predicted_location == actual_location else '✗'}"
                )

                print(
                    f"Actual caption:     {actual_caption}"
                )

                print(
                    f"Predicted caption:  {predicted_caption}"
                )

                displayed += 1

    # --------------------------------------------------
    # Final metrics
    # --------------------------------------------------
    class_accuracy = (
        100.0 * correct_class / total_samples
    )

    location_accuracy = (
        100.0 * correct_location / total_samples
    )

    caption_accuracy = (
        100.0 * correct_caption / total_samples
    )

    caption_location_accuracy = (
        100.0
        * correct_caption_location
        / total_samples
    )

    # --------------------------------------------------
    # Print final results
    # --------------------------------------------------
    print("\n" + "=" * 75)
    print("FINAL TEST RESULTS")
    print("=" * 75)

    print(
        f"Total test annotations: "
        f"{total_samples}"
    )

    print(
        f"Defect classification accuracy: "
        f"{class_accuracy:.2f}%"
    )

    print(
        f"Location classification accuracy: "
        f"{location_accuracy:.2f}%"
    )

    print(
        f"Exact caption accuracy: "
        f"{caption_accuracy:.2f}%"
    )

    print(
        f"Caption location accuracy: "
        f"{caption_location_accuracy:.2f}%"
    )

    print("=" * 75)


if __name__ == "__main__":
    main()