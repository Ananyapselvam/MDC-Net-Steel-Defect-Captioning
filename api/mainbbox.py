import sys
from pathlib import Path

import cv2
import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(
    str(PROJECT_ROOT)
)

from src.config import CFG
from src.model import MDCNet
from src.vocabulary import Vocabulary


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="MDC-Net Steel Defect Detection API",
    description=(
        "AI-powered steel surface defect "
        "classification, localization and "
        "caption generation."
    ),
    version="1.0.0"
)


# --------------------------------------------------
# Location names
# --------------------------------------------------

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


# --------------------------------------------------
# Load model using the current single-box checkpoint
# --------------------------------------------------

checkpoint_path = PROJECT_ROOT / "best_mdc_net_single_box.pth"
checkpoint = torch.load(
    checkpoint_path,
    map_location=CFG.device,
    weights_only=False,
)

model = MDCNet(vocab_size=checkpoint.get("vocab_size", CFG.vocab_size)).to(CFG.device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# --------------------------------------------------
# Build vocabulary
# --------------------------------------------------

vocab = Vocabulary(
    freq_threshold=1
)

# The API uses the same vocabulary size/order
# as the trained model.
vocab.itos = {
    0: "<PAD>",
    1: "<SOS>",
    2: "<EOS>",
    3: "<UNK>",
    4: "a",
    5: "crazing",
    6: "defect",
    7: "is",
    8: "detected",
    9: "in",
    10: "the",
    11: "center",
    12: "region",
    13: "top",
    14: "left",
    15: "bottom",
    16: "right",
    17: "middle",
    18: "patches",
    19: "inclusion",
    20: "pitted",
    21: "surface",
    22: "rolled-in",
    23: "scale",
    24: "scratches"
}

vocab.stoi = {
    word: idx
    for idx, word in vocab.itos.items()
}


# --------------------------------------------------
# Caption decoding
# --------------------------------------------------

def decode_tokens(tokens):

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


# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------

def preprocess_image(image):

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    original_h, original_w = image.shape[:2]

    # For API inference we use the full image.
    resized = cv2.resize(
        image,
        (
            CFG.img_size,
            CFG.img_size
        )
    )

    resized = resized.astype(
        np.float32
    ) / 255.0

    # ImageNet normalization
    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32
    )

    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32
    )

    resized = (
        resized - mean
    ) / std

    tensor = torch.tensor(
        resized,
        dtype=torch.float32
    )

    tensor = tensor.permute(
        2,
        0,
        1
    )

    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(
        CFG.device
    )

    return (
        tensor,
        original_w,
        original_h
    )


# --------------------------------------------------
# Generate caption and predictions from image only
# --------------------------------------------------

def generate_caption(image_tensor):
    with torch.no_grad():
        # Current encoder returns: memory, class logits, location logits, bbox prediction
        memory, class_logits, location_logits, bbox_pred = model.encoder(image_tensor)

        generated = [CFG.sos_idx]
        for _ in range(CFG.max_len):
            tokens = torch.tensor(
                [generated], dtype=torch.long, device=CFG.device
            )
            output = model.decoder(tokens, memory)
            next_token = int(output[0, -1].argmax().item())
            generated.append(next_token)
            if next_token == CFG.eos_idx:
                break

        predicted_class_id = int(class_logits.argmax(dim=1).item())
        predicted_location_id = int(location_logits.argmax(dim=1).item())
        predicted_bbox = bbox_pred[0].detach().cpu().tolist()

    predicted_class = next(
        label for label, class_id in CFG.label_to_id.items()
        if class_id == predicted_class_id
    )
    predicted_location = LOCATION_NAMES[predicted_location_id]
    caption = decode_tokens(generated)

    return predicted_class, predicted_location, caption, predicted_bbox


# --------------------------------------------------
# API routes
# --------------------------------------------------

@app.get("/")
def root():

    return {
        "message": "MDC-Net Steel Defect Detection API",
        "status": "running"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "device": str(CFG.device),
        "model_loaded": True
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    contents = await file.read()

    image_array = np.frombuffer(
        contents,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image is None:

        return JSONResponse(
            status_code=400,
            content={
                "error": "Invalid image file."
            }
        )

    (
        image_tensor,
        original_w,
        original_h
    ) = preprocess_image(
        image
    )

    # Predict class, location, caption and normalized xyxy box directly from image.
    predicted_class, predicted_location, caption, predicted_bbox = generate_caption(
        image_tensor
    )

    # Convert normalized xyxy coordinates to original image pixel coordinates.
    x1, y1, x2, y2 = predicted_bbox
    x_min = max(0, min(original_w - 1, int(round(x1 * original_w))))
    y_min = max(0, min(original_h - 1, int(round(y1 * original_h))))
    x_max = max(x_min + 1, min(original_w, int(round(x2 * original_w))))
    y_max = max(y_min + 1, min(original_h, int(round(y2 * original_h))))

    return {
        "filename": file.filename,
        "image_width": original_w,
        "image_height": original_h,
        "defect": predicted_class,
        "location": predicted_location,
        "bbox": {
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
        },
        "bbox_normalized_xyxy": [round(float(v), 6) for v in predicted_bbox],
        "bbox_type": "model_predicted_bbox",
        "caption": caption,
    }

