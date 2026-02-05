import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.data.map_classes import to_names
from src.data.preprocess_img import crop_image_by_roi
from src.models.yolo_model import yolo_prediction


def detect_and_classify(classifier_name, classifier_model, yolo_model, img_pil):
    results = yolo_prediction(yolo_model, img_pil)
    names = to_names()
    draw = ImageDraw.Draw(img_pil)
    w, h = img_pil.size

    base_size = int(0.04 * min(w, h))
    font_size = max(12, min(40, base_size))
    font = ImageFont.load_default()
    for fname in [
        "arial.ttf",
        "DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]:
        try:
            font = ImageFont.truetype(fname, font_size)
            break
        except Exception:
            continue

    detected_classes = {}

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)

            sign = crop_image_by_roi(img_pil, x1, y1, x2, y2)
            sign_resized = sign.resize((32, 32))
            img_array = np.asarray(sign_resized).astype("float32") / 255.0

            if classifier_name == "cnn":
                inp = np.expand_dims(img_array, axis=0)
                preds = classifier_model.predict(inp, verbose=0)
            else:
                inp = img_array.flatten().reshape(1, -1)
                preds = (
                    classifier_model.predict_proba(inp)
                    if classifier_name == "rfc"
                    else classifier_model.predict(inp, verbose=0)
                )

            class_id = np.argmax(preds, axis=1)[0]
            class_name = names.get(class_id, "Unknown")

            random.seed(int(class_id))
            color = (
                random.randint(50, 255),
                random.randint(50, 255),
                random.randint(50, 255),
            )

            if class_id not in detected_classes:
                detected_classes[class_id] = (class_name, color)

            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

    if detected_classes:
        overlay = Image.new("RGBA", img_pil.size, (0, 0, 0, 0))
        draw_ov = ImageDraw.Draw(overlay)

        pad = 8
        box_size = font_size - 4
        max_text_w = 0

        for name, _ in detected_classes.values():
            try:
                bbox = draw.textbbox((0, 0), name, font=font)
                text_w = bbox[2] - bbox[0]
            except AttributeError:
                text_w = draw.textlength(name, font=font)
            if text_w > max_text_w:
                max_text_w = text_w

        legend_w = pad + box_size + pad + int(max_text_w) + pad
        legend_h = pad + len(detected_classes) * (font_size + 4) + pad - 4

        draw_ov.rectangle([0, 0, legend_w, legend_h], fill=(0, 0, 0, 160))

        img_pil = img_pil.convert("RGBA")
        img_pil = Image.alpha_composite(img_pil, overlay)
        draw = ImageDraw.Draw(img_pil)
        curr_y = pad

        for name, color in detected_classes.values():
            draw.rectangle(
                [pad, curr_y + 2, pad + box_size, curr_y + 2 + box_size],
                fill=color,
                outline="white",
                width=1,
            )
            draw.text((pad + box_size + pad, curr_y), name, fill="white", font=font)
            curr_y += font_size + 4

    return img_pil.convert("RGB")
