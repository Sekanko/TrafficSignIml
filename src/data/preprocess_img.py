import numpy as np
from PIL import Image


def crop_image_by_roi(img, x1, y1, x2, y2):
    if x1 == -1 or x2 <= x1 or y2 <= y1:
        return img

    w, h = img.size

    if w < x2 or h < y2:
        return img

    return img.crop((x1, y1, x2, y2))


def preprocess_image(input_data, target_size=(32, 32)):
    if isinstance(input_data, str):
        paths = [input_data]
    else:
        paths = input_data

    images = []
    for path in paths:
        try:
            img = Image.open(path).convert("RGB")
            img = img.resize(target_size)
            images.append(np.asarray(img))
        except Exception as e:
            print(f"Błąd przy przetwarzaniu {path}: {e}")
            images.append(np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8))

    processed = np.array(images).astype("float32") / 255.0
    return processed
