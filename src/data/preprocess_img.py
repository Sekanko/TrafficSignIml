import numpy as np
from PIL import Image


def preprocess_image(input_data, target_size=(32, 32)):
    if isinstance(input_data, str):
        paths = [input_data]
    else:
        paths = input_data

    images = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        img = img.resize(target_size)
        images.append(np.asarray(img))

    processed = np.array(images).astype("float32") / 255.0
    return processed
