import os
import shutil

import kagglehub
import tensorflow as tf
import yaml
from ultralytics import YOLO

from src.data.prepare_yolo_data import prepare_yolo_dataset


def load_yolo_model(path="yolov8n.pt"):
    model = YOLO(path)
    return model


def yolo_evaluation(model):
    print("Starting YOLO Validation...")
    results = model.val()
    return results


def train_yolo_model(path, model, epochs=10, img_size=640, batch_size=16):
    yaml_filename = "data.yaml"

    data_config = {
        "path": os.path.abspath(path),
        "train": "train/images",
        "val": "val/images",
        "nc": 1,
        "names": ["traffic_sign"],
    }

    with open(yaml_filename, "w") as f:
        yaml.dump(data_config, f)

    model.train(
        data=yaml_filename,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device="cpu" if not tf.test.is_built_with_cuda() else "cuda",
    )

    return model


def yolo_prediction(model, img):
    conf_threshold = 0.6

    results = model.predict(
        source=img,
        conf=conf_threshold,
        device="cuda" if tf.test.is_built_with_cuda() else "cpu",
        verbose=True,
        save=True,
    )

    return results


def run_yolo(action=None, path=None):
    full_path = None
    if path:
        if not os.path.exists("saved_models"):
            os.makedirs("saved_models")
            print("Utworzono folder saved_models")

        full_path = (
            path if os.path.dirname(path) else os.path.join("saved_models", path)
        )

    if action == "load" and full_path and os.path.exists(full_path):
        print(f">> Wczytywanie modelu z {full_path}...")
        model = load_yolo_model(full_path)
    else:
        print(">> Przygotowanie danych...")
        kaggle_path = kagglehub.dataset_download(
            "chriskjm/polish-traffic-signs-dataset"
        )
        detection_data_path = os.path.join(kaggle_path, "detection")
        dataset_path = prepare_yolo_dataset(detection_data_path)
        model = load_yolo_model("yolov8n.pt")

        print(">> Trenowanie modelu...")
        train_yolo_model(dataset_path, model)

    if action == "save" and full_path:
        best_model_path = "runs/detect/train/weights/best.pt"
        if os.path.exists(best_model_path):
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            shutil.copy(best_model_path, full_path)
            print(f">> Model zapisany w {full_path}")
        else:
            model.save(full_path)
            print(f">> Model zapisany w {full_path}")

    print("\nTestowanie na własnym obrazie (podaj ścieżkę lub Enter by pominąć):")
    while True:
        img_path = input("Ścieżka do obrazu (enter żeby zakończyć): ").strip()
        if not img_path:
            break
        try:
            yolo_prediction(model, img_path)
        except Exception as e:
            print(f"Błąd dla {img_path}: {e}")

    return model
