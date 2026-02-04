import os

import numpy as np

from src.data.german_dataset import get_german_dataframes
from src.data.map_classes import to_names
from src.data.balance import oversample_dataframe


def run_model(
    model_name,
    build_fn,
    train_fn,
    evaluate_fn,
    predict_proba_fn,
    save_fn,
    load_fn,
    action=None,
    path=None,
    is_flat=True
):
    full_path = None
    if path:
        if not os.path.exists("saved_models"):
            os.makedirs("saved_models")
            print("Utworzono folder saved_models")

        full_path = (
            path if os.path.dirname(path) else os.path.join("saved_models", path)
        )

    print(f"[{model_name.upper()}] Pobieranie danych...")
    train_df, val_df, test_df = get_german_dataframes()

    train_df = oversample_dataframe(train_df)

    if action == "load" and full_path:
        print(f"[{model_name.upper()}] Wczytywanie modelu z {full_path}...")
        model = load_fn(full_path)
    else:
        print(f"[{model_name.upper()}] Trenowanie nowego modelu...")
        num_classes = 43
        input_shape = (32 * 32 * 3,) if is_flat else (32, 32, 3)
        model = build_fn(input_shape, num_classes)
        
        train_fn(model, train_df, val_df)

    if action == "save" and full_path:
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        save_fn(model, full_path)

    print(f"\n[{model_name.upper()}] Ewaluacja modelu...")
    acc, report = evaluate_fn(model, test_df)
    if report:
        print(f"\n[{model_name.upper()}] Raport klasyfikacji:\n{report}")

    print("\nTestowanie na własnym obrazie (podaj ścieżkę lub Enter by pominąć):")
    while True:
        img_path = input("Ścieżka do obrazu (enter żeby zakończyć): ").strip()
        if not img_path:
            break
        try:
            probas = predict_proba_fn(model, img_path)
            prediction = np.argmax(probas, axis=1)[0]
            
            names = to_names()
            class_name = names.get(prediction, "Nieznana klasa")
            
            print(f"Obraz: {img_path} -> Przewidziana klasa: {prediction} ({class_name})")
        except Exception as e:
            print(f"Błąd dla {img_path}: {e}")

    return model
