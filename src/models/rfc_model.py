import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from src.data.german_dataset import get_german_dataframes
from src.data.preprocess_img import preprocess_image


def build_rfc(n_estimators=100):
    return RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=42)


def train_rfc(model, train_df):
    print("Wczytywanie i przetwarzanie obrazów do treningu...")
    X = preprocess_image(train_df["Path"])

    X_flat = X.reshape(X.shape[0], -1)
    y = train_df["ClassId"].values

    print("Start treningu RFC...")
    model.fit(X_flat, y)
    print("Trening zakończony.")
    return model


def evaluate_rfc(model, test_df):
    X_test = preprocess_image(test_df["Path"])
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    y_test = test_df["ClassId"].values

    predictions = model.predict(X_test_flat)
    acc = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    print(f"RFC Accuracy: {acc:.4f}")
    return acc, report


def predict_proba_rfc(model, image_path):
    X = preprocess_image(image_path)
    X_flat = X.reshape(1, -1)
    return model.predict_proba(X_flat)


def save_rfc(model, path="rfc_model.joblib"):
    joblib.dump(model, path)
    print(f"Model zapisany jako: {path}")


def load_rfc(path="rfc_model.joblib"):
    return joblib.load(path)


def run_rfc(action=None, path=None):
    full_path = None
    if path:
        if not os.path.exists("saved_models"):
            os.makedirs("saved_models")
            print("Utworzono folder saved_models")

        full_path = (
            path if os.path.dirname(path) else os.path.join("saved_models", path)
        )

    print("Pobieranie danych...")
    train_df, _, test_df = get_german_dataframes()

    if action == "load" and full_path:
        print(f"Wczytywanie modelu z {full_path}...")
        model = load_rfc(full_path)
    else:
        print("Trenowanie nowego modelu RFC...")
        model = build_rfc(n_estimators=100)
        model = train_rfc(model, train_df)

    if action == "save" and full_path:
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        save_rfc(model, full_path)

    print("\nEwaluacja modelu...")
    acc, report = evaluate_rfc(model, test_df)
    print(f"Accuracy: {acc}\nRaport klasyfikacji:\n{report}")

    print("\nTestowanie na własnym obrazie (podaj ścieżkę lub Enter by pominąć):")
    while True:
        img_path = input("Ścieżka do obrazu (enter żeby zakończyć): ").strip()
        if not img_path:
            break
        try:
            probas = predict_proba_rfc(model, img_path)
            prediction = np.argmax(probas, axis=1)[0]
            print(f"Obraz: {img_path} -> Przewidziana klasa: {prediction}")
        except Exception as e:
            print(f"Błąd dla {img_path}: {e}")

    return model
