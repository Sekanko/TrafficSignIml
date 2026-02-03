from sklearn.metrics import classification_report
import numpy as np
from tensorflow.keras import layers, models

from src.data.preprocess_img import preprocess_image
from src.models.run_model import run_model


def build_mlp(input_shape, num_classes):
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(64, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def train_mlp(model, train_df, val_df, epochs=10, batch_size=32):
    print("Przygotowanie danych treningowych i walidacyjnych...")

    X_train = preprocess_image(train_df["Path"])
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    y_train = train_df["ClassId"].values

    X_val = preprocess_image(val_df["Path"])
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    y_val = val_df["ClassId"].values

    print(f"Start treningu MLP (Input size: {X_train_flat.shape[1]})...")
    history = model.fit(
        X_train_flat,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val_flat, y_val)
    )
    return history


def evaluate_mlp(model, test_df):
    X_test = preprocess_image(test_df["Path"])
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    y_test = test_df["ClassId"].values

    loss, acc = model.evaluate(X_test_flat, y_test, verbose=0)
    print(f"MLP Test Accuracy: {acc:.4f}")

    predictions = model.predict(X_test_flat, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    report = classification_report(y_test, y_pred)

    return acc, report


def predict_proba_mlp(model, image_path):
    X = preprocess_image(image_path)
    X_flat = X.reshape(X.shape[0], -1)
    return model.predict(X_flat)


def save_mlp(model, path="mlp_model.keras"):
    model.save(path)
    print(f"Model MLP zapisany w {path}")


def load_mlp(path="mlp_model.keras"):
    return models.load_model(path)




def run_mlp(action=None, path=None):
    return run_model(
        model_name="mlp",
        build_fn=build_mlp,
        train_fn=train_mlp,
        evaluate_fn=evaluate_mlp,
        predict_proba_fn=predict_proba_mlp,
        save_fn=save_mlp,
        load_fn=load_mlp,
        action=action,
        path=path,
        is_flat=True
    )
