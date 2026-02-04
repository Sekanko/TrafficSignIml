from sklearn.metrics import classification_report
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from src.data.preprocess_img import preprocess_image
from src.models.run_model import run_model

def build_cnn(input_shape, num_classes, hp=None):
    data_augmentation = models.Sequential([
        layers.Rescaling(255.0),
        layers.RandomRotation(0.05), 
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.05, 0.05), 
        layers.Rescaling(1./255)
    ], name="data_augmentation")

    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(data_augmentation)

    filters_1 = hp.Int('filters_1', 16, 64, step=16) if hp else 32
    model.add(layers.Conv2D(filters_1, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    
    filters_2 = hp.Int('filters_2', 32, 128, step=32) if hp else 64
    model.add(layers.Conv2D(filters_2, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(128, (3, 3), activation="relu"))
    model.add(layers.Flatten())
    
    dense_units = hp.Int('dense_units', 64, 256, step=64) if hp else 128
    model.add(layers.Dense(dense_units, activation="relu"))
    model.add(layers.Dense(num_classes, activation="softmax"))

    lr = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4]) if hp else 0.001

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def train_cnn(model, train_df, val_df, epochs=10, batch_size=32):
    print("Przygotowanie danych treningowych i walidacyjnych dla CNN...")
    X_train = preprocess_image(train_df["Path"])
    y_train = train_df["ClassId"].values.astype('int')
    X_val = preprocess_image(val_df["Path"])
    y_val = val_df["ClassId"].values.astype('int')
    print(f"Start treningu CNN...")
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
    )
    return history

def save_cnn(model, path="cnn_model.keras"):
    model.save(path)
    print(f"Model CNN zapisany w {path}")

def load_cnn(path="cnn_model.keras"):
    return models.load_model(path)

def evaluate_cnn(model, test_df):
    X_test = preprocess_image(test_df["Path"])
    y_test = test_df["ClassId"].values.astype('int')
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"CNN Test Accuracy: {acc:.4f}")
    predictions = model.predict(X_test, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    report = classification_report(y_test, y_pred)
    return acc, report

def predict_proba_cnn(model, image_path):
    X = preprocess_image(image_path)
    return model.predict(X)

def run_cnn(action=None, path=None):
    return run_model(
        model_name="cnn",
        build_fn=build_cnn,
        train_fn=train_cnn,
        evaluate_fn=evaluate_cnn,
        predict_proba_fn=predict_proba_cnn,
        save_fn=save_cnn,
        load_fn=load_cnn,
        action=action,
        path=path,
        is_flat=False
    )