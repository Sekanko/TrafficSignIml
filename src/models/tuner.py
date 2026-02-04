import os
import joblib
import numpy as np
import tensorflow as tf
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from src.models.mlp_model import build_mlp
from src.models.cnn_model import build_cnn
from src.data.ensure import get_merged_data
from src.data.balance import oversample_dataframe
from src.data.preprocess_img import preprocess_image

try:
    import keras_tuner as kt
except ImportError:
    kt = None
    print("Brak biblioteki keras-tuner.")

def get_data_and_preprocess(is_flat=False):
    print("Pobieranie i przygotowanie pełnych danych (PL + DE) do tuningu...")
    train_df, val_df, _ = get_merged_data()
    train_df = oversample_dataframe(train_df)
    
    # Używamy preprocess_img który tylko ładuje i skaluje (ensure.py wcześniej wyciął znaki)
    X_train = preprocess_image(train_df["Path"])
    y_train = train_df["ClassId"].values.astype('int')
    
    X_val = preprocess_image(val_df["Path"])
    y_val = val_df["ClassId"].values.astype('int')

    if is_flat:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_val = X_val.reshape(X_val.shape[0], -1)

    return X_train, y_train, X_val, y_val

def tune_rfc():
    print("START TUNINGU: RANDOM FOREST")
    X_train, y_train, X_val, y_val = get_data_and_preprocess(is_flat=True)
    X_full = np.concatenate((X_train, X_val))
    y_full = np.concatenate((y_train, y_val))
    
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    
    rfc = RandomForestClassifier(random_state=50, n_jobs=-1)
    # n_jobs=-1 wykorzystuje wszystkie rdzenie procesora
    search = RandomizedSearchCV(rfc, param_dist, n_iter=10, cv=3, verbose=2, random_state=42, n_jobs=-1)
    
    search.fit(X_full, y_full)
    
    best_model = search.best_estimator_
    save_path = "saved_models/rfc_best.joblib"
    os.makedirs("saved_models", exist_ok=True)
    joblib.dump(best_model, save_path)
    print(f"Zapisano RFC: {save_path} (Acc: {search.best_score_:.4f})")

def tune_mlp():
    if not kt: return
    print("START TUNINGU: MLP")
    X_train, y_train, X_val, y_val = get_data_and_preprocess(is_flat=True)
    input_shape = (X_train.shape[1],)
    
    tuner = kt.Hyperband(
        lambda hp: build_mlp(input_shape, 43, hp), # Przekazujemy funkcję budującą z parametrami hp
        objective='val_accuracy',
        max_epochs=15,
        factor=3,
        directory='kt_dir',
        project_name='mlp_tuning',
        overwrite=True  # <--- TO NAPRAWIA PROBLEM (Zawsze startuje od nowa)
    )

    tuner.search(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=32)
    
    # Pobieramy najlepszy model
    best_model = tuner.get_best_models(num_models=1)[0]
    os.makedirs("saved_models", exist_ok=True)
    
    # Zapisujemy bez kompilacji, żeby uniknąć problemów z optimizerem przy wczytywaniu
    best_model.save("saved_models/mlp_best.keras") 
    print("Zapisano: saved_models/mlp_best.keras")

def tune_cnn():
    if not kt: return
    print("START TUNINGU: CNN")
    X_train, y_train, X_val, y_val = get_data_and_preprocess(is_flat=False)
    input_shape = (32, 32, 3)
    
    tuner = kt.RandomSearch(
        lambda hp: build_cnn(input_shape, 43, hp),
        objective='val_accuracy',
        max_trials=10,
        directory='kt_dir',
        project_name='cnn_tuning',
        overwrite=True # <--- TO NAPRAWIA PROBLEM
    )

    tuner.search(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=32)
    
    best_model = tuner.get_best_models(num_models=1)[0]
    os.makedirs("saved_models", exist_ok=True)
    best_model.save("saved_models/cnn_best.keras")
    print("Zapisano: saved_models/cnn_best.keras")

def run_tuner(model_name):
    if model_name == 'rfc': tune_rfc()
    elif model_name == 'mlp': tune_mlp()
    elif model_name == 'cnn': tune_cnn()
    else: print(f"Nieznany model: {model_name}")