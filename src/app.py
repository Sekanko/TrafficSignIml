import os
import sys
import shutil
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# --- DIAGNOSTYKA ŚCIEŻEK ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR) # Wychodzimy z src/ do głównego folderu

sys.path.append(ROOT_DIR)

# Sprawdzenie co myśli skrypt
print(f"--- DEBUG START ---")
print(f"Lokalizacja app.py: {CURRENT_DIR}")
print(f"Katalog główny (ROOT): {ROOT_DIR}")
MODELS_DIR = os.path.join(ROOT_DIR, "saved_models")
print(f"Szukam modeli w: {MODELS_DIR}")

if os.path.exists(MODELS_DIR):
    print(f"Pliki w folderze saved_models: {os.listdir(MODELS_DIR)}")
else:
    print(f"BŁĄD: Folder {MODELS_DIR} nie istnieje!")
    # Spróbujmy go stworzyć, żeby nie wywaliło reszty
    os.makedirs(MODELS_DIR, exist_ok=True)
print(f"--- DEBUG END ---")

from src.models.rfc_model import load_rfc, predict_proba_rfc
from src.models.mlp_model import load_mlp, predict_proba_mlp
from src.models.cnn_model import load_cnn, predict_proba_cnn
from src.data.map_classes import to_names

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")

if not os.path.exists(FRONTEND_DIR):
    print(f"CRITICAL: Nie znaleziono folderu frontend w: {FRONTEND_DIR}")

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
app.mount("/css", StaticFiles(directory=os.path.join(FRONTEND_DIR, "css")), name="css")
app.mount("/js", StaticFiles(directory=os.path.join(FRONTEND_DIR, "js")), name="js")

MODELS = {}
NAMES = to_names()

def get_model(model_name):
    # Jeśli model już załadowany, zwróć go
    if model_name in MODELS:
        return MODELS[model_name]
    
    print(f"\n[INFO] Próba załadowania modelu: {model_name}")
    
    path = ""
    model = None
    predict_fn = None

    if model_name == 'rfc':
        path = os.path.join(MODELS_DIR, "rfc_best.joblib")
        predict_fn = predict_proba_rfc
        loader = load_rfc
    elif model_name == 'mlp':
        path = os.path.join(MODELS_DIR, "mlp_best.keras")
        predict_fn = predict_proba_mlp
        loader = load_mlp
    elif model_name == 'cnn':
        path = os.path.join(MODELS_DIR, "cnn_best.keras")
        predict_fn = predict_proba_cnn
        loader = load_cnn
    else:
        raise ValueError(f"Nieznany typ modelu: {model_name}")
    
    print(f"[INFO] Szukam pliku: {path}")
    
    if not os.path.exists(path):
        available = os.listdir(MODELS_DIR) if os.path.exists(MODELS_DIR) else "Brak folderu"
        error_msg = f"Plik modelu nie istnieje: {path}. Dostępne pliki: {available}"
        print(f"[ERROR] {error_msg}")
        raise FileNotFoundError(error_msg)

    try:
        model = loader(path)
        print(f"[SUCCESS] Model {model_name} załadowany poprawnie.")
    except Exception as e:
        print(f"[ERROR] Błąd podczas ładowania pliku modelu: {e}")
        raise e

    MODELS[model_name] = (model, predict_fn)
    return model, predict_fn

@app.get("/")
async def read_index():
    from fastapi.responses import FileResponse
    return FileResponse(os.path.join(FRONTEND_DIR, 'index.html'))

@app.get("/")
async def read_index():
    from fastapi.responses import FileResponse
    return FileResponse(os.path.join(FRONTEND_DIR, 'index.html'))

@app.get("/index.html")
async def read_index_explicit():
    from fastapi.responses import FileResponse
    return FileResponse(os.path.join(FRONTEND_DIR, 'index.html'))

@app.get("/classify.html")
async def read_classify():
    from fastapi.responses import FileResponse
    return FileResponse(os.path.join(FRONTEND_DIR, 'classify.html'))

@app.post("/predict")
async def predict(file: UploadFile = File(...), model: str = Form(...)):
    temp_dir = os.path.join(ROOT_DIR, "temp_uploads")
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Tutaj następuje kluczowy moment - ładowanie modelu
        loaded_model, predict_fn = get_model(model)
        
        probas = predict_fn(loaded_model, file_path)
        prediction_id = int(np.argmax(probas, axis=1)[0])
        confidence = float(np.max(probas))
        class_name = NAMES.get(prediction_id, "Nieznana klasa")
        
        return {
            "class_id": prediction_id,
            "class_name": class_name,
            "confidence": confidence
        }
    except FileNotFoundError as e:
        return {"error": str(e)}, 404
    except Exception as e:
        print(f"[CRITICAL ERROR] Wystąpił błąd w trakcie predykcji: {e}")
        return {"error": str(e)}, 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)