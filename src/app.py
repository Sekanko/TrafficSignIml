import os
import shutil
import sys
from io import BytesIO

import numpy as np
import uvicorn
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(ROOT_DIR)

from src.models.rfc_model import load_rfc, predict_proba_rfc
from src.models.mlp_model import load_mlp, predict_proba_mlp
from src.models.cnn_model import load_cnn, predict_proba_cnn
from src.models.yolo_model import load_yolo_model
from src.detect_and_classify import detect_and_classify
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
MODELS_DIR = os.path.join(ROOT_DIR, "saved_models")

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
app.mount("/css", StaticFiles(directory=os.path.join(FRONTEND_DIR, "css")), name="css")
app.mount("/js", StaticFiles(directory=os.path.join(FRONTEND_DIR, "js")), name="js")

MODELS = {}
YOLO_MODEL = None
NAMES = to_names()


def get_yolo():
    global YOLO_MODEL
    if YOLO_MODEL is None:
        best_path = os.path.join(MODELS_DIR, "yolo_best.pt")
        base_path = "yolov8n.pt"
        load_path = best_path if os.path.exists(best_path) else base_path
        YOLO_MODEL = load_yolo_model(load_path)
    return YOLO_MODEL


def get_model(model_name):
    if model_name in MODELS:
        return MODELS[model_name]

    path = ""
    model = None
    predict_fn = None
    loader = None

    if model_name == "rfc":
        path = os.path.join(MODELS_DIR, "rfc_best.joblib")
        predict_fn = predict_proba_rfc
        loader = load_rfc
    elif model_name == "mlp":
        path = os.path.join(MODELS_DIR, "mlp_best.keras")
        predict_fn = predict_proba_mlp
        loader = load_mlp
    elif model_name == "cnn":
        path = os.path.join(MODELS_DIR, "cnn_best.keras")
        predict_fn = predict_proba_cnn
        loader = load_cnn
    else:
        raise ValueError(f"Unknown model: {model_name}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    model = loader(path)
    MODELS[model_name] = (model, predict_fn)
    return model, predict_fn


@app.get("/")
async def read_index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/index.html")
async def read_index_explicit():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/classify.html")
async def read_classify():
    return FileResponse(os.path.join(FRONTEND_DIR, "classify.html"))


@app.post("/predict")
async def predict(file: UploadFile = File(...), model: str = Form(...)):
    temp_dir = os.path.join(ROOT_DIR, "temp_uploads")
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        loaded_model, predict_fn = get_model(model)
        probas = predict_fn(loaded_model, file_path)
        prediction_id = int(np.argmax(probas, axis=1)[0])
        confidence = float(np.max(probas))
        class_name = NAMES.get(prediction_id, "Unknown class")

        return {
            "class_id": prediction_id,
            "class_name": class_name,
            "confidence": confidence,
        }
    except Exception as e:
        return {"error": str(e)}, 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@app.post("/detect")
async def detect(file: UploadFile = File(...), model: str = Form(...)):
    try:
        yolo = get_yolo()
        clf_model, _ = get_model(model)

        img_data = await file.read()
        img = Image.open(BytesIO(img_data)).convert("RGB")

        result_img = detect_and_classify(model, clf_model, yolo, img)

        buf = BytesIO()
        result_img.save(buf, format="PNG")
        buf.seek(0)

        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        return {"error": str(e)}, 500


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
