import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from src.data.preprocess_img import preprocess_image
from src.models.run_model import run_model


def build_rfc(input_shape, num_classes):
    # input_shape i num_classes ignorowane dla RFC, ale zachowane dla spójności interfejsu
    return RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=50)


def train_rfc(model, train_df, val_df=None):
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
    X_flat = X.reshape(X.shape[0], -1)
    return model.predict_proba(X_flat)


def save_rfc(model, path="rfc_model.joblib"):
    joblib.dump(model, path)
    print(f"Model zapisany jako: {path}")


def load_rfc(path="rfc_model.joblib"):
    return joblib.load(path)




def run_rfc(action=None, path=None):
    return run_model(
        model_name="rfc",
        build_fn=build_rfc,
        train_fn=train_rfc,
        evaluate_fn=evaluate_rfc,
        predict_proba_fn=predict_proba_rfc,
        save_fn=save_rfc,
        load_fn=load_rfc,
        action=action,
        path=path,
        is_flat=True
    )
