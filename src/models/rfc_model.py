import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def build_rfc(n_estimators=100):
    return RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1)

def train_rfc(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def save_rfc(model, path="rfc_model.pkl"):
    joblib.dump(model, path)
    print(f"Model RFC zapisany w {path}")

def evaluate_rfc(model, X_test, y_test):
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    print(f"RFC Accuracy: {acc:.4f}")
    return acc, report

def predict_proba_rfc(model, X_input):
    return model.predict_proba(X_input)

def load_rfc(path="rfc_model.pkl"):
    return joblib.load(path)