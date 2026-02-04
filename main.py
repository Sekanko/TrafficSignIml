import sys
from src.models.cnn_model import run_cnn
from src.models.mlp_model import run_mlp
from src.models.rfc_model import run_rfc

try:
    from src.models.tuner import run_tuner
except ImportError:
    run_tuner = None

def main():
    args = sys.argv[1:]
    if len(args) < 1:
        print("Błąd: Podaj model.")
        return

    model_name = args[0].lower()
    action = args[1].lower() if len(args) > 1 else None
    path = args[2] if len(args) > 2 else None

    if action == "tune":
        if run_tuner:
            run_tuner(model_name)
        else:
            print("Brak modułu tunera.")
        return

    match model_name:
        case "rfc":
            run_rfc(action=action, path=path)
        case "mlp":
            run_mlp(action=action, path=path)
        case "cnn":
            run_cnn(action=action, path=path)
        case _:
            print(f"Model '{model_name}' nieznany.")

if __name__ == "__main__":
    main()