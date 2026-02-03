import sys

from src.models.cnn_model import run_cnn
from src.models.mlp_model import run_mlp
from src.models.rfc_model import run_rfc


def main():
    args = sys.argv[1:]

    if len(args) < 1:
        print("Błąd: Musisz podać nazwę modelu jako pierwszy argument.")
        print("Użycie: python main.py <model> [action] [path]")
        return

    model_name = args[0].lower()

    if len(args) > 2:
        action = args[1].lower()
        path = args[2]
    else:
        action = None
        path = None

    match model_name:
        case "rfc":
            run_rfc(action=action, path=path)
        case "mlp":
            run_mlp(action=action, path=path)
        case "cnn":
            run_cnn(action=action, path=path)
        case _:
            print(f"Model '{model_name}' nie jest jeszcze zaimplementowany.")


if __name__ == "__main__":
    main()
