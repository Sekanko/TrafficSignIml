import sys

from src.models.rfc_model import run_rfc


def main():
    args = sys.argv[1:]

    if len(args) < 1:
        print("Błąd: Musisz podać nazwę modelu jako pierwszy argument.")
        print("Użycie: python main.py <model> [action] [path]")
        return

    model_name = args[0].lower()
    action = args[1].lower() if len(args) > 2 else None
    path = args[2] if len(args) > 2 else None

    match model_name:
        case "rfc":
            run_rfc(action=action, path=path)
        case _:
            print(f"Model '{model_name}' nie jest jeszcze zaimplementowany.")


if __name__ == "__main__":
    main()
