import argparse
from model_pipeline import (
    evaluate_model,
    load_model,
    predict,
    prepare_data,
    save_model,
    train_model,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true", help="Prepare the data")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    parser.add_argument("--save", action="store_true", help="Save the model")
    parser.add_argument("--load", action="store_true", help="Load the model")
    parser.add_argument("--predict", action="store_true", help="Run predictions")

    args = parser.parse_args()

    if args.prepare:
        prepare_data()
        print("Data prepared.")

    if args.train:
        x_train, x_test, y_train, y_test = prepare_data()
        model = train_model(x_train, y_train)
        save_model(model)
        print("Model trained and saved.")

    if args.evaluate:
        x_train, x_test, y_train, y_test = prepare_data()
        model = load_model()
        evaluate_model(model, x_test, y_test)

    if args.predict:
        x_train, x_test, y_train, y_test = prepare_data()
        predict(x_test)

    if args.save and not args.train:
        x_train, x_test, y_train, y_test = prepare_data()
        model = train_model(x_train, y_train)
        save_model(model)

    if args.load and not args.evaluate:
        load_model()


if __name__ == "__main__":
    main()
