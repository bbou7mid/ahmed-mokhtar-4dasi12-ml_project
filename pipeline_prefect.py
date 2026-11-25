import argparse
import shutil
import subprocess
import sys
from prefect import flow, task
from main import (
    evaluate_model as main_evaluate,
    load_model as main_load,
    prepare_data as main_prepare,
    save_model as main_save,
    train_model as main_train,
)

TARGET_FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]


@task
def install_dependencies():
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
        check=True,
    )


@task
def format_code():
    subprocess.run(["black", *TARGET_FILES], check=True)


@task
def check_code_quality():
    subprocess.run(["flake8", *TARGET_FILES], check=True)


@task
def check_code_security():
    bandit_bin = shutil.which("bandit") or "bandit"
    subprocess.run([bandit_bin, "-ll", "-r", *TARGET_FILES], check=True)


@task
def run_unit_tests():
    subprocess.run(["pytest", "-q"], check=True)


@task
def prepare_data_task():
    return main_prepare()


@task
def train_model_task(x_train, y_train):
    return main_train(x_train, y_train)


@task
def save_model_task(model):
    main_save(model)


@task
def load_model_task():
    return main_load()


@task
def evaluate_model_task(model, x_test, y_test):
    return main_evaluate(model, x_test, y_test)


@flow(name="code")
def code_flow():
    format_code()
    check_code_quality()
    check_code_security()
    run_unit_tests()


@flow(name="train")
def train_flow():
    x_train, _, y_train, _ = prepare_data_task()
    model = train_model_task(x_train, y_train)
    save_model_task(model)


@flow(name="evaluate")
def evaluate_flow():
    _, x_test, _, y_test = prepare_data_task()
    model = load_model_task()
    evaluate_model_task(model, x_test, y_test)


@flow(name="all")
def all_flow():
    install_dependencies()
    format_code()
    check_code_quality()
    check_code_security()
    run_unit_tests()
    x_train, x_test, y_train, y_test = prepare_data_task()
    model = train_model_task(x_train, y_train)
    save_model_task(model)
    evaluate_model_task(model, x_test, y_test)


def main():
    parser = argparse.ArgumentParser(description="Run Prefect ML flows")
    parser.add_argument(
        "--flow",
        required=True,
        choices=["all", "train", "entrainement", "evaluate", "code"],
        help="Flow name to execute",
    )
    args = parser.parse_args()

    if args.flow == "all":
        all_flow()
    elif args.flow in ("train", "entrainement"):
        train_flow()
    elif args.flow == "evaluate":
        evaluate_flow()
    elif args.flow == "code":
        code_flow()


if __name__ == "__main__":
    main()
