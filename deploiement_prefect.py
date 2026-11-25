from datetime import timedelta
from pipeline_prefect import all_flow, code_flow, evaluate_flow, train_flow


def main():
    all_deploy = all_flow.to_deployment(
        name="ml-pipeline-all",
        interval=timedelta(days=1),
        tags=["full-pipeline", "mlops"],
    )
    train_deploy = train_flow.to_deployment(
        name="ml-pipeline-train",
        interval=timedelta(days=1),
        tags=["training", "mlops"],
    )
    evaluate_deploy = evaluate_flow.to_deployment(
        name="ml-pipeline-evaluate",
        tags=["evaluation", "mlops"],
    )
    code_deploy = code_flow.to_deployment(
        name="ml-pipeline-code",
        tags=["quality", "mlops"],
    )

    from prefect import serve

    serve(all_deploy, train_deploy, evaluate_deploy, code_deploy)


if __name__ == "__main__":
    main()
