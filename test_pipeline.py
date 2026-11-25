import os
import pandas as pd
from model_pipeline import evaluate_model, prepare_data, train_model


def test_prepare_and_train_shapes():
    assert os.path.exists("Churn_Modelling.csv")
    x_train, x_test, y_train, y_test = prepare_data("Churn_Modelling.csv")
    assert len(x_train) > 0 and len(x_test) > 0
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)

    model = train_model(x_train, y_train)
    acc = evaluate_model(model, x_test, y_test)
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0
