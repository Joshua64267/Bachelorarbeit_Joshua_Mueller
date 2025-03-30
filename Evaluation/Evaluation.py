import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


class Evaluation:
    @staticmethod
    def r2(data: pd.DataFrame) -> float:
        y_true = data["Tatsächliche Zeit (min)"]
        y_pred = data["Predicted Zeit (min)"]

        return r2_score(y_true, y_pred)

    @staticmethod
    def mae(data: pd.DataFrame) -> float:
        y_true = data["Tatsächliche Zeit (min)"]
        y_pred = data["Predicted Zeit (min)"]
        return mean_absolute_error(y_true, y_pred)

    @staticmethod
    def mse(data: pd.DataFrame) -> float:
        y_true = data["Tatsächliche Zeit (min)"]
        y_pred = data["Predicted Zeit (min)"]
        return mean_squared_error(y_true, y_pred)

    @staticmethod
    def rmse(data: pd.DataFrame) -> float:
        return Evaluation.mse(data) ** 0.5