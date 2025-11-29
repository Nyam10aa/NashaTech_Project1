from datetime import datetime

import japanize_matplotlib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import setuptools
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             root_mean_squared_error)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from .utils.data_loader import DataClient
from .utils.jma import JmaClient


def save_feature_importance(
    model, feature_names, output_path="workdir/feature_importance.png", use_mlflow=False
):
    """
    Save XGBoost model feature importance as an image with important features on top.
    """

    # Extract raw importance dictionary
    booster = model.get_booster()
    importance_dict = booster.get_score(importance_type="gain")

    # Convert to list aligned with feature names
    importances = np.array(
        [importance_dict.get(f"f{i}", 0) for i in range(len(feature_names))]
    )

    # Sort by importance DESC so highest is first
    sorted_idx = np.argsort(importances)[::-1]

    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_importances = importances[sorted_idx]

    # Plot
    plt.figure(figsize=(8, 4))
    plt.barh(sorted_features, sorted_importances)
    plt.xlabel("Importance (Gain)")
    plt.ylabel("Feature")
    plt.title("XGBoost Feature Importance (Sorted)")
    plt.gca().invert_yaxis()  # ← IMPORTANT: flip so highest importance is on top
    plt.tight_layout()

    plt.savefig(output_path)  # , dpi=300)
    plt.close()

    print(f"Feature importance image saved to: {output_path}")

    if use_mlflow:
        mlflow.log_artifact(output_path)


def plot_actual_vs_pred(
    y_true, y_pred, output_path="workdir/actual_vs_pred.png", use_mlflow=False
):
    """
    Create and save scatter plot of actual vs predicted values for XGBoost models.

    Parameters
    ----------
    model : xgboost.XGBRegressor
        Trained XGBoost model.
    X : pandas DataFrame or numpy array
        Input features used for prediction.
    y : pandas Series or numpy array
        Ground truth (actual values).
    output_path : str
        File path to save image.
    """

    # Convert to numpy
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Plot settings
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, s=15, alpha=0.6)

    # 45-degree line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)

    # Labels
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted (XGBoost Regression)")
    plt.grid(True, linestyle="--", alpha=0.5)

    # Save
    plt.tight_layout()
    plt.savefig(output_path)  # , dpi=300)
    plt.close()

    print(f"Scatter plot saved to: {output_path}")
    if use_mlflow:
        mlflow.log_artifact(output_path)


def daily_metrics(group: pd.DataFrame) -> pd.Series:
    actual = group["actual"]
    pred = group["predicted"]

    rmse = root_mean_squared_error(actual, pred)
    mse = mean_squared_error(actual, pred)
    mae = mean_absolute_error(actual, pred)

    return pd.Series({"RMSE": rmse, "MAE": mae, "MSE": mse})


def save_actual_pred_to_excel(
    y_true, y_pred, index, output_path, daily_metric_path, use_mlflow=False
):

    # Convert to numpy
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Create DataFrame
    df_out = pd.DataFrame(
        {
            "index": np.array(index),
            "actual": y_true,
            "predicted": y_pred,
            "residual": y_true - y_pred,  # optional: error column
        }
    )

    # Save to Excel
    df_out.to_excel(output_path, index=False)

    daily_result = df_out.copy()
    daily_result["day"] = daily_result["index"].apply(lambda x: x.date())
    print(daily_result)
    print("daily_result")
    daily_result = daily_result.groupby("day").apply(daily_metrics)
    daily_result.to_excel(daily_metric_path)

    print(f"Excel saved to: {output_path}")
    if use_mlflow:
        mlflow.log_artifact(output_path)
        mlflow.log_artifact(daily_metric_path)


def extract_from_datetime(dt):
    # dt = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
    return dt.month, dt.day, dt.weekday(), dt.hour, dt.minute


def str_to_datetime(datetime_str):
    return datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")


def train(**kwargs):
    # print(args.mode)
    print(kwargs)
    print(kwargs["use_mlflow"])
    print("aa train")

    if kwargs["use_mlflow"]:
        mlflow.xgboost.autolog()

    # ---------------------------------- Sensor Data ----------------------------------
    numerical_sensors = []
    categorical_sensors = []
    if kwargs["numerical_col"] != "":
        numerical_sensors = kwargs["numerical_col"].split(",")

    if kwargs["categorical_col"] != "":
        categorical_sensors = kwargs["categorical_col"].split(",")
    target_sensor = kwargs["target_col"]

    sensor_list = numerical_sensors + categorical_sensors + [target_sensor]
    print(sensor_list)

    data_client = DataClient(kwargs["data"], sensor_list=sensor_list)

    df = data_client.get_all_data_by_sensor_list(sensor_list=sensor_list)
    # df.columns = sensor_list
    df.ffill(inplace=True)

    df[["month", "day", "weekday", "hour", "minute"]] = pd.DataFrame(
        df.index.map(extract_from_datetime).tolist(), index=df.index
    )
    print(df)

    numerical_cols = numerical_sensors + []
    categorical_cols = categorical_sensors + [
        "month",
        "day",
        "weekday",
        "hour",
        "minute",
    ]

    # ---------------------------------- JMA ----------------------------------
    JMA = JmaClient("data/外部データ/jma_札幌.csv")
    jma_categorical_features = []
    jma_numerical_features = []
    if kwargs["jma_categorical_col"] != "":
        jma_categorical_features = kwargs["jma_categorical_col"].split(",")
    if kwargs["jma_numerical_col"] != "":
        jma_numerical_features = kwargs["jma_numerical_col"].split(",")

    def get_jma_features(datetime_idx):
        out = []
        for feature in jma_categorical_features:
            out.append(JMA.get_flag(date_idx=datetime_idx.date(), col=feature))
        for feature in jma_numerical_features:
            out.append(JMA.get_value(date_idx=datetime_idx.date(), col=feature))
        return out

    df[jma_categorical_features + jma_numerical_features] = pd.DataFrame(
        df.index.map(get_jma_features).tolist(),
        index=df.index,
    )
    print(df.head(3))

    categorical_cols += jma_categorical_features
    numerical_cols += jma_numerical_features
    # ----------------------------------  ----------------------------------

    X = df.drop(target_sensor, axis=1)
    y = df[target_sensor]

    # Identify columns
    # numerical_cols = ["1286", "1287", "1289"]
    # categorical_cols = []

    # -------------------------
    # 2. Preprocessing
    # -------------------------
    preprocess = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    # -------------------------
    # 3. XGBoost model
    # -------------------------
    model = XGBRegressor(
        n_estimators=kwargs["model_n_estimators"],
        learning_rate=kwargs["model_learning_rate"],
        max_depth=kwargs["model_max_depth"],
        subsample=kwargs["model_subsample"],
        colsample_bytree=kwargs["model_colsample_bytree"],
        objective=kwargs["model_objective"],
    )

    # -------------------------
    # 4. Pipeline = preprocessing + model
    # -------------------------
    pipeline = Pipeline(steps=[("preprocess", preprocess), ("model", model)])

    # -------------------------
    # 5. Train/Test split
    # -------------------------
    X_train = X.loc[
        str_to_datetime(kwargs["train_start"]) : str_to_datetime(kwargs["train_end"]), :
    ]
    y_train = y.loc[
        str_to_datetime(kwargs["train_start"]) : str_to_datetime(kwargs["train_end"])
    ]

    X_test = X.loc[
        str_to_datetime(kwargs["test_start"]) : str_to_datetime(kwargs["test_end"]), :
    ]
    y_test = y.loc[
        str_to_datetime(kwargs["test_start"]) : str_to_datetime(kwargs["test_end"])
    ]
    print("--------------")
    print(X_train)
    print("--------------")
    print(X_test)
    print("--------------")
    print(y_train)
    print("--------------")
    print(y_test)
    print("--------------")

    # -------------------------
    # 6. Train
    # -------------------------
    pipeline.fit(X_train, y_train)

    if kwargs["use_mlflow"] and kwargs["no_save_model"]:
        # mlflow.xgboost.log_model(model,  name="model")

        model_loaded = mlflow.xgboost.load_model(
            f"runs:/{mlflow.active_run().info.run_id}/model"
        )
        pipeline_loaded = Pipeline(
            steps=[("preprocess", preprocess), ("model", model_loaded)]
        )

    # -------------------------
    # 7. Prediction & evaluation
    # -------------------------
    save_feature_importance(
        model=model, feature_names=X.columns.to_list(), use_mlflow=kwargs["use_mlflow"]
    )

    for data_type, input, actual, pipe in [
        ("train", X_train, y_train, pipeline),
        ("test", X_test, y_test, pipeline),
        # ("test_by_loaded_model", X_test, y_test, pipeline_loaded),
    ]:
        pred = pipe.predict(input)
        rmse = root_mean_squared_error(actual, pred)
        mse = mean_squared_error(actual, pred)
        mae = mean_absolute_error(actual, pred)

        print(f"{data_type} rmse", rmse)
        print(f"{data_type} mse", mse)
        print(f"{data_type} mae", mae)

        if kwargs["use_mlflow"]:
            mlflow.log_metric(f"{data_type}_rmse", rmse)
            mlflow.log_metric(f"{data_type}_mse", mse)
            mlflow.log_metric(f"{data_type}_mae", mae)
        plot_actual_vs_pred(
            y_true=actual,
            y_pred=pred,
            output_path=f"workdir/{data_type}_scatter.png",
            use_mlflow=kwargs["use_mlflow"],
        )
        save_actual_pred_to_excel(
            y_true=actual,
            y_pred=pred,
            index=input.index,
            output_path=f"workdir/{data_type}.xlsx",
            daily_metric_path=f"workdir/{data_type}_daily_metrics.xlsx",
            use_mlflow=kwargs["use_mlflow"],
        )
