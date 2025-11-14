from datetime import datetime

import japanize_matplotlib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import setuptools
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from .utils.data_loader import DataClient


def save_feature_importance(model, feature_names, output_path="feature_importance.png"):
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
    plt.figure(figsize=(10, 6))
    plt.barh(sorted_features, sorted_importances)
    plt.xlabel("Importance (Gain)")
    plt.ylabel("Feature")
    plt.title("XGBoost Feature Importance (Sorted)")
    plt.gca().invert_yaxis()  # ← IMPORTANT: flip so highest importance is on top
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Feature importance image saved to: {output_path}")


def plot_actual_vs_pred(y_true, y_pred, output_path="actual_vs_pred.png"):
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
    plt.figure(figsize=(7, 7))
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
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Scatter plot saved to: {output_path}")


def extract_from_datetime(datetime_str):
    dt = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")

    # month = dt.month
    # day = dt.day
    # weekday = dt.weekday()     # Monday=0, Sunday=6
    # hour = dt.hour
    # minute = dt.minute
    return dt.month, dt.day, dt.weekday(), dt.hour, dt.minute


def train(**kwargs):
    # print(args.mode)
    print(kwargs)
    print(kwargs["use_mlflow"])
    print("aa train")

    numerical_sensors = [
        "外気温度",
        "外気湿度",
    ]  # "供給先A温水往圧力", "供給先A温水往温度", "供給先A温水還圧力"]
    categorical_sensors = []
    target_sensor = "供給先A冷水熱量"

    sensor_list = numerical_sensors + categorical_sensors + [target_sensor]

    data_client = DataClient(kwargs["data"], sensor_list=sensor_list)

    df = data_client.get_all_data_by_sensor_list(sensor_list=sensor_list)
    df.columns = sensor_list
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
        n_estimators=500,
        learning_rate=0.01,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.8,
        objective="reg:squarederror",
    )

    # -------------------------
    # 4. Pipeline = preprocessing + model
    # -------------------------
    pipeline = Pipeline(steps=[("preprocess", preprocess), ("model", model)])

    # -------------------------
    # 5. Train/Test split
    # -------------------------
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=42
    # )

    mid = X.shape[0] // 2
    X_train = X.iloc[:mid, :]  # 上半分
    y_train = y.iloc[:mid]

    X_test = X.iloc[mid:, :]  # 下半分
    y_test = y.iloc[mid:]
    # X_test = X_train
    # y_test = y_train
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

    # -------------------------
    # 7. Prediction & evaluation
    # -------------------------
    y_pred = pipeline.predict(X_test)

    rmse = root_mean_squared_error(y_test, y_pred)

    print("Predictions:", y_pred)
    print("RMSE:", rmse)

    save_feature_importance(model=model, feature_names=X.columns.to_list())

    plot_actual_vs_pred(
        y_true=y_train,
        y_pred=pipeline.predict(X_train),
        output_path="train_scatter.png",
    )
    plot_actual_vs_pred(
        y_true=y_test, y_pred=pipeline.predict(X_test), output_path="test_scatter.png"
    )
