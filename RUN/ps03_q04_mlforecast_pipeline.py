"""PS03 - Pregunta 4: Pipeline MLForecast con M5 (FOODS_3, CA_1-4).
Ejecucion: python ps03_q04_mlforecast_pipeline.py
"""
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean, RollingStd
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE = Path(__file__).resolve().parent.parent / "DT"
OUTPUT_DIR = Path(__file__).resolve().parent


def prepare_m5_subset() -> pd.DataFrame:
    sales_path = BASE / "sales_train_evaluation.csv"
    calendar_path = BASE / "calendar.csv"
    prices_path = BASE / "sell_prices.csv"

    usecols = [
        "id",
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
    ] + [f"d_{i}" for i in range(1, 1914)]  # hasta d_1913

    sales = pd.read_csv(sales_path, usecols=lambda c: c in usecols)
    sales = sales[(sales["dept_id"] == "FOODS_3") & (sales["store_id"].isin(["CA_1", "CA_2", "CA_3", "CA_4"]))]

    id_vars = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    sales_long = sales.melt(id_vars=id_vars, var_name="d", value_name="y")

    calendar = pd.read_csv(calendar_path)
    prices = pd.read_csv(prices_path)
    sales_long = sales_long.merge(calendar, on="d", how="left")
    sales_long = sales_long.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")

    sales_long["ds"] = pd.to_datetime(sales_long["date"])
    sales_long = sales_long.rename(columns={"id": "unique_id"})

    sales_long["is_event"] = np.where(sales_long[["event_name_1", "event_name_2"]].notnull().any(axis=1), 1, 0)
    sales_long["day_of_week"] = sales_long["ds"].dt.dayofweek
    sales_long["is_weekend"] = (sales_long["day_of_week"] >= 5).astype(int)

    cols = [
        "unique_id",
        "ds",
        "y",
        "snap_CA",
        "sell_price",
        "day_of_week",
        "is_weekend",
        "is_event",
    ]
    sales_long = sales_long[cols].sort_values(["unique_id", "ds"]).reset_index(drop=True)
    return sales_long


def split_train_test(df: pd.DataFrame):
    train_cut = pd.to_datetime(pd.read_csv(BASE / "calendar.csv").set_index("d").loc["d_1900", "date"])
    df_train = df[df["ds"] <= train_cut]
    df_test = df[df["ds"] > train_cut]
    return df_train, df_test


def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mape = (np.abs((y_true - y_pred) / np.where(y_true == 0, np.nan, y_true))).mean()
    return mae, rmse, mape


def main():
    print(">> Preparando datos M5 (FOODS_3, CA_1-4)...")
    df = prepare_m5_subset()
    df_train, df_test = split_train_test(df)

    print(f"Train: {df_train.shape}, Test: {df_test.shape}")

    models = [
        LGBMRegressor(random_state=42, n_estimators=300, learning_rate=0.05, max_depth=-1, num_leaves=31, verbose=-1),
        XGBRegressor(random_state=42, n_estimators=400, learning_rate=0.05, max_depth=6, n_jobs=4, subsample=0.8, colsample_bytree=0.8),
    ]

    fcst = MLForecast(
        models=models,
        freq="D",
        lags=[1, 7, 14, 28],
        lag_transforms={
            1: [RollingMean(window_size=7), RollingMean(window_size=14), RollingStd(window_size=7)],
        },
        date_features=["day", "dayofyear", "weekofyear"],
        num_threads=4,
    )

    print(">> Entrenando modelos...")
    fcst.fit(df_train, id_col="unique_id", time_col="ds", target_col="y", static_features=[])

    horizon = df_test["ds"].nunique()
    future_exog = df_test[["unique_id", "ds", "snap_CA", "sell_price", "day_of_week", "is_weekend", "is_event"]]
    preds = fcst.predict(horizon, X_df=future_exog)

    eval_df = preds.merge(df_test[["unique_id", "ds", "y"]], on=["unique_id", "ds"], how="left")

    metrics = {}
    for model_name in [m.__class__.__name__ for m in models]:
        mae, rmse, mape = compute_metrics(eval_df["y"], eval_df[model_name])
        metrics[model_name] = {"MAE": mae, "RMSE": rmse, "MAPE": mape}
        print(f"{model_name}: MAE={mae:.3f}, RMSE={rmse:.3f}, MAPE={mape:.3%}")

    sample_id = eval_df["unique_id"].iloc[0]
    hist = df_train[df_train["unique_id"] == sample_id].tail(60)
    future = eval_df[eval_df["unique_id"] == sample_id]

    plt.figure(figsize=(10, 5))
    plt.plot(hist["ds"], hist["y"], label="Historico (ultimos 60d)")
    plt.plot(future["ds"], future["y"], label="Real test", marker="o")
    for model_name in metrics:
        plt.plot(future["ds"], future[model_name], label=model_name)
    plt.title(f"Prediccion horizonte 13 dias - {sample_id}")
    plt.xlabel("Fecha")
    plt.ylabel("Ventas")
    plt.legend()
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "q04_predicciones.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Grafico guardado en {plot_path}")


if __name__ == "__main__":
    main()
