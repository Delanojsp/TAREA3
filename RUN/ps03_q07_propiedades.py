"""PS03 - Pregunta 7: Prediccion de precios de propiedades.
Ejecucion: python ps03_q07_propiedades.py --input DT/precios_propiedades.xlsx --group-id X
"""
from pathlib import Path
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import numpy as np


def build_parser():
    parser = argparse.ArgumentParser(description="Predice precios UF y genera submission.")
    parser.add_argument("--input", default="DT/Precios_PS03.xlsx", help="Ruta al archivo XLSX de entrada")
    parser.add_argument("--group-id", default="X", help="Numero de grupo para el nombre de salida")
    parser.add_argument("--output-dir", default="RUN", help="Directorio de salida")
    return parser


def load_data(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"No se encontro el archivo {path}. Cargalo en la carpeta DT.")
    train_df = pd.read_excel(path, sheet_name="Estudiantes-train-test")
    target_df = pd.read_excel(path, sheet_name="Estudiantes-target")
    return train_df, target_df


def make_pipeline(train_df: pd.DataFrame):
    target_col = "precio_uf"
    drop_cols = {target_col, "titulo_propiedad", "id", "Id"}
    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    cat_cols = [c for c in feature_cols if train_df[c].dtype == "object"]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ]
    )

    model = HistGradientBoostingRegressor(random_state=42, learning_rate=0.05, max_depth=10, max_iter=400)
    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    return pipe, feature_cols


def find_lat_lon_cols(df: pd.DataFrame):
    lat_candidates = [c for c in df.columns if c.lower() in {"lat", "latitude", "latitud"}]
    lon_candidates = [c for c in df.columns if c.lower() in {"lon", "longitude", "longitud", "lng"}]
    lat_col = lat_candidates[0] if lat_candidates else None
    lon_col = lon_candidates[0] if lon_candidates else None
    return lat_col, lon_col


def find_zone_col(df: pd.DataFrame):
    candidates = ["comuna", "barrio", "zona", "district", "city", "region"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def build_zone_from_coords(df: pd.DataFrame, lat_col: str, lon_col: str, decimals: int = 2):
    """Crea una etiqueta de zona a partir de lat/lon redondeados para agrupar."""
    lat_round = df[lat_col].round(decimals)
    lon_round = df[lon_col].round(decimals)
    return ("cell_" + lat_round.astype(str) + "_" + lon_round.astype(str))


def main():
    args = build_parser().parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, target_df = load_data(input_path)
    pipe, feature_cols = make_pipeline(train_df)

    X = train_df[feature_cols]
    y = train_df["precio_uf"]

    # pequeña validacion interna
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
    pipe.fit(X_train, y_train)
    val_pred = pipe.predict(X_val)
    mape = mean_absolute_percentage_error(y_val, val_pred)
    print(f"MAPE validacion interna: {mape:.3%}")

    pipe.fit(X, y)
    preds = pipe.predict(target_df[feature_cols])

    submission = pd.DataFrame({
        "titulo_propiedad": target_df["titulo_propiedad"],
        "precio_uf": preds,
    })

    output_file = output_dir / f"Grupo_{args.group_id}_PS03Q07.xlsx"
    submission.to_excel(output_file, index=False)
    print(f"Archivo generado: {output_file}")

    # Guardar version extendida con coordenadas si existen
    lat_col, lon_col = find_lat_lon_cols(target_df)
    extended_path = None
    if lat_col and lon_col:
        submission_ext = submission.copy()
        submission_ext[lat_col] = target_df[lat_col]
        submission_ext[lon_col] = target_df[lon_col]
        extended_path = output_dir / f"Grupo_{args.group_id}_PS03Q07_con_coords.xlsx"
        submission_ext.to_excel(extended_path, index=False)
        print(f"Archivo con coordenadas: {extended_path}")

        # Mapa: top 10 zonas más valorizadas. Si no hay zona, se crea una grilla por coordenadas redondeadas.
        zone_col = find_zone_col(target_df)
        target_for_map = target_df.copy()
        if not zone_col:
            target_for_map["zona_grid"] = build_zone_from_coords(target_for_map, lat_col, lon_col, decimals=2)
            zone_col = "zona_grid"

        grouped = submission_ext.join(target_for_map[[zone_col]])
        top_zones = grouped.groupby(zone_col)["precio_uf"].mean().sort_values(ascending=False).head(10).reset_index()
        top_merged = top_zones.merge(target_for_map[[zone_col, lat_col, lon_col]], on=zone_col, how="left").dropna(subset=[lat_col, lon_col])
        # Drop duplicates keeping first coord per zone
        top_merged = top_merged.drop_duplicates(subset=[zone_col])
        plt.figure(figsize=(8, 6))
        sc = plt.scatter(top_merged[lon_col], top_merged[lat_col], c=top_merged["precio_uf"], cmap="plasma", s=120, edgecolor="k")
        for _, row in top_merged.iterrows():
            plt.text(row[lon_col], row[lat_col], row[zone_col], fontsize=8, ha="left", va="bottom")
        plt.colorbar(sc, label="Precio UF promedio")
        plt.title("Top 10 zonas mas valorizadas (prediccion)")
        plt.xlabel("Longitud")
        plt.ylabel("Latitud")
        plt.tight_layout()
        map_path = output_dir / f"Grupo_{args.group_id}_PS03Q07_mapa_top10.png"
        plt.savefig(map_path, dpi=150)
        plt.close()
        print(f"Mapa generado: {map_path}")
    else:
        print("No se hallaron columnas lat/lon en el dataset; no se genera archivo extendido ni mapa.")


if __name__ == "__main__":
    main()
