"""PS03 - Pregunta 5: Elasticidad de precios con Online Retail II.
Ejecucion: python ps03_q05_pricing_elasticidad.py
"""
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

BASE = Path(__file__).resolve().parent.parent / "DT" / "online_retail_II.csv"


def preparar_datos():
    df = pd.read_csv(BASE)
    df = df[df["Country"] == "United Kingdom"]
    df = df[~df["Invoice"].astype(str).str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0) & (df["Customer ID"].notnull())]

    top_codes = df.groupby("StockCode")["Quantity"].sum().nlargest(10).index.tolist()
    df_top = df[df["StockCode"].isin(top_codes)].copy()

    # Descripcion por producto (primera encontrada)
    descriptions = df_top.groupby("StockCode")["Description"].agg(lambda s: s.dropna().iloc[0] if len(s.dropna()) else "NA")

    df_top["InvoiceDate"] = pd.to_datetime(df_top["InvoiceDate"])
    df_top["Week"] = df_top["InvoiceDate"].dt.to_period("W").apply(lambda r: r.start_time)
    agg = df_top.groupby(["StockCode", "Week"]).agg(
        quantity_total=("Quantity", "sum"),
        price_promedio=("Price", "mean"),
        n_transacciones=("Invoice", "nunique"),
    ).reset_index()

    agg = agg[(agg["quantity_total"] > 0) & (agg["price_promedio"] > 0)]
    agg["ln_Q"] = np.log(agg["quantity_total"])
    agg["ln_P"] = np.log(agg["price_promedio"])
    agg["Month"] = agg["Week"].dt.month

    return agg, top_codes, descriptions


def elasticidad_individual(agg: pd.DataFrame, top_codes, descriptions):
    rows = []
    for code in top_codes:
        df_prod = agg[agg["StockCode"] == code]
        if len(df_prod) < 5:
            continue
        model = smf.ols("ln_Q ~ ln_P", data=df_prod).fit()
        beta = model.params.get("ln_P", np.nan)
        pval = model.pvalues.get("ln_P", np.nan)
        r2 = model.rsquared
        rows.append({
            "StockCode": code,
            "Description": descriptions.get(code, ""),
            "beta": beta,
            "p_value": pval,
            "R2": r2,
        })
    results = pd.DataFrame(rows)
    results["significativo"] = results["p_value"] < 0.05
    results["elastico"] = results["beta"].abs() > 1
    return results


def modelo_panel(agg: pd.DataFrame):
    model = smf.ols("ln_Q ~ ln_P + C(StockCode) + C(Month)", data=agg).fit()
    beta_panel = model.params.get("ln_P", np.nan)
    return model, beta_panel


def main():
    print(">> Preparando datos...")
    agg, top_codes, descriptions = preparar_datos()
    print(f"Top 10 productos por cantidad: {top_codes}")
    print(agg.groupby("StockCode")["quantity_total"].sum().reset_index().sort_values("quantity_total", ascending=False).head(10))

    print("\n>> Elasticidades individuales (log-log)...")
    ind = elasticidad_individual(agg, top_codes, descriptions)
    print(ind)
    n_sig = ind["significativo"].sum()
    n_elast = ind[ind["significativo"]]["elastico"].sum()
    n_inel = n_sig - n_elast
    print(f"Significativos p<0.05: {n_sig}; elasticos: {n_elast}; inelasticos: {n_inel}")

    print("\n>> Modelo pooled con efectos de producto y mes...")
    panel_model, beta_panel = modelo_panel(agg)
    print(f"Beta pooled ln_P: {beta_panel:.4f}")
    print(panel_model.summary().tables[1])

    print("\nInterpretacion: efectos temporales controlan estacionalidad; si beta cambia respecto a promedio individual, indica que parte de la relacion precio-demanda estaba confundida por shocks de temporada (endogeneidad).")


if __name__ == "__main__":
    main()
