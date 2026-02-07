"""PS03 - Pregunta 6: Pricing dinamico RideFlow.
Ejecucion: python ps03_q06_dynamic_pricing.py
"""
import pandas as pd

segments = pd.DataFrame({
    "Segmento": ["Ejecutivos", "Casual", "Nocturno", "Aeropuerto"],
    "Viajes_dia": [12500, 17500, 10000, 10000],
    "Elasticidad": [-0.6, -1.8, -1.2, -0.4],
})
PRECIO_BASE = 3500
COSTO = 2100


def regla_lerner(beta):
    return -1 / beta


def precio_optimo(beta, costo):
    beta_abs = abs(beta)
    if beta_abs <= 1:
        return None  # no hay optimo finito
    return costo * (beta_abs / (beta_abs - 1))


def main():
    segments["Margen_actual_pct"] = (PRECIO_BASE - COSTO) / PRECIO_BASE
    segments["Margen_opt_teorico"] = segments["Elasticidad"].apply(regla_lerner)
    segments["Precio_opt"] = segments["Elasticidad"].apply(lambda b: precio_optimo(b, COSTO))
    segments["Multiplicador_vs_base"] = segments["Precio_opt"] / PRECIO_BASE

    segments["Contrib_actual"] = segments["Viajes_dia"] * (PRECIO_BASE - COSTO)
    segments["Contrib_opt"] = segments.apply(lambda r: r["Viajes_dia"] * ((r["Precio_opt"] - COSTO) if pd.notnull(r["Precio_opt"]) else 0), axis=1)

    total_actual = segments["Contrib_actual"].sum()
    total_opt = segments["Contrib_opt"].sum()
    uplift = (total_opt - total_actual) / total_actual if total_actual else 0

    print(segments[["Segmento", "Elasticidad", "Margen_actual_pct", "Margen_opt_teorico", "Precio_opt", "Multiplicador_vs_base"]])
    print("\nContribucion diaria actual:", total_actual)
    print("Contribucion diaria con precios optimos (sin cambio de demanda en corto plazo):", total_opt)
    print(f"Uplift porcentual vs precio uniforme: {uplift:.2%}")
    print("Nota: Para elasticidades inelasticas (|beta|<=1) el precio optimo teorico no es finito; se debe fijar un tope por percepcion de justicia y riesgo de reduccion de demanda.")


if __name__ == "__main__":
    main()
