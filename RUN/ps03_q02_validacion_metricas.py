"""PS03 - Pregunta 2: Validacion y metricas de negocio.
Ejecucion: python ps03_q02_validacion_metricas.py
"""
import math


def costo_por_bias(unidades_promedio: int = 10000, bias: float = -0.08, costo_over: float = 0.50, costo_under: float = 3.0) -> float:
    """Calcula costo esperado semanal atribuible al sesgo."""
    esperado = unidades_promedio * bias
    if esperado < 0:
        return abs(esperado) * costo_under
    return esperado * costo_over


def main():
    out = []
    # a) costo por sesgo
    costo_bias = costo_por_bias()
    sentido = "sub-stockear" if costo_bias > 0 else "sin costo"
    out.append(f"a) Bias -8% implica sub-forecast: se espera sub-stockear. Costo esperado semanal adicional ~${costo_bias:,.0f} (10,000 u * 8% * $3).")

    # b) validacion temporal
    out.append("b) K-Fold aleatorio rompe la causalidad temporal: el modelo veria el futuro en entrenamiento. Expanding window usa train desde el inicio hasta t y valida en t+h; Sliding window mantiene ventana fija (ej. ultimos 52 semanas) y se desplaza. Expanding sirve para series estables con largo historial; sliding para adaptarse a cambios de regimen y reducir costo computacional.")

    # c) ceros censurados
    out.append("c) Si 15% de ceros son quiebres de stock, el modelo aprende demanda menor a la real (sesgo negativo). Estrategias: (1) data augmentation: imputar esos ceros con pronostico de un modelo entrenado solo en dias con inventario>0 o con rolling mean condicional; (2) modelado: usar modelos con variable de inventario y tratamiento de censura (Zero-Inflated Poisson/Negative Binomial o hurdle) o clasificar ocurrencia y cantidad. Es un problema etico porque el sistema subestima ventas, perpetua desabastecimiento y pierde ingresos para clientes y empresa.")

    print("\n\n".join(out))


if __name__ == "__main__":
    main()
