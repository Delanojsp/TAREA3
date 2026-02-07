"""PS03 - Pregunta 3: Seleccion de algoritmos y casos especiales.
Ejecucion: python ps03_q03_modelos_casos.py
"""


def main():
    out = []
    out.append("a) Demanda intermitente tiene muchos ceros y valores esporadicos: modelos continuos suavizan hacia la media y pierden ocurrencias. Croston separa dos procesos: probabilidad de ocurrencia (intervalo entre demandas) y tamanio condicional cuando ocurre; pronostica ambos y combina. Para ocurrencia es mejor una distribucion Bernoulli/Geometrica; para cantidad, Poisson o Negativa Binomial condicionada.")
    out.append("b) NHITS para todo el catalogo no siempre es mejor: solo 2 anos de datos limita capacidad de deep learning; 50k series multiplican costo de entrenamiento e inferencia; para series cortas o estacionales simples, modelos clasicos o ML ligeros pueden ser mas robustos y explicables. Reservar deep learning para segmentos con suficiente historia y patrones complejos.")
    out.append("c) Black Friday con solo 2 eventos: modelos sin regresores no capturan efecto. Estrategia hibrida: (1) modelo base (Prophet/LightGBM) entrenado sin evento; (2) variable exogena categorica para Black Friday y ventana previa/post; (3) uplift experto multiplicativo basado en historial y comparables. Sin backtesting, evaluar ex post con MAPE sobre periodo post-evento y con KPI de negocio (stock-out, rotacion).")
    print("\n\n".join(out))


if __name__ == "__main__":
    main()
