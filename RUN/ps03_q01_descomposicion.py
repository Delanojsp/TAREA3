"""PS03 - Pregunta 1: Descomposicion temporal e ingenieria de variables.
Ejecucion desde consola: python ps03_q01_descomposicion.py
"""

def main():
    answer = []
    answer.append("a) Usar lag_0 equivale a pasar la respuesta verdadera como feature: el modelo veria la venta ya conocida del mismo dia, lo que es data leakage. En backtesting parece perfecto (mismo horizonte usa el valor real), pero en produccion ese valor no existe y el error real se dispara. La metrica queda inflada y las decisiones operativas serian demasiado optimistas.")
    answer.append("b) Si la tendencia de 15% anual proviene de aperturas de tiendas, no refleja crecimiento organico por tienda. El forecast sobreestima la demanda por tienda y sesga decisiones de inventario. Para aislar la tendencia real se puede normalizar ventas por tienda activa, modelar una serie por store_nbr y agregar una variable exogena de aperturas, o bien restar el efecto de nuevas tiendas mediante una regresion con indicador de tiendas abiertas y luego pronosticar la parte organica.")
    answer.append("c) Dia_semana como entero impone orden y distancia falsa (0 y 6 no son extremos). La codificacion ciclica usa seno y coseno: sin_d = sin(2*pi*dia/7), cos_d = cos(2*pi*dia/7). Asi lunes y domingo quedan cercanos en el espacio trigonometrico y el modelo captura periodicidad.")
    answer.append("d) Residuo positivo tras feriados sugiere efecto post-feriado. Se debe agregar una variable exogena de distancia a feriado o indicador post_holiday. Con regresores exogenos un modelo estadistico (ARIMAX/Prophet) puede capturarlo; en ML incluir la feature permite que modelos de arboles ajusten ese uplift. Sin la variable, el modelo confunde el patron con ruido y reduce precision.")
    print("\n\n".join(answer))

if __name__ == "__main__":
    main()
