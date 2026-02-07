# Problem Set 03 - Marketing y Analitica del Retail

Scripts y entregables ubicados en la carpeta RUN.

## Requisitos
- Python 3.11 (venv ya configurado)
- Instalar dependencias: `pip install -r requirements.txt`

## Datasets
- DT/store-sales_ECUADOR/* para preguntas 1-3 (conceptual) y 7 (no aplica).
- DT/calendar.csv, DT/sell_prices.csv, DT/sales_train_evaluation.csv para la pregunta 4 (M5).
- DT/online_retail_II.csv para la pregunta 5.
- DT/Precios_PS03.xlsx para la pregunta 7 (asegurate de tenerlo en carpeta DT).

## Ejecucion por pregunta
- Q1: `python RUN/ps03_q01_descomposicion.py`
- Q2: `python RUN/ps03_q02_validacion_metricas.py`
- Q3: `python RUN/ps03_q03_modelos_casos.py`
- Q4: `python RUN/ps03_q04_mlforecast_pipeline.py`
  - Genera grafico en RUN/q04_predicciones.png
- Q5: `python RUN/ps03_q05_pricing_elasticidad.py`
- Q6: `python RUN/ps03_q06_dynamic_pricing.py`
- Q7: `python RUN/ps03_q07_propiedades.py --input DT/Precios_PS03.xlsx --group-id X`
  - Produce archivo Grupo_X_PS03Q07.xlsx en RUN.

## Notas
- Q4 puede tardar por el volumen de M5; filtra a FOODS_3 y CA_1-4 como pide el enunciado.
- Q5 lee ~1M filas; requiere memoria moderada.

