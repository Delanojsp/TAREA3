import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean, RollingStd
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import warnings
from pathlib import Path

# Configuración inicial
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("--- Librerías cargadas correctamente ---\n")

EC_BASE = Path(__file__).resolve().parent.parent / "DT" / "store-sales_ECUADOR"

# ==========================================
# PARTE 0: Conceptuales (Preguntas 1-3)
# ==========================================
def ejecutar_conceptuales():
    print("\n>>> RESPUESTAS PREGUNTAS 1-3 (CONCEPTUALES)")
    # Respuestas textuales
    respuestas = [
        "P1a: lag_0 es leakage: usa la venta real del mismo dia; en backtest luce perfecto pero en produccion no existe y el error real sube. Se inflan metricas y se sobreestima desempeño.",
        "P1b: La tendencia 15% proviene de aperturas, no de demanda organica por tienda. El forecast sobreestima por tienda. Normalizar por tiendas activas o incluir variable de aperturas para aislar la tendencia real.",
        "P1c: Dia_semana como entero impone orden lineal; 0 y 6 quedan lejos. Codificar ciclico: sin = sin(2π*d/7), cos = cos(2π*d/7).", 
        "P1d: Residuos +25% post feriado => incluir variable exogena post-feriado/lag de feriado. Modelos con regresores (ARIMAX/Prophet) o ML con feature; sin ella confunde ruido y reduce precision.",
        "P2a: Bias -8% => sub-forecast/sub-stock. Costo esperado: 10k u * 8% * $3 ≈ $2,400/semana.",
        "P2b: K-Fold aleatorio rompe temporalidad. Expanding: ventana crece; Sliding: ventana fija que se mueve. Expanding para series estables/largo historial; Sliding para regimen cambiante y costo controlado.",
        "P2c: Ceros por stockout sesgan a la baja. Mitigar: (1) imputar/augmentation usando dias con inventario>0 o rolling condicional; (2) modelos con censura/ZIP-hurdle y feature de inventario. Es etico porque perpetua quiebres y pérdida de ventas.",
        "P3a: Demanda intermitente: ARIMA/ETS suavizan a la media. Croston separa frecuencia e intensidad; ocurrencia ~ Bernoulli/Geom, cantidad ~ Poisson/NB.",
        "P3b: NHITS para todo: 2 años y 50k series => costo alto; deep learning no siempre mejor para series cortas/ simples. Usarlo solo donde haya patrones complejos y datos suficientes.",
        "P3c: Black Friday con 2 eventos: modelos sin regresores fallan. Estrategia: modelo base + variable categorica de evento + uplift experto. Evaluar ex post con MAPE del evento y KPIs de negocio (stock-out, rotacion).",
    ]
    print("\n".join(respuestas))

    # Breve respaldo numerico para P2a (costo por bias) y ejemplo de codificacion ciclica
    unidades = 10_000
    bias = -0.08
    costo_under = 3.0
    costo_bias = abs(unidades * bias) * costo_under
    print(f"\nP2a soporte numerico: unidades={unidades}, bias={bias:+.0%}, costo_under=${costo_under} => costo semanal ≈ ${costo_bias:,.0f}")

    # Muestra de codificacion ciclica para dia de semana (0=lunes, 6=domingo)
    dias = [0, 1, 5, 6]
    ciclico = [(d, np.sin(2 * np.pi * d / 7), np.cos(2 * np.pi * d / 7)) for d in dias]
    print("\nP1c ejemplo codificacion ciclica (d, sin, cos):")
    for d, s, c in ciclico:
        print(f"  d={d}: sin={s:.3f}, cos={c:.3f}")

    # ===============================
    # Apoyo analitico cargando DT
    # ===============================
    try:
        df_ec = pd.read_csv(EC_BASE / "train.csv", parse_dates=["date"])
        df_ec = df_ec.sort_values(["store_nbr", "date"])

        # P1: correlacion con lags en GROCERY I
        df_g = df_ec[df_ec["family"] == "GROCERY I"].copy()
        df_g["lag_1"] = df_g.groupby("store_nbr")["sales"].shift(1)
        corr_lag1 = df_g[["sales", "lag_1"]].corr().iloc[0, 1]
        print(f"\nP1 soporte: Corr(sales, lag_1) GROCERY I = {corr_lag1:.3f} (lag_0 seria 1.0 y es leakage)")

        # P1b: tendencia por tienda vs total (apertura tiendas)
        daily = df_ec.groupby("date").agg(total_sales=("sales", "sum"), stores=("store_nbr", "nunique"))
        daily["per_store"] = daily["total_sales"] / daily["stores"]
        annual = daily.resample("A").mean()[["total_sales", "per_store"]]
        annual_growth_total = annual.pct_change().rename(columns=lambda c: c + "_growth")
        print("\nP1b soporte: crecimiento total vs por tienda (promedio anual):")
        print(pd.concat([annual, annual_growth_total], axis=1).dropna().tail(3))

        # P1d: efecto post feriado
        holidays = pd.read_csv(EC_BASE / "holidays_events.csv", parse_dates=["date"])
        feriados = holidays[holidays["type"] == "Holiday"]["date"]
        sales_day = df_ec.groupby("date")["sales"].sum()
        weekday_mean = sales_day.groupby(sales_day.index.dayofweek).mean()
        post_dates = feriados + pd.Timedelta(days=1)
        post_sales = sales_day.reindex(post_dates).dropna()
        uplift = (post_sales - post_sales.index.dayofweek.map(weekday_mean)) / post_sales.index.dayofweek.map(weekday_mean)
        if len(uplift):
            print(f"\nP1d soporte: uplift promedio dia post-feriado vs mismo dia-semana: {uplift.mean():.2%} (n={len(uplift)})")

        # P2c: ceros censurados aproximacion
        zero_share = (df_ec["sales"] == 0).mean()
        family_zero = df_ec.groupby("family")["sales"].apply(lambda s: (s == 0).mean()).sort_values(ascending=False).head(5)
        print(f"\nP2c soporte: porcentaje de registros con ventas=0 en todo el set: {zero_share:.2%}")
        print("Top 5 familias por proporcion de ceros:")
        print(family_zero)

        # P3a: demanda intermitente indicador simple por familia
        family_mean = df_ec.groupby("family")["sales"].mean()
        intermittency = pd.DataFrame({"zero_share": family_zero, "mean_sales": family_mean}).dropna().sort_values("zero_share", ascending=False).head(5)
        print("\nP3a soporte: familias con alta intermitencia (zeros altos y baja media):")
        print(intermittency)

    except FileNotFoundError:
        print("\nNo se encontraron datos de Ecuador en DT/store-sales_ECUADOR; se omite apoyo analitico.")
    except Exception as e:
        print(f"\nApoyo analitico omitido por error: {e}")

# ==========================================
# PARTE 1: Forecasting (M5 Walmart)
# ==========================================
def ejecutar_forecasting():
    print("\n>>> EJECUTANDO PREGUNTA 4: M5 FORECASTING")
    try:
        # Carga de datos
        print("Cargando datasets...")
        df_sales = pd.read_csv('sales_train_evaluation.csv', nrows=10000)
        df_calendar = pd.read_csv('calendar.csv')
        df_prices = pd.read_csv('sell_prices.csv')

        # Procesamiento
        print("Procesando datos...")
        df_sales = df_sales[df_sales['dept_id'] == 'FOODS_3']
        df_sales = df_sales[df_sales['store_id'].isin(['CA_1', 'CA_2', 'CA_3', 'CA_4'])]

        id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        df_melted = df_sales.melt(id_vars=id_vars, var_name='d', value_name='y')

        df_final = df_melted.merge(df_calendar, on='d', how='left')
        df_final = df_final.merge(df_prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')

        df_final['ds'] = pd.to_datetime(df_final['date'])
        df_final = df_final.rename(columns={'id': 'unique_id'})

        # Features
        df_final['is_event'] = np.where(df_final['event_name_1'].notnull(), 1, 0)
        df_final['is_weekend'] = np.where(df_final['wday'].isin([1, 2]), 1, 0)
        df_final['day_of_week'] = df_final['ds'].dt.dayofweek

        df_final = df_final[['unique_id', 'ds', 'y', 'snap_CA', 'sell_price', 'wday', 'is_event', 'is_weekend', 'day_of_week']]
        df_final = df_final.sort_values(['unique_id', 'ds']).reset_index(drop=True)

        print("Data preparada. Head:")
        print(df_final.head())

        # Modelado
        models = [
            LGBMRegressor(random_state=42, n_estimators=100, verbose=-1),
            XGBRegressor(random_state=42, n_estimators=100)
        ]

        fcst = MLForecast(
            models=models,
            freq='D',
            lags=[1, 7, 14, 28],
            lag_transforms={
                1: [RollingMean(window_size=7), RollingMean(window_size=14), RollingStd(window_size=7)]
            },
            date_features=['day', 'month', 'dayofweek', 'week'],
            num_threads=4
        )

        test_days = 13
        train_end = df_final['ds'].max() - pd.Timedelta(days=test_days)
        df_train = df_final[df_final['ds'] <= train_end]
        df_test = df_final[df_final['ds'] > train_end]

        print("Entrenando modelos...")
        fcst.fit(df_train, id_col='unique_id', time_col='ds', target_col='y', static_features=[])

        future_exog = df_test[['unique_id', 'ds', 'sell_price', 'is_event', 'snap_CA', 'is_weekend', 'wday', 'day_of_week']]
        preds = fcst.predict(test_days, X_df=future_exog)

        eval_df = preds.merge(df_test[['unique_id', 'ds', 'y']], on=['unique_id', 'ds'], how='left')

        # Métricas
        mae_lgbm = mean_absolute_error(eval_df['y'], eval_df['LGBMRegressor'])
        rmse_lgbm = np.sqrt(mean_squared_error(eval_df['y'], eval_df['LGBMRegressor']))
        
        mae_xgb = mean_absolute_error(eval_df['y'], eval_df['XGBRegressor'])
        rmse_xgb = np.sqrt(mean_squared_error(eval_df['y'], eval_df['XGBRegressor']))

        print(f"Resultados LightGBM: MAE={mae_lgbm:.4f}, RMSE={rmse_lgbm:.4f}")
        print(f"Resultados XGBoost: MAE={mae_xgb:.4f}, RMSE={rmse_xgb:.4f}")

        # Gráfico
        one_series = eval_df['unique_id'].unique()[0]
        subset = eval_df[eval_df['unique_id'] == one_series]

        plt.figure(figsize=(10, 5))
        plt.plot(subset['ds'], subset['y'], label='Real')
        plt.plot(subset['ds'], subset['LGBMRegressor'], label='LGBM')
        plt.plot(subset['ds'], subset['XGBRegressor'], label='XGB')
        plt.title(f'Predicción para {one_series}')
        plt.legend()
        plt.show()

    except FileNotFoundError:
        print("⚠️ ARCHIVOS FALTANTES: Asegúrate de tener 'sales_train_evaluation.csv', 'calendar.csv' y 'sell_prices.csv' en la misma carpeta.")
    except Exception as e:
        print(f"Error en Forecasting: {e}")

# ==========================================
# PARTE 2: Pricing & Elasticidad
# ==========================================
def ejecutar_pricing():
    print("\n>>> EJECUTANDO PREGUNTA 5: PRICING & ELASTICIDAD")
    try:
        df = pd.read_csv('online_retail_II.csv')
        
        # Preparación
        df_uk = df[df['Country'] == 'United Kingdom'].copy()
        df_uk = df_uk[~df_uk['Invoice'].astype(str).str.startswith('C')]
        df_uk = df_uk[(df_uk['Quantity'] > 0) & (df_uk['Price'] > 0) & (df_uk['Customer ID'].notnull())]

        top_products = df_uk.groupby('StockCode')['Quantity'].sum().nlargest(10).index.tolist()
        df_top = df_uk[df_uk['StockCode'].isin(top_products)].copy()
        
        df_top['InvoiceDate'] = pd.to_datetime(df_top['InvoiceDate'])
        df_top['Week'] = df_top['InvoiceDate'].dt.to_period('W').apply(lambda r: r.start_time)

        df_weekly = df_top.groupby(['StockCode', 'Week']).agg(
            quantity_total=('Quantity', 'sum'),
            price_avg=('Price', 'mean')
        ).reset_index()

        df_weekly['ln_Q'] = np.log(df_weekly['quantity_total'])
        df_weekly['ln_P'] = np.log(df_weekly['price_avg'])
        df_weekly['Month'] = df_weekly['Week'].dt.month

        # Modelo Panel
        model_panel = smf.ols("ln_Q ~ ln_P + C(StockCode) + C(Month)", data=df_weekly).fit()
        print(f"Elasticidad Estimada (Beta del Panel): {model_panel.params['ln_P']:.4f}")
        print(model_panel.summary().tables[1])

    except FileNotFoundError:
        print("⚠️ ARCHIVO FALTANTE: Falta 'online_retail_II.csv'.")
    except Exception as e:
        print(f"Error en Pricing: {e}")

# ==========================================
# PARTE 3: RideFlow (Cálculo Numérico)
# ==========================================
def ejecutar_rideflow():
    print("\n>>> EJECUTANDO PREGUNTA 6: RIDEFLOW")
    segments = ['Ejecutivos', 'Casual', 'Nocturno', 'Aeropuerto']
    demand_daily = np.array([12500, 17500, 10000, 10000])
    elasticity = np.array([-0.6, -1.8, -1.2, -0.4])
    price_current = 3500
    cost_variable = 2100

    # Cálculos
    margin_curr_pct = (price_current - cost_variable) / price_current
    optimal_prices = cost_variable * (elasticity / (1 + elasticity))
    
    results = pd.DataFrame({
        'Segmento': segments,
        'Elasticidad': elasticity,
        'Precio_Optimo_Teorico': optimal_prices,
        'Margen_Actual': margin_curr_pct
    })
    
    print(results)
    print("\nNota: Precios negativos/altos en segmentos inelásticos indican necesidad de tope máximo.")

# ==========================================
# PARTE 4: Predicción Precios (XGBoost)
# ==========================================
def ejecutar_propiedades():
    print("\n>>> EJECUTANDO PREGUNTA 7: PREDICCIÓN PROPIEDADES")
    file_name = 'DT/Precios_PS03.xlsx'
    try:
        df_train = pd.read_excel(file_name, sheet_name='Estudiantes-train-test')
        df_target = pd.read_excel(file_name, sheet_name='Estudiantes-target')

        features = [c for c in df_train.columns if c not in ['precio_uf', 'titulo_propiedad', 'id', 'Id']]
        X = df_train[features]
        y = df_train['precio_uf']
        X_submit = df_target[features]

        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=features)
        X_submit_imputed = pd.DataFrame(imputer.transform(X_submit), columns=features)

        model = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, n_jobs=-1, random_state=42)
        model.fit(X_imputed, y)
        
        final_preds = model.predict(X_submit_imputed)
        
        submission = pd.DataFrame({
            'titulo_propiedad': df_target['titulo_propiedad'],
            'precio_uf': final_preds
        })
        
        output_filename = 'Grupo_3_PS03007.xlsx'
        submission.to_excel(output_filename, index=False)
        print(f"✅ Archivo generado: {output_filename}")

    except FileNotFoundError:
        print(f"⚠️ ARCHIVO FALTANTE: Falta '{file_name}'.")
    except Exception as e:
        print(f"Error en Propiedades: {e}")

# ==========================================
# EJECUCIÓN PRINCIPAL
# ==========================================
if __name__ == "__main__":
    # Descomenta la función que desees correr si tienes los datos
    ejecutar_conceptuales()
    ejecutar_forecasting()
    ejecutar_pricing()
    ejecutar_rideflow()
    ejecutar_propiedades()