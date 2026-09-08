# Demanda eléctrica y % renovable en España -- forecasting a 7 días

Predicción a 7 días de (1) la demanda eléctrica horaria de España y (2) el % de generación
renovable diario, usando datos 100% reales y públicos: [REE (Red Eléctrica de
España)](https://www.ree.es/) para demanda y mix de generación real, y
[Open-Meteo](https://open-meteo.com/) para clima (radiación solar, viento, oleaje...). Se
comparan 4 enfoques de forecasting de complejidad creciente -- desde un baseline ingenuo
hasta [Chronos-2](https://github.com/amazon-science/chronos-forecasting), el *foundation
model* de series temporales de Amazon -- y la app final usa, para cada objetivo, el que de
verdad funciona mejor, en vez de forzar un único modelo para todo.

## Por qué este proyecto

No es un ejercicio de "voy a usar un dataset random para practicar". Mi grado (Business
Analytics) está pensado para formar un perfil híbrido entre negocio, ingeniería de datos y
estadística, y este proyecto es un paso deliberado hacia Data Science: entender de verdad
por qué se elige cada modelo, medir honestamente cuándo funciona y cuándo no, y dejarlo
documentado de forma que pueda defenderlo en una entrevista, no solo enseñarlo.

## Fuentes de datos

- **REE apidatos** -- demanda eléctrica horaria (`/demanda/evolucion`) y mix de generación
  real por tecnología (`/generacion/estructura-generacion`), de donde sale el **% renovable
  real histórico** (no una estimación a partir del clima -- el clima es la variable
  predictiva, la generación real de REE es la verdad de entrenamiento y evaluación).
- **Open-Meteo** -- radiación solar (directa y global), nubosidad, viento a 10m y a 100m (la
  altura real de buje de un aerogenerador moderno), precipitación, temperatura, humedad y
  presión. Todo gratis, sin API key.
- **Open-Meteo Marine** -- oleaje cerca de Mutriku (País Vasco), la única planta undimotriz
  real de España. Se prueba como variable exploratoria: como esa tecnología es casi
  anecdótica en el mix nacional, se esperaba (y se confirma en la importancia de variables)
  que aporte poca señal frente a sol y viento -- se reporta el resultado real, no solo el que
  "queda mejor".

## Los 4 modelos

1. **Baseline estacional** -- el mismo valor que hace una semana. La referencia mínima que
   cualquier modelo tiene que superar.
2. **SARIMAX** (`statsmodels` vía `pmdarima`) -- método estadístico clásico, con clima y
   calendario como variables exógenas.
3. **XGBoost** -- recursivo, con lags, medias móviles y clima/calendario como *features*,
   hiperparámetros ajustados con **Optuna** (búsqueda bayesiana, no a mano).
4. **Chronos-2** (Amazon) -- *foundation model* de series temporales pre-entrenado.
   Funciona en modo **zero-shot**: no se reentrena con estos datos, solo se le da el
   histórico reciente como contexto y el clima previsto como covariable adicional. Que
   compita -- y en un objetivo incluso gane -- sin haber visto nunca electricidad española es
   el resultado más interesante del proyecto.

Todos se evalúan sobre el mismo **holdout temporal** (los últimos 7 días, nunca un split
aleatorio -- mezclar fechas al azar en series temporales dejaría que el modelo "viera" datos
posteriores al punto que se le pide predecir).

## Resultados

*(5 años de histórico -- 43.807 puntos horarios de demanda, 1.803 días de % renovable --
holdout = últimos 7 días, nunca aleatorio)*

### Demanda eléctrica (MWh/h, horizonte 7 días)

| Modelo | MAE | RMSE | MAPE % | sMAPE % | R² | Mejora vs. baseline |
|---|---|---|---|---|---|---|
| Baseline | 3386.1 | 3964.4 | 10.96 | 11.82 | -0.06 | -- |
| **SARIMAX 🏆** | 1843.6 | 2338.2 | **6.14** | 6.43 | 0.63 | **+44.0%** |
| XGBoost (Optuna) | 2452.1 | 2678.3 | 7.91 | 8.29 | 0.52 | +27.8% |
| Chronos-2 (zero-shot + covariables) | 2772.2 | 3105.1 | 8.86 | 9.36 | 0.35 | +19.2% |

### % Generación renovable (horizonte 7 días)

| Modelo | MAE | RMSE | MAPE % | sMAPE % | R² | Mejora vs. baseline |
|---|---|---|---|---|---|---|
| Baseline | 5.97 | 7.92 | 28.83 | 23.04 | -0.32 | -- |
| SARIMAX | 7.55 | 8.93 | 35.88 | 28.17 | -0.67 | -24.5% |
| XGBoost (Optuna) | 5.28 | 7.34 | 25.08 | 19.22 | -0.13 | +13.0% |
| **Chronos-2 (zero-shot + covariables) 🏆** | 4.09 | 5.88 | **21.55** | 16.45 | 0.28 | **+25.3%** |

**Por qué gana un modelo distinto en cada objetivo:** la demanda horaria tiene decenas de
miles de puntos con un patrón diario/semanal muy estable -- ahí SARIMAX, bien especificado,
explota ese patrón mejor que nadie (44% de mejora sobre el baseline). El % renovable diario
tiene muchos menos puntos y depende del clima día a día (más ruidoso) -- ahí SARIMAX queda
incluso *peor* que el baseline (con 5 años de datos, `auto_arima` ni siquiera encuentra
componente estacional semanal: orden `(2,0,1)x(0,0,0,7)`, señal de que el patrón semanal que
sí existe en demanda no existe de forma útil aquí), XGBoost pasa a superar al baseline al
darle más historia y features, y Chronos-2 -- que ya "sabe" de series temporales en general
por su preentrenamiento, sin haber visto nunca esta serie -- es quien mejor generaliza con
pocos datos específicos del problema. La app usa el campeón de cada objetivo por separado en
vez de forzar un único modelo para las dos cosas.

![Demanda: real vs. predicción](outputs/demanda_mwh_comparativa.png)
![% renovable: real vs. predicción](outputs/renewable_pct_comparativa.png)

## App interactiva (Streamlit)

`app.py` tiene 4 secciones: histórico interactivo, comparativa de los 4 modelos (tabla de
métricas + gráfica real-vs-predicción, no solo una imagen estática), una predicción "en
tiempo real" (recalculada bajo demanda con la previsión de clima real de los próximos 7
días) y una pestaña de metodología con las explicaciones de arriba.

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Uso

```bash
pip install -r requirements.txt

python data.py      # descarga 5 años de demanda/generación (REE) + clima (Open-Meteo)
python train.py     # entrena y compara los 4 modelos para los 2 objetivos
python predict.py   # genera la predicción a 7 días con el modelo campeón de cada objetivo
streamlit run app.py
```

## Estructura

```
data.py         # descarga y construye el dataset horario unificado (REE + Open-Meteo)
train.py         # entrena y compara los 4 modelos para los 2 objetivos, guarda métricas
predict.py        # predicción a 7 días "en vivo" con el modelo campeón de cada objetivo
app.py             # dashboard de Streamlit
outputs/            # métricas, gráficas y predicciones generadas por los scripts
energy_weather_hourly.csv  # dataset cacheado (se regenera con data.py)
```

## Limitaciones honestas

- SARIMAX para demanda usa un orden fijo, no una búsqueda automática completa: con
  estacionalidad de 24 horas sobre decenas de miles de puntos horarios, la búsqueda de
  `pmdarima.auto_arima` es intratable en CPU en tiempo razonable. Es una decisión de
  ingeniería documentada, no una limitación escondida.
- La previsión meteorológica a 7 días es, como cualquier previsión del tiempo, menos fiable
  cuanto más lejos mira -- por eso las métricas se reportan también por separado para el día
  1 y el día 7 del horizonte.
- El oleaje se probó como variable exploratoria y aporta poca señal frente a sol y viento,
  como cabía esperar dado el peso casi nulo de la energía undimotriz en el mix español.
- El MAPE del día 7 en % renovable dispara por encima del 100% en varios modelos (incluido
  el baseline): el día concreto que cae al final del holdout tuvo un % renovable real muy
  bajo, y MAPE se dispara cuando el valor real se acerca a 0 (un error pequeño en términos
  absolutos se convierte en un porcentaje enorme). Por eso se reporta también sMAPE, que no
  tiene ese problema, y por eso Chronos-2 sigue siendo el campeón pese a ese día concreto:
  es el que menos se desvía en términos absolutos (MAE) y relativos estables (sMAPE).
