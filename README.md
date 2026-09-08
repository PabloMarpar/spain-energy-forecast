# Demanda eléctrica y % renovable en España -- forecasting a 7 días

Predicción a 7 días de (1) la demanda eléctrica horaria de España y (2) el % de generación
renovable diario, usando datos 100% reales y públicos: [REE (Red Eléctrica de
España)](https://www.ree.es/) para demanda y mix de generación real, y
[Open-Meteo](https://open-meteo.com/) para clima (radiación solar, viento, oleaje...). Se
comparan 5 enfoques de forecasting de complejidad creciente -- desde un baseline ingenuo
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
  real por tecnología (`/generacion/estructura-generacion`): Solar fotovoltaica, Solar
  térmica, Eólica, Hidráulica, Nuclear, Ciclo combinado, Carbón, Cogeneración y el resto de
  tecnologías del sistema peninsular, día a día. De ahí sale el **% renovable real
  histórico** (no una estimación a partir del clima -- el clima es la variable predictiva,
  la generación real de REE es la verdad de entrenamiento y evaluación) y el desglose
  completo del mix, que alimenta la gráfica de generación tipo *stacked-area*.
- **Open-Meteo** -- radiación solar (media y máxima diaria, directa y global), nubosidad,
  viento a 10m y a 100m (la altura real de buje de un aerogenerador moderno, con media y
  máxima diaria), precipitación, temperatura, humedad y presión. Todo gratis, sin API key.
- **Open-Meteo Marine** -- oleaje cerca de Mutriku (País Vasco), la única planta undimotriz
  real de España. Se prueba como variable exploratoria: como esa tecnología es casi
  anecdótica en el mix nacional, se esperaba (y se confirma en la importancia de variables)
  que aporte poca señal frente a sol y viento -- se reporta el resultado real, no solo el que
  "queda mejor".
- **Ciclo anual (seno/coseno del día del año)** -- con 5 años de histórico ya hay ciclos
  anuales completos de sobra para que el modelo aprenda que el % solar depende sobre todo de
  la época del año, una señal mucho más estable que la meteorología día a día.

### Dos bugs reales, encontrados y corregidos

- **Escala rota en la API de REE.** Al construir la gráfica de mix de generación por
  tecnología, la suma de los campos `percentage` de las 13 tecnologías (sin la fila-resumen
  "Generación total") daba **exactamente 0.5000000** en vez de 1.0 -- verificado contra la
  suma de los `value` en MWh de esas mismas tecnologías, que sí coincide exactamente con el
  total. El campo `percentage` que expone la API para este endpoint viene escalado
  sistemáticamente a la mitad (no hay documentación pública que lo explique). Sin este
  hallazgo, el % renovable habría salido siempre a la mitad de su valor real (un 46% real se
  calculaba como ~23%). Ahora el % de cada tecnología se calcula a mano como
  `value_tecnología / value_total_del_día`, ignorando el campo `percentage` de la API.
- **El último día descargado casi nunca está completo.** `data.py` pedía datos hasta "ayer",
  pero REE no siempre tiene el día anterior publicado del todo -- se detectó un día con
  **1 sola hora** de demanda publicada de 24, que se colaba en el holdout de evaluación como
  si fuera un día real y parecía un desplome brutal del % renovable que ningún modelo
  acertaba. `data.py` ahora recorta del final los días con menos de 20 horas reales antes de
  guardar el dataset (`trim_incomplete_trailing_days`).

## Los modelos

1. **Baseline estacional** -- el mismo valor que hace una semana. La referencia mínima que
   cualquier modelo tiene que superar.
2. **SARIMAX** (`statsmodels` vía `pmdarima`) -- método estadístico clásico, con clima y
   calendario como variables exógenas.
3. **XGBoost -- forecasting DIRECTO, no recursivo.** Un único modelo predice cualquier paso
   `h` del horizonte a partir de lags/medias móviles calculados siempre con datos reales
   (nunca encadenando sus propias predicciones) más el clima del instante futuro y `h` como
   *feature* explícita. Hiperparámetros ajustados con **Optuna**. La primera versión era
   recursiva (una predicción a 1 paso realimentada paso a paso) y se aplanaba con el
   horizonte -- se explica más abajo por qué se cambió y qué mejoró.
4. **Chronos-2** (Amazon) -- *foundation model* de series temporales pre-entrenado.
   Funciona en modo **zero-shot**: no se reentrena con estos datos, solo se le da el
   histórico reciente como contexto y el clima previsto como covariable adicional. Que
   compita sin haber visto nunca electricidad española, sin reentrenar nada, es un resultado
   interesante en sí mismo -- gane o no gane en cada objetivo concreto.
5. **Ensemble** -- media simple de los tres modelos anteriores. Promediar modelos con
   errores no perfectamente correlacionados suele reducir el error total; se reporta como
   uno más y solo "gana" si de verdad mejora sobre los demás (en este proyecto no gana en
   ningún objetivo, y se reporta así en vez de forzarlo).

Todos se evalúan sobre el mismo **holdout temporal** (los últimos 7 días, nunca un split
aleatorio -- mezclar fechas al azar en series temporales dejaría que el modelo "viera" datos
posteriores al punto que se le pide predecir).

### Por qué XGBoost pasó de recursivo a directo

La primera versión encadenaba: predice el paso 1, usa esa predicción como "lag" del paso 2,
y así hasta el paso 168 (o 7). Con horizontes largos eso tiene un problema conocido: en
cuanto el modelo predice un valor cercano a la media, ese valor (ya suavizado) pasa a ser el
lag de entrada del siguiente paso -- y se va aplanando solo. Se notó al mirar la gráfica de
% renovable: la predicción apenas se movía día a día mientras el real oscilaba mucho más, y
la importancia de variables lo confirmaba -- `lag_1` (el valor de ayer) acaparaba el **60%**
de la importancia, todo el clima junto no llegaba al 15%. Ahora, con un modelo directo por
horizonte, `lag_1` baja a un 3-4% y el modelo se apoya en las medias móviles reales y en el
ciclo anual -- justo lo que se quería. El MAPE de % renovable con XGBoost bajó de 18.6% a
**5.3%** solo con este cambio (sobre datos ya corregidos).

## Resultados

*(5 años de histórico -- ~43.800 puntos horarios de demanda, ~1.800 días de % renovable --
holdout = últimos 7 días, nunca aleatorio; ya con los dos bugs corregidos)*

### Demanda eléctrica (MWh/h, horizonte 7 días)

| Modelo | MAE | RMSE | MAPE % | sMAPE % | R² | Mejora vs. baseline |
|---|---|---|---|---|---|---|
| Baseline | 3211.3 | 3811.4 | 10.37 | 11.14 | 0.15 | -- |
| **SARIMAX 🏆** | 1607.7 | 2146.8 | **5.50** | 5.65 | 0.73 | **+47.0%** |
| Ensemble (SARIMAX+XGBoost+Chronos-2) | 2004.5 | 2291.5 | 6.43 | 6.68 | 0.69 | +38.0% |
| XGBoost (directo, Optuna) | 2235.7 | 2577.5 | 7.10 | 7.42 | 0.61 | +31.5% |
| Chronos-2 (zero-shot + covariables) | 2648.5 | 2991.1 | 8.45 | 8.90 | 0.48 | +18.5% |

### % Generación renovable (horizonte 7 días)

| Modelo | MAE | RMSE | MAPE % | sMAPE % | R² | Mejora vs. baseline |
|---|---|---|---|---|---|---|
| Baseline | 4.89 | 5.82 | 9.68 | 9.04 | -2.07 | -- |
| **XGBoost (directo, Optuna) 🏆** | 2.69 | 3.37 | **5.26** | 5.08 | -0.03 | **+45.7%** |
| Ensemble (SARIMAX+XGBoost+Chronos-2) | 2.97 | 3.53 | 5.78 | 5.61 | -0.13 | +40.3% |
| SARIMAX | 3.11 | 3.61 | 6.01 | 5.87 | -0.18 | +37.9% |
| Chronos-2 (zero-shot + covariables) | 3.18 | 3.77 | 6.25 | 6.02 | -0.29 | +35.4% |

**Por qué gana un modelo distinto en cada objetivo:** la demanda horaria tiene decenas de
miles de puntos con un patrón diario/semanal muy estable -- ahí SARIMAX, bien especificado,
explota ese patrón mejor que nadie (47% de mejora sobre el baseline). El % renovable diario
tiene muchos menos puntos (la API de REE solo publica el mix de generación a nivel diario) y
depende del clima día a día -- ahí, una vez corregidos los dos bugs y con el ciclo anual
(`doy_sin`/`doy_cos`) y el forecasting directo, es XGBoost quien mejor explota esa señal
estacional, superando a SARIMAX y a Chronos-2 por primera vez con margen claro. Los cuatro
modelos "de verdad" superan ya con comodidad al baseline en los dos objetivos -- antes de
corregir los bugs, SARIMAX llegaba a quedar *peor* que el baseline en renovables, y XGBoost
apenas se despegaba de él por el aplanamiento del forecasting recursivo. La app usa el
campeón de cada objetivo por separado en vez de forzar un único modelo para las dos cosas.

![Demanda: real vs. predicción](outputs/demanda_mwh_comparativa.png)
![Mix de generación eléctrica por tecnología](outputs/generation_mix.png)
![% renovable: real vs. predicción](outputs/renewable_pct_comparativa.png)

## App interactiva (Streamlit)

`app.py` tiene 4 secciones:
- **Histórico interactivo**: demanda horaria y un mix de generación por tecnología tipo
  *stacked-area* (el estándar visual del sector), no solo el % renovable agregado.
- **Comparativa de modelos**: tabla completa de métricas (MAE, RMSE, MAPE, sMAPE, R²,
  sesgo, mejora vs. baseline, MAPE día 1 vs. día 7) y gráfica interactiva real-vs-predicción
  con los 5 modelos -- por defecto solo se muestran real, baseline y el campeón para que no
  sature, el resto se activa con un clic en la leyenda.
- **Predicción en tiempo real**: recalculada bajo demanda con la previsión de clima real de
  los próximos 7 días (no es un stream continuo -- de ahí las comillas), con el progreso
  visible paso a paso (qué modelo se está ejecutando y cuánto lleva tardando) en vez de un
  spinner ciego.
- **Metodología**: las explicaciones de este README pensadas para defenderlas en una
  entrevista, no solo para leerlas.

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Uso

```bash
pip install -r requirements.txt

python data.py      # descarga 5 años de demanda/generación (REE) + clima (Open-Meteo)
python train.py     # entrena y compara los 5 modelos para los 2 objetivos
python predict.py   # genera la predicción a 7 días con el modelo campeón de cada objetivo
streamlit run app.py
```

## Estructura

```
data.py         # descarga y construye el dataset horario unificado (REE + Open-Meteo)
train.py         # entrena y compara los modelos para los 2 objetivos, guarda métricas
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
- El % renovable es una serie diaria (no horaria) porque así lo publica REE en su API
  pública (se probó pedir el desglose por tecnología en horario y lo rechaza siempre con
  400) -- con datos horarios el modelo probablemente mejoraría bastante más, capturando el
  patrón diario de subida y bajada solar.
- En ~0.3% de los días la suma de tecnologías supera muy ligeramente (100-103%) el valor de
  "Generación total" que publica REE, por redondeos independientes entre ambas cifras en la
  fuente -- se recorta a 100 al calcular el % renovable.
- El recorte de días incompletos (`trim_incomplete_trailing_days`) usa un umbral fijo de 20
  horas -- funciona bien en la práctica, pero es una heurística, no una certeza de que REE
  haya terminado de publicar ese día.
