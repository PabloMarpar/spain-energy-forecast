# Demanda eléctrica y % renovable en España -- forecasting a 7 días

Predicción a 7 días de (1) la demanda eléctrica horaria de España y (2) el % de generación
renovable diario, usando datos 100% reales y públicos: [REE (Red Eléctrica de
España)](https://www.ree.es/) para demanda y mix de generación real, y
[Open-Meteo](https://open-meteo.com/) para clima (radiación solar, viento, oleaje...). Se
comparan varios enfoques de forecasting de complejidad creciente -- desde un baseline ingenuo
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
- **Viento en las 3 regiones eólicas de verdad** (Aragón, Castilla y León, Galicia), no en
  Madrid -- Madrid no es zona eólica, y medir ahí el viento para explicar la generación
  eólica nacional era medir en el sitio equivocado. Se usa la media, el máximo y la
  desviación entre las tres regiones, más `wind_power_proxy` (viento medio al cubo, capado a
  12 m/s): la potencia de un aerogenerador escala aproximadamente con el cubo de la
  velocidad del viento, no linealmente, así que dar esa variable ya calculada le ahorra al
  modelo tener que reconstruir la curva a base de cortes.
- **Ciclo anual (seno/coseno del día del año)** -- con varios años de histórico ya hay ciclos
  anuales completos de sobra para que el modelo aprenda que el % solar depende sobre todo de
  la época del año, una señal mucho más estable que la meteorología día a día.
- **Clima de 5 áreas metropolitanas ponderado por población** (Madrid, Barcelona, Valencia,
  Sevilla, Bilbao) para demanda -- mismo razonamiento que con el viento: la demanda nacional
  depende de dónde vive la gente de verdad, no de un único punto. De ahí se derivan
  **HDD/CDD** (grados-día de calefacción/refrigeración, `max(0,18-T)` / `max(0,T-18)`), el
  estándar del sector para demanda sensible al clima, que capturan el doble pico anual de
  España (verano por AC, invierno por calefacción) mejor que la temperatura en bruto.
- **Precio spot horario** (REE, endpoint de mercados) -- usado solo como *lag* (precio de
  ayer, de hace una semana), nunca como "precio futuro conocido": en producción el precio de
  mañana en adelante no se conoce con certeza más allá de un día.
- **Festivos regionales ponderados por población** -- una fiesta en La Rioja (0.6% de la
  población) no pesa igual que una en Andalucía o Cataluña (16-18%).

## Los datos: 10 años, no 5

Se amplió el histórico de 5 a 10 años para dar más ejemplos a los modelos entrenados desde
cero (especialmente el GRU). Aviso honesto: para % renovable esto mete años con mucha menos
solar/eólica instalada que ahora (el mix cambió mucho desde 2016) -- se probó igualmente
porque la ganancia de más datos para XGBoost/GRU compensaba, y de hecho el rendimiento se
mantuvo o mejoró.

### Bugs reales, encontrados y corregidos

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
- **Open-Meteo devuelve el viento en km/h, no en m/s.** Sin pedirlo explícitamente, `wind_power_proxy`
  (pensado en m/s, con el corte físico a 12) capaba casi todos los valores porque en km/h
  casi todo supera 12 -- la variable salía prácticamente constante. Se corrigió pidiendo
  `wind_speed_unit=ms` a la API.
- **La validación de Chronos-2 rechazaba la predicción en vivo.** En la evaluación offline el
  histórico y el futuro son contiguos (se trocea la misma serie), pero en producción casi
  nunca: el histórico real termina 1-2 días antes de "hoy" (por el mismo motivo del punto
  anterior) mientras que la previsión de clima empieza "hoy". Chronos-2 valida por defecto
  que no haya hueco entre ambos y lo rechazaba (`validate_inputs=False` para desactivarlo,
  documentado como una discontinuidad real y esperada, no un error).
- **NaN de arranque envenenaban los pesos del GRU en producción.** `precio_lag_168` no existe
  hasta la hora 168 del histórico (es un `.shift()`) -- en `train.py` esos NaN ya se filtraban
  antes de entrenar cualquier modelo, pero en la predicción en vivo llegaban sin filtrar.
  Un único NaN en un tensor de PyTorch contamina todo el forward/backward, y basta con que
  caiga en un origen de entrenamiento para que **todos** los pesos del modelo acaben en NaN
  para siempre -- la predicción salía en 0 GWh de demanda sin ningún error visible (la suma
  de una serie de puros NaN da 0 con `pandas.sum()`, no NaN). Se corrigió filtrando dentro de
  la propia función que construye las secuencias de entrenamiento, no confiando en que quien
  llame ya haya limpiado los datos.
- **Un hueco de ~77 días en el oleaje histórico rompía SARIMAX -- y llevaba recortando el
  histórico de renovables a la mitad sin que se notase.** Al ampliar a 10 años, Open-Meteo
  Marine no cubre toda esa ventana (solo ~5 años hacia atrás). SARIMAX para renovables entrena
  con todo el histórico sin ventana, así que ese hueco rompía el ajuste (`exog contains inf or
  nans`). Pero el efecto real era peor: el `dropna` que limpia cualquier NaN antes de entrenar
  estaba descartando **todos** los días anteriores al hueco, así que el resto de modelos
  (XGBoost, GRU, Chronos-2) llevaban entrenando en secreto con solo ~5 años, no los 10
  pretendidos. Como el oleaje ya se sabía poco relevante (importancia de variable siempre
  baja), se retiró del modelado en vez de parchear el hueco -- y el histórico de renovables
  pasó de golpe de 1.802 a 3.650 puntos reales.

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
4. **GRU directo multi-horizonte** (PyTorch) -- el clásico de deep learning que faltaba en la
   comparativa. Un encoder GRU procesa la ventana de histórico reciente (14 días para
   demanda, 60 para renovables), su último estado oculto se combina con un resumen del clima
   futuro conocido, y una capa densa produce las predicciones del horizonte completo de una
   sola vez -- directo, no recursivo, mismo motivo que XGBoost. Con ~1.800-3.650 puntos en
   renovables era dudoso que una red entrenada desde cero tuviera suficientes ejemplos para
   no sobreajustar frente a SARIMAX o Chronos-2 zero-shot -- en demanda (87K puntos horarios)
   tenía muchas más opciones.
5. **Chronos-2** (Amazon) -- *foundation model* de series temporales pre-entrenado.
   Funciona en modo **zero-shot**: no se reentrena con estos datos, solo se le da el
   histórico reciente como contexto y el clima previsto como covariable adicional. Que
   compita sin haber visto nunca electricidad española, sin reentrenar nada, es un resultado
   interesante en sí mismo -- gane o no gane en cada objetivo concreto.
6. **Ensemble** -- media simple de los modelos anteriores. Promediar modelos con
   errores no perfectamente correlacionados suele reducir el error total; se reporta como
   uno más y solo "gana" si de verdad mejora sobre los demás.

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

*(10 años de histórico -- ~87.400 puntos horarios de demanda, 3.650 días de % renovable --
holdout = últimos 7 días, nunca aleatorio; con todos los bugs ya corregidos)*

### Demanda eléctrica (MWh/h, horizonte 7 días)

| Modelo | MAE | RMSE | MAPE % | sMAPE % | R² | Mejora vs. baseline |
|---|---|---|---|---|---|---|
| Baseline | 3211.3 | 3811.4 | 10.37 | 11.14 | 0.15 | -- |
| **GRU (PyTorch, directo) 🏆** | 1021.5 | 1244.0 | **3.27** | 3.34 | 0.91 | **+68.5%** |
| Ensemble (SARIMAX+XGBoost+GRU+Chronos-2) | 1879.0 | 2133.9 | 5.97 | 6.19 | 0.74 | +42.4% |
| XGBoost (directo, Optuna) | 1885.7 | 2199.0 | 6.00 | 6.23 | 0.72 | +42.1% |
| SARIMAX | 2480.9 | 2836.9 | 8.10 | 8.43 | 0.53 | +21.9% |
| Chronos-2 (zero-shot + covariables) | 2527.5 | 2851.6 | 8.05 | 8.46 | 0.53 | +22.4% |

Con 10 años de histórico y clima multi-ciudad/HDD-CDD/precio/festivos regionales, el GRU
entrenado desde cero destroza al resto -- pasa de no existir en la comparativa a ganar con
mucho margen (68.5% de mejora, R²=0.91). SARIMAX, en cambio, **empeoró** al doblar el número
de variables exógenas (de 7 a 14) sin ampliar su ventana fija de entrenamiento (120 días) --
un caso real de sobreajuste/multicolinealidad en los coeficientes exógenos que los árboles y
la red manejan mucho mejor gracias a la regularización.

### % Generación renovable (horizonte 7 días)

| Modelo | MAE | RMSE | MAPE % | sMAPE % | R² | Mejora vs. baseline |
|---|---|---|---|---|---|---|
| Baseline | 5.29 | 6.48 | 10.54 | 9.74 | -2.65 | -- |
| **XGBoost (descompuesto por tecnología) 🏆** | 1.67 | 2.05 | **3.27** | 3.21 | 0.63 | **+69.0%** |
| Ensemble (SARIMAX+XGBoost+GRU+Chronos-2) | 1.47 | 2.08 | 2.93 | 2.86 | 0.62 | +72.2% |
| XGBoost (directo, Optuna) | 1.67 | 2.48 | 3.35 | 3.23 | 0.47 | +68.2% |
| Chronos-2 (zero-shot + covariables) | 1.74 | 2.58 | 3.52 | 3.39 | 0.42 | +66.6% |
| GRU (PyTorch, directo) | 1.86 | 2.41 | 3.69 | 3.67 | 0.50 | +65.0% |
| SARIMAX | 1.98 | 2.07 | 3.82 | 3.82 | 0.63 | +63.8% |

Todos los modelos "de verdad" quedan en un rango muy apretado (2.9%-3.8% MAPE, 63-72% de
mejora sobre el baseline) -- la señal de renovables mejoró tanto al arreglar el viento y
limpiar los bugs que la diferencia entre "el mejor" y "el peor" real ya es pequeña. El
Ensemble (2.93%) y la descomposición por tecnología (3.27%) quedan a 0.34 puntos, dentro de
la tolerancia de empate ya documentada (0.5 puntos) -- se prefiere la descomposición porque,
a igualdad (casi) de precisión, da el desglose por tecnología que el resto no ofrece.

### Un experimento con final honesto: descomponer por tecnología

Hipótesis: solar, eólica e hidráulica tienen dinámicas muy distintas (solar casi
determinista por el ciclo anual, eólica errática, hidráulica lenta), así que predecir cada
una por separado y sumarlas debería ganarle al modelo sobre el agregado. La primera vez que
se probó (antes de arreglar el viento) perdía por goleada -- el error de la pieza más
ruidosa (eólica) se acumulaba en la suma en vez de cancelarse. Arreglado el viento (regiones
eólicas reales + unidades correctas + los 10 años completos), la eólica mejoró mucho (MAE de
3.87 a ~1.7-2.2) y la brecha se cerró del todo: en la comparativa final, la descomposición
(3.27% MAPE) queda dentro del margen de empate del mejor modelo agregado (Ensemble, 2.93%) y
se adopta como campeón oficial de renovables -- no porque "gane" siempre (en pruebas
anteriores, con menos datos, perdía claramente), sino porque con datos limpios y suficientes
la hipótesis de partida se sostiene.

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
  spinner ciego, más un desglose por tecnología (solar/eólica/hidráulica/otras) de la
  predicción de % renovable, independiente de cuál sea el modelo campeón del agregado.
- **Metodología**: las explicaciones de este README pensadas para defenderlas en una
  entrevista, no solo para leerlas.

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Uso

```bash
pip install -r requirements.txt

python data.py      # descarga 10 años de demanda/generación (REE) + clima (Open-Meteo)
python train.py     # entrena y compara los modelos para los 2 objetivos
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

## Próximos pasos

El % renovable es diario porque esa es la resolución de la API pública de REE
(`apidatos`). Para tenerlo por hora habría que cambiar de fuente:
- **ESIOS** (`api.esios.ree.es`), la plataforma de indicadores de REE, sí tiene generación
  medida por tecnología a resolución horaria (y hasta un indicador directo de "% de
  Generación Renovable Peninsular") -- pero pedir el token de API exige un CIF de empresa,
  no vale para una cuenta personal.
- **ENTSO-E Transparency Platform** (el operador europeo) es la alternativa: registro
  abierto a particulares, con solicitud de acceso a la API por email. Pendiente de
  aprobación -- en cuanto esté disponible, se integraría igual que las demás fuentes.

## Limitaciones honestas

- SARIMAX para demanda usa un orden fijo, no una búsqueda automática completa: con
  estacionalidad de 24 horas sobre decenas de miles de puntos horarios, la búsqueda de
  `pmdarima.auto_arima` es intratable en CPU en tiempo razonable. Es una decisión de
  ingeniería documentada, no una limitación escondida.
- La previsión meteorológica a 7 días es, como cualquier previsión del tiempo, menos fiable
  cuanto más lejos mira -- por eso las métricas se reportan también por separado para el día
  1 y el día 7 del horizonte.
- El oleaje se probó como variable exploratoria (aportaba poca señal, como cabía esperar dado
  el peso casi nulo de la energía undimotriz en el mix español) y se retiró del todo al
  extender el histórico, cuando un hueco real en su cobertura empezó a romper el pipeline
  (ver "bugs reales" arriba).
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
