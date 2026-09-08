"""Dashboard de Streamlit: histórico interactivo (incluida una gráfica de mix de
generación tipo stacked-area, el estándar visual del sector), comparativa de los
4-5 modelos (con métricas y gráficas interactivas, no solo un número suelto) y una
sección de predicción "en tiempo real" -- entre comillas porque no es un stream
continuo, es un recálculo bajo demanda con la previsión de clima más reciente
disponible, con progreso visible paso a paso -- para demanda eléctrica y % de
generación renovable a 7 días.
"""

import json
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data import CACHE_PATH, add_calendar_features

OUTPUT_DIR = Path("outputs")

st.set_page_config(page_title="Demanda eléctrica y renovables en España", page_icon="⚡", layout="wide")

# Paleta pensada para el tema oscuro de Streamlit (ver .streamlit/config.toml) --
# "real" en blanco roto para que destaque siempre sobre el fondo oscuro, algo que
# el negro casi puro que se usa en las gráficas estáticas (fondo blanco) no hacía.
COLORS = {"real": "#f2f2f2", "Baseline": "#9aa0a6", "SARIMAX": "#ff8a3d",
          "XGBoost": "#4da3ff", "Chronos-2": "#2dd4a7", "Ensemble": "#c874e0"}
DASH = {"Baseline": "dot", "SARIMAX": "dash", "XGBoost": "dashdot", "Chronos-2": "dash", "Ensemble": "longdash"}

TECH_BUCKETS = {
    "Solar": ["tech_solar_fv_pct", "tech_solar_termica_pct"],
    "Eólica": ["tech_eolica_pct"],
    "Hidráulica": ["tech_hidraulica_pct"],
    "Otras renovables": ["tech_otras_renovables_pct", "tech_residuos_renovables_pct"],
    "Nuclear": ["tech_nuclear_pct"],
    "No renovable (resto)": ["tech_carbon_pct", "tech_fuel_gas_pct", "tech_turbina_vapor_pct",
                              "tech_ciclo_combinado_pct", "tech_cogeneracion_pct",
                              "tech_residuos_no_renovables_pct"],
}
BUCKET_COLORS = {"Solar": "#ffc857", "Eólica": "#4da3ff", "Hidráulica": "#2d7dd2",
                  "Otras renovables": "#2dd4a7", "Nuclear": "#c874e0", "No renovable (resto)": "#5a5f6b"}


# ---------------------------------------------------------------------------
# Carga de datos (cacheada -- no se relee del disco en cada interacción)
# ---------------------------------------------------------------------------

@st.cache_data
def load_history() -> pd.DataFrame:
    df = pd.read_csv(CACHE_PATH, parse_dates=["datetime"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert("Europe/Madrid")
    return df


@st.cache_data
def load_metrics() -> dict:
    with open(OUTPUT_DIR / "metrics.json", encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data
def load_holdout(target_col: str) -> pd.DataFrame:
    return pd.read_csv(OUTPUT_DIR / f"{target_col}_predicciones_holdout.csv", parse_dates=["datetime"])


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_cached_forecast() -> dict | None:
    path = OUTPUT_DIR / "latest_forecast.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def metrics_table(target_results: dict) -> pd.DataFrame:
    champion = target_results.get("_campeon")
    rows = []
    for name, vals in target_results.items():
        if name == "_campeon":
            continue
        rows.append({
            "Modelo": f"{name} 🏆" if name == champion else name,
            "MAE": vals["MAE"], "RMSE": vals["RMSE"], "MAPE %": vals["MAPE_%"],
            "sMAPE %": vals["sMAPE_%"], "R²": vals["R2"], "Sesgo": vals["Bias"],
            "MAPE día 1 %": vals.get("MAPE_dia1_%"), "MAPE día 7 %": vals.get("MAPE_dia7_%"),
            "Mejora vs. baseline %": vals.get("mejora_vs_baseline_%", "--"),
        })
    return pd.DataFrame(rows)


def line_chart(x, series_dict: dict, title: str, ylabel: str, hidden: set = frozenset()) -> go.Figure:
    """`hidden`: nombres de serie que arrancan ocultas (clic en la leyenda para
    mostrarlas) -- por defecto se muestran real + baseline + el campeón, y el
    resto queda disponible pero oculto para no saturar la gráfica de entrada."""
    fig = go.Figure()
    for name, y in series_dict.items():
        style = dict(color=COLORS.get(name, "#c9c9c9"), width=3 if name == "real" else 2.2)
        if name in DASH:
            style["dash"] = DASH[name]
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines", name=name, line=style,
            visible="legendonly" if name in hidden else True,
        ))
    fig.update_layout(
        title=title, yaxis_title=ylabel, hovermode="x unified", template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.12), margin=dict(t=60, l=10, r=10), height=440,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
    return fig


def stacked_generation_chart(view: pd.DataFrame, title: str) -> go.Figure:
    daily = view.groupby(view["datetime"].dt.date).first(numeric_only=True).reset_index()
    fig = go.Figure()
    for bucket, cols in TECH_BUCKETS.items():
        present = [c for c in cols if c in daily.columns]
        if not present:
            continue
        values = daily[present].sum(axis=1)
        fig.add_trace(go.Scatter(
            x=daily["datetime"], y=values, mode="lines", name=bucket, stackgroup="mix",
            line=dict(width=0.5, color=BUCKET_COLORS.get(bucket, "#888")),
            fillcolor=BUCKET_COLORS.get(bucket, "#888"),
        ))
    fig.update_layout(
        title=title, yaxis_title="% del mix de generación", hovermode="x unified", template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.15), margin=dict(t=60, l=10, r=10), height=460,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)", range=[0, 100])
    return fig


# ---------------------------------------------------------------------------
# Cabecera
# ---------------------------------------------------------------------------

st.title("⚡ Demanda eléctrica y % renovable en España -- a 7 días vista")
st.markdown(
    """
Proyecto de forecasting con datos 100% reales y públicos: demanda y mix de generación de
[REE (Red Eléctrica de España)](https://www.ree.es/) combinados con previsión meteorológica de
[Open-Meteo](https://open-meteo.com/) (radiación solar, viento, oleaje...). Se comparan varios
enfoques de forecasting -- desde un baseline ingenuo hasta un *foundation model* de series
temporales -- y se usa, para cada objetivo, el que de verdad funciona mejor.
"""
)

history = load_history()
metrics_data = load_metrics()

tab_historico, tab_modelos, tab_vivo, tab_metodologia = st.tabs(
    ["📈 Histórico", "🏁 Comparativa de modelos", "🔮 Predicción en tiempo real", "🧠 Metodología"]
)

# ---------------------------------------------------------------------------
# Tab: histórico interactivo
# ---------------------------------------------------------------------------

with tab_historico:
    st.subheader("Explora el histórico")
    col1, col2 = st.columns([1, 3])
    with col1:
        variable = st.radio("Variable", ["Demanda eléctrica (MWh)", "Mix de generación (% renovable)"])
        min_d, max_d = history["datetime"].min().date(), history["datetime"].max().date()
        default_start = max_d.replace(year=max_d.year - 1)
        date_range = st.slider("Rango de fechas", min_value=min_d, max_value=max_d,
                                value=(default_start, max_d))
    mask = (history["datetime"].dt.date >= date_range[0]) & (history["datetime"].dt.date <= date_range[1])
    view = history.loc[mask]
    with col2:
        if variable.startswith("Demanda"):
            fig = line_chart(view["datetime"], {"real": view["demanda_mwh"]},
                              "Demanda eléctrica horaria", "MWh")
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = stacked_generation_chart(view, "Mix de generación eléctrica por tecnología (% diario)")
            st.plotly_chart(fig, use_container_width=True)
            daily_view = view.groupby(view["datetime"].dt.date)["renewable_pct"].first().reset_index()
            fig2 = line_chart(daily_view["datetime"], {"real": daily_view["renewable_pct"]},
                               "% total de generación renovable (solar + eólica + hidráulica + otras)", "%")
            st.plotly_chart(fig2, use_container_width=True)
    st.caption(f"Datos desde {min_d} hasta {max_d} -- {len(history):,} filas horarias.".replace(",", "."))

# ---------------------------------------------------------------------------
# Tab: comparativa de modelos
# ---------------------------------------------------------------------------

with tab_modelos:
    for target_col, label, ylabel in [("demanda_mwh", "Demanda eléctrica (MWh/h)", "MWh"),
                                       ("renewable_pct", "% Generación renovable", "%")]:
        st.subheader(label)
        results = metrics_data[target_col]
        champion = results["_campeon"]
        st.markdown(f"**Modelo campeón para este objetivo: `{champion}`** "
                    f"(MAPE {results[champion]['MAPE_%']}%)")
        if champion == "Ensemble":
            st.caption(f"Ensemble = media de: {', '.join(results['Ensemble'].get('compuesto_por', []))}")
        st.dataframe(metrics_table(results), use_container_width=True, hide_index=True)

        holdout = load_holdout(target_col)
        series_dict = {c: holdout[c] for c in holdout.columns if c != "datetime"}
        non_priority = {"real", "Baseline", champion}
        hidden = {name for name in series_dict if name not in non_priority}
        fig = line_chart(holdout["datetime"], series_dict,
                          f"{label}: real vs. predicción (últimos 7 días, holdout de evaluación) -- "
                          "clic en la leyenda para mostrar/ocultar modelos",
                          ylabel, hidden=hidden)
        st.plotly_chart(fig, use_container_width=True)

        importancia_path = OUTPUT_DIR / f"{target_col}_importancia.png"
        if importancia_path.exists():
            with st.expander("Importancia de variables (XGBoost)"):
                st.image(str(importancia_path))
        st.divider()

    st.markdown(
        """
**Cómo leer "mejora vs. baseline":** el baseline es "el mismo valor que hace una semana" --
la referencia mínima que cualquier modelo serio tiene que superar. Un modelo con mejora
negativa (peor que el baseline) no es un modelo roto: es una señal real de que, para esa
serie en concreto, con esos datos, el patrón es demasiado ruidoso o hay pocos datos para que
ese modelo aprenda algo mejor que "repetir la semana pasada" -- y reportarlo así, en vez de
esconderlo, es parte de hacer esto bien. El **Ensemble** es la media de los modelos "reales"
(no el baseline): promediar modelos con errores distintos entre sí suele reducir el error
total, y solo se convierte en el modelo campeón si de verdad mejora sobre los demás.
"""
    )

# ---------------------------------------------------------------------------
# Tab: predicción en tiempo real
# ---------------------------------------------------------------------------

with tab_vivo:
    st.subheader("Predicción a 7 días con datos y clima actuales")
    st.markdown(
        """
Esta sección **recalcula la predicción bajo demanda**, combinando el histórico real más
reciente con la previsión meteorológica real de Open-Meteo para los próximos 7 días
(no es un stream continuo -- de ahí las comillas en "tiempo real": se genera cuando se
pulsa el botón, tarda entre 30 segundos y 2 minutos según qué modelo toque cargar, y se
cachea unas horas para no repetir el cálculo en cada visita).
"""
    )
    cached = load_cached_forecast()
    col_a, col_b = st.columns([1, 3])
    with col_a:
        refresh = st.button("🔄 Generar predicción ahora", use_container_width=True)

    forecast = cached
    if refresh:
        import predict
        log_box = st.status("Generando predicción...", expanded=True)
        t0 = time.time()
        try:
            def _progress(msg):
                log_box.write(f"`{time.time() - t0:5.1f}s` {msg}")

            forecast = predict.build_forecast(progress=_progress)
            with open(OUTPUT_DIR / "latest_forecast.json", "w", encoding="utf-8") as fh:
                json.dump(forecast, fh, indent=2, ensure_ascii=False)
            load_cached_forecast.clear()
            log_box.update(label=f"Predicción lista en {time.time() - t0:.0f} segundos", state="complete")
        except Exception as exc:
            log_box.update(label=f"Falló la predicción en vivo ({exc}) -- se muestra la última guardada",
                            state="error")
            forecast = cached

    if forecast is None:
        st.info("Todavía no hay ninguna predicción generada. Pulsa el botón de arriba.")
    else:
        st.caption(f"Generada: {forecast['generado_en']} · Histórico real hasta: "
                   f"{forecast['historico_hasta']} · Modelo demanda: **{forecast['modelo_demanda']}** · "
                   f"Modelo % renovable: **{forecast['modelo_renovable']}**")

        resumen = pd.DataFrame(forecast["resumen_diario"])
        st.markdown("#### Resumen diario: demanda prevista y cobertura renovable")
        st.dataframe(resumen, use_container_width=True, hide_index=True)

        demanda_fut = pd.DataFrame(forecast["demanda_mwh"])
        demanda_fut["datetime"] = pd.to_datetime(demanda_fut["datetime"])
        recent_history = history[history["datetime"] >= history["datetime"].max() - pd.Timedelta(days=7)]
        fig_d = line_chart(
            pd.concat([recent_history["datetime"], demanda_fut["datetime"]]),
            {"real (últimos 7 días)": pd.concat(
                [recent_history["demanda_mwh"], pd.Series([None] * len(demanda_fut))]),
             "predicción próximos 7 días": pd.concat(
                [pd.Series([None] * len(recent_history)), demanda_fut["mwh"]])},
            "Demanda eléctrica: contexto reciente + predicción a 7 días", "MWh")
        st.plotly_chart(fig_d, use_container_width=True)

        renovable_fut = pd.DataFrame(forecast["renovable_pct"])
        fig_r = line_chart(renovable_fut["date"], {"predicción % renovable": renovable_fut["pct"]},
                            "% de generación renovable prevista", "%")
        st.plotly_chart(fig_r, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab: metodología / comprensión
# ---------------------------------------------------------------------------

with tab_metodologia:
    st.markdown(
        """
### Fuentes de datos
- **REE (apidatos.ree.es)**: demanda eléctrica horaria y mix de generación real por
  tecnología (de ahí sale el % renovable real, no una estimación) -- Solar fotovoltaica,
  Solar térmica, Eólica, Hidráulica, Nuclear, Ciclo combinado, Carbón, Cogeneración y el
  resto de tecnologías del sistema peninsular, día a día (REE no publica el mix de
  generación con resolución horaria en su API pública, a diferencia de la demanda).
- **Open-Meteo**: radiación solar (media y máxima diaria), nubosidad, viento a 10m y a
  100m (la altura real de buje de un aerogenerador, con media y máxima diaria),
  precipitación, temperatura, humedad, presión y oleaje (Open-Meteo Marine). Todo gratis,
  sin necesidad de API key.
- **Ciclo anual (seno/coseno del día del año)**: con 5 años de histórico ya hay ciclos
  anuales completos de sobra para que el modelo aprenda que el % solar depende sobre todo
  de la época del año -- una señal mucho más estable que la meteorología día a día.

### Por qué el holdout es temporal, no aleatorio
Si se evaluase con fechas elegidas al azar, el modelo podría entrenar con datos de *después*
del punto que se le pide predecir -- literalmente vería el futuro. En una serie temporal el
único holdout honesto es "los últimos N días", igual que en producción: en el momento de
predecir, el futuro sencillamente no existe todavía.

### Qué significa que Chronos-2 sea "zero-shot"
SARIMAX y XGBoost se entrenan específicamente con estos datos: aprenden los patrones de la
demanda española o de la generación renovable española y solo sirven para eso. Chronos-2 es
un *foundation model* -- se pre-entrenó una sola vez con series temporales muy variadas (nada
que ver con electricidad española) y aquí se usa tal cual, sin reentrenarlo, solo dándole el
histórico reciente como contexto y el clima previsto como variable adicional. Que compita con
modelos entrenados a medida (y que incluso gane en % renovable) es el resultado interesante:
no hace falta reentrenar nada para tener un punto de partida competitivo.

### Por qué gana un modelo distinto en cada objetivo
La demanda horaria tiene decenas de miles de puntos con un patrón diario y semanal muy
marcado y estable -- ahí un modelo estadístico clásico bien especificado (SARIMAX) explota
ese patrón mejor que nadie. El % renovable diario tiene muchísimos menos puntos (la API de
REE solo publica el mix de generación a nivel diario) y depende del clima día a día, que es
más ruidoso -- ahí SARIMAX y XGBoost, que tienen que aprender el patrón desde cero solo con
estos datos, a veces ni superan al baseline ingenuo, mientras que Chronos-2, que ya "sabe"
de series temporales en general por su preentrenamiento, generaliza mejor con menos datos
específicos. Por eso la app usa el campeón de cada objetivo por separado en vez de forzar un
único modelo para todo -- y por eso, cuando promediar ayuda, el campeón es el Ensemble.

### Limitaciones honestas
- El oleaje ("fuerza del mar") se probó porque España tiene generación undimotriz real, pero
  es una instalación piloto casi anecdótica en el mix nacional -- se esperaba (y se confirma
  en la importancia de variables) que aporte poca señal frente a sol y viento.
- SARIMAX para demanda usa un orden fijo, no una búsqueda automática completa: con
  estacionalidad de 24 horas sobre decenas de miles de puntos, la búsqueda automática de
  pmdarima es intratable en CPU en tiempo razonable. Es una decisión de ingeniería
  documentada, no un intento de esconder una limitación.
- La sección "en tiempo real" usa la previsión meteorológica de Open-Meteo a 7 días, que como
  cualquier previsión del tiempo es menos fiable cuanto más lejos mira -- por eso las métricas
  del modelo también se reportan por separado para el día 1 y el día 7 del horizonte.
- El % renovable es una serie diaria (no horaria) porque así lo publica REE en su API
  pública -- con datos horarios de generación por tecnología el modelo probablemente
  mejoraría bastante más, capturando el patrón diario de subida y bajada solar.
"""
    )
