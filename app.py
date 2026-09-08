"""Dashboard de Streamlit: histórico interactivo, comparativa de los 4 modelos
(con métricas y gráficas interactivas, no solo un número suelto) y una sección
de predicción "en tiempo real" -- entre comillas porque no es un stream
continuo, es un recálculo bajo demanda con la previsión de clima más reciente
disponible -- para demanda eléctrica y % de generación renovable a 7 días.
"""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data import CACHE_PATH, add_calendar_features

OUTPUT_DIR = Path("outputs")

st.set_page_config(page_title="Demanda eléctrica y renovables en España", page_icon="⚡", layout="wide")

COLORS = {"real": "#0b0b0b", "Baseline": "#898781", "SARIMAX": "#eb6834",
          "XGBoost": "#2a78d6", "Chronos-2": "#1baf7a"}


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
    df = pd.read_csv(OUTPUT_DIR / f"{target_col}_predicciones_holdout.csv", parse_dates=["datetime"])
    return df


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_cached_forecast() -> dict | None:
    path = OUTPUT_DIR / "latest_forecast.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data(ttl=6 * 3600, show_spinner="Generando predicción con clima real de los próximos 7 días...")
def generate_live_forecast() -> dict:
    import predict
    result = predict.build_forecast()
    with open(OUTPUT_DIR / "latest_forecast.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    return result


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


def line_chart(x, series_dict: dict, title: str, ylabel: str) -> go.Figure:
    fig = go.Figure()
    for name, y in series_dict.items():
        style = dict(color=COLORS.get(name, "#999999"))
        if name != "real":
            style["dash"] = "dash"
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name, line=style))
    fig.update_layout(title=title, yaxis_title=ylabel, hovermode="x unified",
                       legend=dict(orientation="h", y=1.1), margin=dict(t=60))
    return fig


# ---------------------------------------------------------------------------
# Cabecera
# ---------------------------------------------------------------------------

st.title("⚡ Demanda eléctrica y % renovable en España -- a 7 días vista")
st.markdown(
    """
Proyecto de forecasting con datos 100% reales y públicos: demanda y mix de generación de
[REE (Red Eléctrica de España)](https://www.ree.es/) combinados con previsión meteorológica de
[Open-Meteo](https://open-meteo.com/) (radiación solar, viento, oleaje...). Se comparan 4 enfoques
de forecasting -- desde un baseline ingenuo hasta un *foundation model* de series temporales -- y
se usa, para cada objetivo, el que de verdad funciona mejor.
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
        variable = st.radio("Variable", ["Demanda eléctrica (MWh)", "% Generación renovable"])
        min_d, max_d = history["datetime"].min().date(), history["datetime"].max().date()
        date_range = st.slider("Rango de fechas", min_value=min_d, max_value=max_d,
                                value=(max_d.replace(year=max_d.year - 1), max_d))
    mask = (history["datetime"].dt.date >= date_range[0]) & (history["datetime"].dt.date <= date_range[1])
    view = history.loc[mask]
    with col2:
        if variable.startswith("Demanda"):
            fig = line_chart(view["datetime"], {"real": view["demanda_mwh"]},
                              "Demanda eléctrica horaria", "MWh")
        else:
            daily_view = view.groupby(view["datetime"].dt.date)["renewable_pct"].first().reset_index()
            fig = line_chart(daily_view["datetime"], {"real": daily_view["renewable_pct"]},
                              "% de generación renovable (diario)", "%")
        st.plotly_chart(fig, use_container_width=True)
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
        st.dataframe(metrics_table(results), use_container_width=True, hide_index=True)

        holdout = load_holdout(target_col)
        series_dict = {c: holdout[c] for c in holdout.columns if c != "datetime"}
        fig = line_chart(holdout["datetime"], series_dict,
                          f"{label}: real vs. predicción (últimos 7 días, holdout de evaluación)", ylabel)
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
esconderlo, es parte de hacer esto bien.
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
pulsa el botón, y se cachea unas horas para no reentrenar en cada visita).
"""
    )
    cached = load_cached_forecast()
    col_a, col_b = st.columns([1, 3])
    with col_a:
        refresh = st.button("🔄 Generar predicción ahora", use_container_width=True)
    if refresh:
        try:
            forecast = generate_live_forecast()
        except Exception as exc:
            st.error(f"No se pudo generar la predicción en vivo ahora mismo ({exc}). "
                     "Se muestra la última predicción guardada.")
            forecast = cached
    else:
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
  tecnología (de ahí sale el % renovable real, no una estimación).
- **Open-Meteo**: radiación solar, viento (a 10m y a 100m, la altura real de buje de un
  aerogenerador), precipitación, temperatura, humedad, presión y oleaje (Open-Meteo Marine).
  Todo gratis, sin necesidad de API key.

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
La demanda horaria tiene ~17.500-40.000 puntos con un patrón diario y semanal muy marcado y
estable -- ahí un modelo estadístico clásico bien especificado (SARIMAX) explota ese patrón
mejor que nadie. El % renovable diario tiene muchísimos menos puntos y depende del clima
día a día, que es más ruidoso -- ahí SARIMAX y XGBoost, que tienen que aprender el patrón
desde cero solo con estos datos, a veces ni superan al baseline ingenuo, mientras que
Chronos-2, que ya "sabe" de series temporales en general por su preentrenamiento, generaliza
mejor con menos datos específicos. Por eso la app usa el campeón de cada objetivo por
separado en vez de forzar un único modelo para todo.

### Limitaciones honestas
- El oleaje ("fuerza del mar") se probó porque España tiene generación undimotriz real, pero
  es una instalación piloto casi anecdótica en el mix nacional -- se esperaba (y se confirma
  en la importancia de variables) que aporte poca señal frente a sol y viento.
- SARIMAX para demanda usa un orden fijo, no una búsqueda automática completa: con
  estacionalidad de 24 horas sobre miles de puntos, la búsqueda automática de pmdarima es
  intratable en CPU en tiempo razonable. Es una decisión de ingeniería documentada, no un
  intento de esconder una limitación.
- La sección "en tiempo real" usa la previsión meteorológica de Open-Meteo a 7 días, que como
  cualquier previsión del tiempo es menos fiable cuanto más lejos mira -- por eso las métricas
  del modelo también se reportan por separado para el día 1 y el día 7 del horizonte.
"""
    )
