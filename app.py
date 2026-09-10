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

from data import CACHE_PATH, add_calendar_features, fetch_ree_demanda, fetch_ree_generacion

OUTPUT_DIR = Path("outputs")

st.set_page_config(page_title="Demanda eléctrica y renovables en España", layout="wide")

# Paleta pensada para el tema oscuro de Streamlit (ver .streamlit/config.toml) --
# "real" en blanco roto para que destaque siempre sobre el fondo oscuro, algo que
# el negro casi puro que se usa en las gráficas estáticas (fondo blanco) no hacía.
COLORS = {"real": "#f2f2f2", "predicción": "#4da3ff", "Baseline": "#9aa0a6", "SARIMAX": "#ff8a3d",
          "XGBoost": "#4da3ff", "GRU": "#ff5c7a", "Chronos-2": "#2dd4a7", "Ensemble": "#c874e0",
          "XGBoost (descompuesto)": "#ffc857"}
DASH = {"Baseline": "dot", "SARIMAX": "dash", "XGBoost": "dashdot", "GRU": "longdashdot", "Chronos-2": "dash",
        "Ensemble": "longdash", "XGBoost (descompuesto)": "dot"}

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


@st.cache_resource(show_spinner=False)
def load_chronos_pipeline():
    """Cachea el foundation model entre clicks del botón (y entre sesiones que
    comparten el mismo proceso) -- cargarlo desde HuggingFace tarda varios
    segundos y no cambia de una predicción a otra, así que repetirlo en cada
    click era puro desperdicio de CPU (justo lo que hace saltar el throttling
    de Streamlit Community Cloud)."""
    from chronos import Chronos2Pipeline
    return Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")


@st.cache_data(ttl=1800, show_spinner=False)
def load_recent_actuals(start_iso: str, end_iso: str) -> tuple[pd.Series, pd.Series]:
    """Demanda horaria y % renovable diario YA PUBLICADOS por REE entre `start_iso`
    y `end_iso` (fechas ISO, `end_iso` exclusivo) -- se usa para comparar, en la
    predicción "en tiempo real", lo que de verdad pasó en los días del horizonte
    que ya han quedado atrás contra lo que se había predicho para ellos. REE tarda
    unas horas en publicar los datos más recientes, así que la cola más reciente
    puede faltar todavía -- eso se traduce en huecos (NaN), no en un error."""
    from datetime import date as _date
    start_d, end_d = _date.fromisoformat(start_iso), _date.fromisoformat(end_iso)
    if start_d >= end_d:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    demanda = fetch_ree_demanda(start_d, end_d)
    generacion = fetch_ree_generacion(start_d, end_d)
    demanda_s = demanda.set_index("datetime")["demanda_mwh"] if not demanda.empty else pd.Series(dtype=float)
    renovable_s = pd.Series(dtype=float)
    if not generacion.empty:
        renovable_s = generacion.assign(date=pd.to_datetime(generacion["date"])).set_index("date")["renewable_pct"]
    return demanda_s, renovable_s


def _elapsed_mape(pred: pd.Series, real: pd.Series) -> float | None:
    """MAPE de la predicción sobre la parte del horizonte que ya se puede
    contrastar con datos reales -- solo sobre los puntos donde ambas series
    tienen valor (real ya publicado por REE), nunca sobre huecos."""
    common = pred.dropna().index.intersection(real.dropna().index)
    if len(common) == 0:
        return None
    actual, predicted = real.loc[common], pred.loc[common]
    nonzero = actual != 0
    if nonzero.sum() == 0:
        return None
    return float((abs(actual[nonzero] - predicted[nonzero]) / abs(actual[nonzero])).mean() * 100)


METADATA_KEYS = {"_campeon", "_campeon_backtest", "_backtest"}


def metrics_table(target_results: dict) -> pd.DataFrame:
    champion = target_results.get("_campeon")
    rows = []
    for name, vals in target_results.items():
        if name in METADATA_KEYS:
            continue
        rows.append({
            "Modelo": f"{name} (campeón)" if name == champion else name,
            "MAE": vals["MAE"], "RMSE": vals["RMSE"], "MAPE %": vals["MAPE_%"],
            "sMAPE %": vals["sMAPE_%"], "R²": vals["R2"], "Sesgo": vals["Bias"],
            "MAPE día 1 %": vals.get("MAPE_dia1_%"), "MAPE día 7 %": vals.get("MAPE_dia7_%"),
            "Mejora vs. baseline %": vals.get("mejora_vs_baseline_%", "--"),
        })
    return pd.DataFrame(rows)


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def line_chart(x, series_dict: dict, ylabel: str, hidden: set = frozenset(), band: dict = None) -> go.Figure:
    """`hidden`: nombres de serie que arrancan ocultas (clic en la leyenda para
    mostrarlas) -- por defecto se muestran real + baseline + el campeón, y el
    resto queda disponible pero oculto para no saturar la gráfica de entrada.
    `band`: {nombre_serie: (lower, upper)} -- pinta una banda de confianza
    (percentiles del backtest walk-forward) alrededor de esa serie, sin añadir
    entradas nuevas a la leyenda. El título se pinta con `chart_title` (un
    `st.markdown` fuera de la figura), no con el `title` de Plotly -- con
    leyendas de varias líneas en pantallas estrechas, el título interno de
    Plotly podía acabar solapado con la leyenda."""
    fig = go.Figure()
    band = band or {}
    for name, y in series_dict.items():
        color = COLORS.get(name, "#c9c9c9")
        if name in band and band[name][0] is not None:
            lower, upper = band[name]
            fig.add_trace(go.Scatter(x=x, y=lower, mode="lines", line=dict(width=0),
                                      showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=x, y=upper, mode="lines", line=dict(width=0), fill="tonexty",
                                      fillcolor=_hex_to_rgba(color, 0.15),
                                      showlegend=False, hoverinfo="skip", name=f"{name} (P10-P90)"))
        style = dict(color=color, width=3 if name == "real" else 2.2)
        if name in DASH:
            style["dash"] = DASH[name]
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines", name=name, line=style,
            visible="legendonly" if name in hidden else True,
        ))
    fig.update_layout(
        yaxis_title=ylabel, hovermode="x unified", template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.15), margin=dict(t=40, l=10, r=10), height=420,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
    return fig


def chart_title(title: str) -> None:
    """Título de gráfica pintado con Streamlit, no con el `title` interno de
    Plotly -- ver la nota en `line_chart`."""
    st.markdown(f"###### {title}")


def stacked_generation_chart(view: pd.DataFrame) -> go.Figure:
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
        yaxis_title="% del mix de generación", hovermode="x unified", template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.2), margin=dict(t=50, l=10, r=10), height=460,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)", range=[0, 100])
    return fig


# ---------------------------------------------------------------------------
# Cabecera
# ---------------------------------------------------------------------------

st.title("Demanda eléctrica y % renovable en España -- a 7 días vista")
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
    ["Histórico", "Comparativa de modelos", "Predicción en tiempo real", "Metodología"]
)

# ---------------------------------------------------------------------------
# Tab: histórico interactivo
# ---------------------------------------------------------------------------

with tab_historico:
    st.subheader("Explora el histórico")
    col1, col2 = st.columns([1, 3])
    with col1:
        variable = st.radio("Variable", ["Mix de generación (% renovable)", "Demanda eléctrica (MWh)"])
        min_d, max_d = history["datetime"].min().date(), history["datetime"].max().date()
        default_start = max_d.replace(year=max_d.year - 1)
        date_range = st.slider("Rango de fechas", min_value=min_d, max_value=max_d,
                                value=(default_start, max_d))
    mask = (history["datetime"].dt.date >= date_range[0]) & (history["datetime"].dt.date <= date_range[1])
    view = history.loc[mask]
    with col2:
        if variable.startswith("Demanda"):
            chart_title("Demanda eléctrica horaria")
            fig = line_chart(view["datetime"], {"real": view["demanda_mwh"]}, "MWh")
            st.plotly_chart(fig, use_container_width=True)
        else:
            chart_title("Mix de generación eléctrica por tecnología (% diario)")
            fig = stacked_generation_chart(view)
            st.plotly_chart(fig, use_container_width=True)
            daily_view = view.groupby(view["datetime"].dt.date)["renewable_pct"].first().reset_index()
            chart_title("% total de generación renovable (solar + eólica + hidráulica + otras)")
            fig2 = line_chart(daily_view["datetime"], {"real": daily_view["renewable_pct"]}, "%")
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
        interval_cols = {f"{champion}_lower", f"{champion}_upper"}
        series_dict = {c: holdout[c] for c in holdout.columns if c != "datetime" and c not in interval_cols}
        non_priority = {"real", "Baseline", champion}
        hidden = {name for name in series_dict if name not in non_priority}
        band = {}
        if interval_cols <= set(holdout.columns):
            band[champion] = (holdout[f"{champion}_lower"], holdout[f"{champion}_upper"])
        chart_title(f"{label}: real vs. predicción (últimos 7 días, holdout de evaluación) -- "
                    "clic en la leyenda para mostrar/ocultar modelos")
        fig = line_chart(holdout["datetime"], series_dict, ylabel, hidden=hidden, band=band)
        st.plotly_chart(fig, use_container_width=True)

        importancia_path = OUTPUT_DIR / f"{target_col}_importancia.png"
        if importancia_path.exists():
            with st.expander("Importancia de variables (XGBoost)"):
                st.image(str(importancia_path))

        backtest = results.get("_backtest")
        if backtest and backtest.get("agregado"):
            with st.expander(f"Backtesting: robustez con {backtest['n_folds']} ventanas "
                              "independientes (walk-forward)"):
                st.caption(
                    "Un único holdout de 7 días puede ser optimista o pesimista solo por suerte -- "
                    "aquí se evalúan varias ventanas independientes y consecutivas, con entrenamiento "
                    "expansivo. SARIMAX y XGBoost reutilizan el orden/hiperparámetros ya validados en "
                    "el holdout único (repetir la búsqueda en cada ventana sería intratable en CPU); "
                    "el GRU sí reentrena en cada ventana. \"XGBoost (descompuesto)\" queda fuera de "
                    "este backtest por simplicidad."
                )
                agregado_df = pd.DataFrame([
                    {"Modelo": name, "MAPE % (media)": vals["MAPE_%_mean"],
                     "MAPE % (desv.)": vals["MAPE_%_std"], "MAE (media)": vals["MAE_mean"],
                     "MAE (desv.)": vals["MAE_std"], "Ventanas OK": vals["n_folds_ok"]}
                    for name, vals in backtest["agregado"].items()
                ]).sort_values("MAPE % (media)")
                st.dataframe(agregado_df, use_container_width=True, hide_index=True)

                campeon_backtest = results.get("_campeon_backtest")
                if campeon_backtest and campeon_backtest != champion:
                    st.caption(
                        f"El campeón del holdout único (`{champion}`) no coincide con el del backtest "
                        f"multi-ventana (`{campeon_backtest}`) -- diferencia real entre evaluar una sola "
                        "semana y varias. La selección de modelo en producción sigue el holdout único."
                    )

                cobertura = backtest.get("cobertura_empirica_loo_%", {})
                cobertura_txt = ", ".join(f"{name}: {pct}%" for name, pct in cobertura.items()
                                           if pct is not None)
                if cobertura_txt:
                    st.caption(
                        "Cobertura real del intervalo de predicción (P10-P90, nominal ~80%), medida "
                        f"sin circularidad (leave-one-fold-out): {cobertura_txt}."
                    )

                fold_rows = []
                for f in backtest["folds"]:
                    row = {"Ventana": f["fold"], "Test": f"{f['test_inicio'][:10]} → {f['test_fin'][:10]}"}
                    for name in backtest["agregado"]:
                        if name in f:
                            row[name] = f[name]["MAPE_%"]
                    fold_rows.append(row)
                st.caption("MAPE % por ventana -- para ver si el error es consistente o varía mucho de una a otra.")
                st.dataframe(pd.DataFrame(fold_rows), use_container_width=True, hide_index=True)
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
pulsa el botón, tarda entre 1 y 3 minutos según qué modelo toque cargar, y se cachea
unas horas para no repetir el cálculo en cada visita). Las gráficas siempre muestran
los **3 últimos días** antes de hoy con **predicho y real a la vez** -- para esos días
no hay ninguna predicción guardada de antes, así que se reconstruye relanzando el
modelo campeón con el clima que de verdad ocurrió desde entonces (no una previsión) --
seguidos de lo que quede del horizonte por delante. Así "hoy" cae siempre en el mismo
punto de la gráfica y se ve de un vistazo cómo ha ido acertando la predicción.
"""
    )
    cached = load_cached_forecast()
    col_a, col_b = st.columns([1, 3])
    with col_a:
        refresh = st.button("Generar predicción ahora", use_container_width=True)

    forecast = cached
    if refresh:
        import predict
        log_box = st.status("Generando predicción...", expanded=True)
        t0 = time.time()
        try:
            def _progress(msg):
                log_box.write(f"`{time.time() - t0:5.1f}s` {msg}")

            chronos_pipeline = None
            if (predict.uses_chronos(metrics_data["demanda_mwh"])
                    or predict.uses_chronos(metrics_data["renewable_pct"])):
                _progress("Cargando Chronos-2 (cacheado tras la primera vez en este servidor)...")
                chronos_pipeline = load_chronos_pipeline()

            forecast = predict.build_forecast(progress=_progress, chronos_pipeline=chronos_pipeline)
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

        demanda_fut_full = pd.DataFrame(forecast["demanda_mwh"])
        demanda_fut_full["datetime"] = pd.to_datetime(
            demanda_fut_full["datetime"], utc=True).dt.tz_convert("Europe/Madrid")
        renovable_fut_full = pd.DataFrame(forecast["renovable_pct"])
        renovable_fut_full["date_dt"] = pd.to_datetime(renovable_fut_full["date"])

        # Para los días ya pasados se usa el "retrospectivo" que calcula
        # predict.py (relanza el modelo campeón con el clima real ya ocurrido),
        # para mostrar predicho y real también ahí. Forecasts guardados antes
        # de este campo simplemente no lo tienen (`.get(..., [])`).
        retro_demanda = pd.DataFrame(forecast.get("retro_demanda_mwh", []))
        if not retro_demanda.empty:
            retro_demanda["datetime"] = pd.to_datetime(
                retro_demanda["datetime"], utc=True).dt.tz_convert("Europe/Madrid")
            demanda_fut_full = pd.concat(
                [retro_demanda[retro_demanda["datetime"] < demanda_fut_full["datetime"].min()], demanda_fut_full],
                ignore_index=True)

        retro_renovable = pd.DataFrame(forecast.get("retro_renovable_pct", []))
        if not retro_renovable.empty:
            retro_renovable["date_dt"] = pd.to_datetime(retro_renovable["date"])
            renovable_fut_full = pd.concat(
                [retro_renovable[retro_renovable["date_dt"] < renovable_fut_full["date_dt"].min()],
                 renovable_fut_full],
                ignore_index=True)

        # Eje anclado en "hoy": 3 días hacia atrás más lo que quede del
        # horizonte hacia delante, para que "hoy" caiga siempre en la misma
        # posición sin importar cuándo se generó la predicción.
        today_date = pd.Timestamp.now(tz="Europe/Madrid").date()
        today = pd.Timestamp(today_date, tz="Europe/Madrid")
        window_start = today - pd.Timedelta(days=3)
        demanda_real, renovable_real = load_recent_actuals(str(window_start.date()), str(today_date))
        # El % renovable de "hoy" que da REE es un acumulado a medias del día
        # en curso, no el valor final -- se descarta (igual que
        # `trim_incomplete_trailing_days` en el histórico).
        renovable_real = renovable_real[renovable_real.index < pd.Timestamp(today_date)]

        demanda_end = max(demanda_fut_full["datetime"].max(), today + pd.Timedelta(hours=23))
        full_hours = pd.date_range(window_start, demanda_end, freq="h")
        demanda_indexed = demanda_fut_full.set_index("datetime")
        demanda_pred_s = demanda_indexed["mwh"].reindex(full_hours)
        demanda_real_s = demanda_real.reindex(full_hours)

        renovable_end = max(renovable_fut_full["date_dt"].max(), pd.Timestamp(today_date))
        full_days = pd.date_range(window_start.tz_localize(None).normalize(), renovable_end, freq="D")
        renovable_indexed = renovable_fut_full.set_index("date_dt")
        renovable_pred_s = renovable_indexed["pct"].reindex(full_days)
        renovable_real_s = renovable_real.reindex(full_days)

        # Banda de confianza (P10-P90) del backtest walk-forward -- solo
        # existe para el tramo hacia delante, el retrospectivo no lleva
        # intervalo propio (ver predict.py).
        demanda_band = {"predicción": (
            demanda_indexed["mwh_p10"].reindex(full_hours) if "mwh_p10" in demanda_indexed.columns else None,
            demanda_indexed["mwh_p90"].reindex(full_hours) if "mwh_p90" in demanda_indexed.columns else None,
        )}
        renovable_band = {"predicción": (
            renovable_indexed["pct_p10"].reindex(full_days) if "pct_p10" in renovable_indexed.columns else None,
            renovable_indexed["pct_p90"].reindex(full_days) if "pct_p90" in renovable_indexed.columns else None,
        )}

        chart_title("Demanda eléctrica: últimos 3 días + predicción a lo que queda de horizonte")
        fig_d = line_chart(full_hours, {"real": demanda_real_s, "predicción": demanda_pred_s}, "MWh",
                            band=demanda_band)
        st.plotly_chart(fig_d, use_container_width=True)

        chart_title("% de generación renovable: últimos 3 días + predicción")
        fig_r = line_chart(full_days, {"real": renovable_real_s, "predicción": renovable_pred_s}, "%",
                            band=renovable_band)
        st.plotly_chart(fig_r, use_container_width=True)

        if forecast.get("intervalo_metodo"):
            cobertura_d = (metrics_data.get("demanda_mwh", {}).get("_backtest", {})
                           .get("cobertura_empirica_loo_%", {}).get(forecast["modelo_demanda"]))
            cobertura_r = (metrics_data.get("renewable_pct", {}).get("_backtest", {})
                           .get("cobertura_empirica_loo_%", {}).get(forecast["modelo_renovable"]))
            partes = ["La banda sombreada es un intervalo de predicción (P10-P90, ~80% nominal) "
                      "calculado a partir de los residuos del backtest walk-forward, no una previsión "
                      "de otro modelo."]
            if cobertura_d is not None:
                partes.append(f"Cobertura real medida en demanda: {cobertura_d}%.")
            if cobertura_r is not None:
                partes.append(f"En % renovable: {cobertura_r}%.")
            st.caption(" ".join(partes))

        mape_demanda = _elapsed_mape(demanda_pred_s, demanda_real_s)
        mape_renovable = _elapsed_mape(renovable_pred_s, renovable_real_s)
        if mape_demanda is not None or mape_renovable is not None:
            st.caption("**Error real de esta predicción, medido sobre los días del horizonte que ya han "
                       "pasado** (comparado contra lo publicado por REE, no contra el holdout de "
                       "entrenamiento):")
            col_m1, col_m2 = st.columns(2)
            if mape_demanda is not None:
                col_m1.metric("MAPE demanda (transcurrido)", f"{mape_demanda:.1f}%")
            if mape_renovable is not None:
                col_m2.metric("MAPE % renovable (transcurrido)", f"{mape_renovable:.1f}%")

        if forecast.get("desglose_tecnologia"):
            st.markdown("#### Desglose por tecnología (solar, eólica, hidráulica, otras)")
            st.caption("Cada tecnología se predice por separado y se muestra tal cual -- no es la "
                       "descomposición la que gana la comparativa de precisión (ver pestaña anterior), "
                       "pero ver el reparto por fuente es útil aunque el modelo campeón sea el que "
                       "predice el % total de una sola vez.")
            tech_df = pd.DataFrame(forecast["desglose_tecnologia"])
            fig_tech = go.Figure()
            tech_labels = {"solar_pct": "Solar", "eolica_pct": "Eólica",
                           "hidraulica_pct": "Hidráulica", "otras_pct": "Otras renovables"}
            for col, label in tech_labels.items():
                if col in tech_df.columns:
                    fig_tech.add_trace(go.Bar(x=tech_df["date"], y=tech_df[col], name=label,
                                               marker_color=BUCKET_COLORS.get(label)))
            fig_tech.update_layout(
                barmode="stack", yaxis_title="%",
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", y=1.15), margin=dict(t=40, l=10, r=10), height=420,
            )
            chart_title("% renovable previsto, por tecnología")
            st.plotly_chart(fig_tech, use_container_width=True)

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

### Backtesting: por qué un solo holdout no basta
El holdout de 7 días de la pestaña "Comparativa de modelos" es una sola muestra -- si esa
semana en concreto fue rara (una ola de calor, un puente festivo largo), el MAPE que sale
puede ser optimista o pesimista solo por suerte, no porque el modelo sea mejor o peor de
verdad. Por eso, además, se hace un **backtest walk-forward**: se evalúan varias ventanas de
7 días consecutivas al final de la serie (5 en demanda, 8 en % renovable), cada una con su
propio entrenamiento expansivo (todo lo anterior a esa ventana). SARIMAX y XGBoost reutilizan
en cada ventana el orden/hiperparámetros ya validados en el holdout único -- repetir esa
búsqueda en cada ventana sería intratable en CPU, y no es lo que se está midiendo aquí. El
GRU sí reentrena y recalcula su normalización en cada ventana, porque esas estadísticas
salen de los datos de esa ventana en concreto; reutilizar las del entrenamiento completo
sería una fuga de información hacia el pasado. El resultado (media ± desviación del MAPE
sobre varias ventanas) está en un desplegable bajo cada gráfica de la pestaña de
comparativa.

### Intervalos de predicción (P10-P90)
Hasta aquí todo eran predicciones puntuales -- un único número, sin decir qué tan seguro
está el modelo de él. La banda sombreada que aparece en las gráficas es un intervalo de
predicción calculado a partir de los residuos (real - predicho) del propio backtest
walk-forward: se agrupan por día del horizonte (o por paso, en % renovable) y se toman los
percentiles 10 y 90 -- el mismo método para todos los modelos, para que sea comparable. La
cobertura real (¿el intervalo nominal del 80% cubre de verdad ~80% de las observaciones?) se
mide sin trampa: el intervalo de cada ventana se calcula solo con las OTRAS ventanas
(leave-one-fold-out), nunca con sus propios datos. El número sale donde salga -- si el
intervalo queda mal calibrado, se reporta así, no se ajusta a posteriori para que cuadre.

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
- El backtest walk-forward usa solo 5-8 ventanas -- suficiente para ver si el error es
  consistente o varía mucho, pero los percentiles de los intervalos de predicción salen de
  relativamente pocas muestras (sobre todo en % renovable, 8 ventanas de 7 días): son
  estimaciones útiles, no un intervalo estadísticamente muy afinado.
- "XGBoost (descompuesto)" (la suma de 4 sub-modelos por tecnología) se queda fuera del
  backtest walk-forward por simplicidad -- sí se sigue mostrando en la comparativa del
  holdout único y en el desglose por tecnología de la predicción en vivo.
"""
    )
