"""Genera la predicción a 7 días "en vivo": parte del histórico cacheado por
data.py (demanda + generación real de REE, clima histórico) y lo combina con la
previsión meteorológica real de los próximos 7 días (Open-Meteo /forecast, no el
archivo histórico) para predecir demanda y % renovable a futuro de verdad -- no
solo reproducir el holdout de evaluación de train.py.

Usa, para cada objetivo, el modelo que salió "campeón" (menor MAPE) en
outputs/metrics.json -- generado por train.py --, no siempre el mismo modelo
para todo: si SARIMAX gana en demanda y Chronos-2 gana en % renovable, aquí se
respeta esa elección en vez de forzar un único modelo para las dos cosas. Si el
campeón es "Ensemble", se calculan los modelos que lo componen y se promedian --
reutilizando siempre los hiperparámetros/orden ya validados en train.py, nunca
se vuelve a lanzar Optuna ni una búsqueda de auto_arima en cada predicción.

Se guarda en outputs/latest_forecast.json, que es lo que lee la app de
Streamlit para la sección de predicción "en tiempo real" (entre comillas
porque no es un stream continuo: se recalcula cuando se pide, con la previsión
de clima más reciente disponible).
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from data import CACHE_PATH, add_calendar_features, fetch_weather, fetch_marine
from train import (
    OUTPUT_DIR, sarimax_auto_forecast, xgb_recursive_forecast,
    chronos_covariate_forecast, baseline_seasonal,
)

DEMAND_FEATURES = ["temperature_2m", "relative_humidity_2m", "shortwave_radiation",
                    "wind_speed_10m", "wind_speed_100m", "is_weekend", "is_holiday"]
DEMAND_LAGS = [24, 48, 168, 336]
DEMAND_ROLLS = [24, 168]
DEMAND_SARIMAX_DEFAULTS = (24, 24 * 120)  # (periodo estacional, ventana de entrenamiento)

RENEWABLE_FEATURES = ["shortwave_radiation", "shortwave_radiation_max", "direct_radiation",
                       "cloud_cover", "wind_speed_10m", "wind_speed_10m_max",
                       "wind_speed_100m", "wind_speed_100m_max", "precipitation",
                       "surface_pressure", "wave_height", "doy_sin", "doy_cos",
                       "is_weekend", "is_holiday"]
RENEWABLE_SARIMAX_DEFAULTS = (7, None)


def load_target_results() -> tuple[dict, dict]:
    """Devuelve el bloque de resultados COMPLETO (todos los modelos, no solo el
    campeón) de cada objetivo -- hace falta para, si el campeón es "Ensemble",
    saber qué modelos lo componen y con qué hiperparámetros/orden se validó cada
    uno."""
    with open(OUTPUT_DIR / "metrics.json", encoding="utf-8") as fh:
        m = json.load(fh)
    return m["demanda_mwh"], m["renewable_pct"]


def load_history() -> pd.DataFrame:
    df = pd.read_csv(CACHE_PATH, parse_dates=["datetime"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert("Europe/Madrid")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    df = df.set_index("datetime").asfreq("h")
    df[numeric_cols] = df[numeric_cols].interpolate(limit=6)
    df = df.reset_index()
    df = add_calendar_features(df)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df


def fetch_future_weather(horizon_days: int = 7) -> pd.DataFrame:
    weather = fetch_weather(forecast_days=horizon_days)
    marine = fetch_marine(forecast_days=horizon_days)
    future = weather.merge(marine, on="datetime", how="left")
    future = add_calendar_features(future)
    future["hour_sin"] = np.sin(2 * np.pi * future["hour"] / 24)
    future["hour_cos"] = np.cos(2 * np.pi * future["hour"] / 24)
    return future


def _forecast_one_model(name: str, combined: pd.DataFrame, target_col: str, horizon: int,
                         lag_steps: list, roll_windows: list, feature_cols: list, freq: pd.Timedelta,
                         train_series_full: pd.Series, train_feat: pd.DataFrame, fut_feat: pd.DataFrame,
                         target_results: dict, chronos_pipeline, sarimax_defaults: tuple) -> pd.Series:
    if name == "SARIMAX":
        series = combined.set_index("datetime")[target_col]
        exog = combined.set_index("datetime")[feature_cols]
        info = target_results.get("SARIMAX", {})
        order, seasonal_order = info.get("orden_pdq"), info.get("orden_estacional")
        seasonal_period, max_train = sarimax_defaults
        if order and seasonal_order:
            pred, _ = sarimax_auto_forecast(series, exog, horizon, seasonal_period, max_train,
                                             False, tuple(order), tuple(seasonal_order))
        else:
            pred, _ = sarimax_auto_forecast(series, exog, horizon, seasonal_period, max_train)
        return pred
    if name == "XGBoost":
        info = target_results.get("XGBoost", {})
        pred, *_ = xgb_recursive_forecast(combined, target_col, horizon, lag_steps, roll_windows,
                                           feature_cols, freq, fixed_params=info.get("mejores_hiperparametros"))
        return pred
    if name == "Chronos-2":
        pred = chronos_covariate_forecast(train_series_full, train_feat, fut_feat, horizon, chronos_pipeline)
        return pd.Series(pred, index=pd.DatetimeIndex(fut_feat["datetime"]))
    raise ValueError(f"Modelo desconocido: {name}")


def _forecast_target(target_col: str, combined: pd.DataFrame, horizon: int, lag_steps: list,
                      roll_windows: list, feature_cols: list, freq: pd.Timedelta,
                      train_series_full: pd.Series, train_feat: pd.DataFrame, fut_feat: pd.DataFrame,
                      target_results: dict, chronos_pipeline, sarimax_defaults: tuple,
                      baseline_season_len: int, progress=None, label: str = "") -> pd.Series:
    champion = target_results["_campeon"]
    sub_names = [champion]
    if champion == "Ensemble":
        sub_names = target_results["Ensemble"].get("compuesto_por", ["SARIMAX", "XGBoost", "Chronos-2"])

    preds = []
    for sub in sub_names:
        if progress:
            progress(f"Prediciendo {label} con {sub}" + (" (parte del ensemble)..." if champion == "Ensemble" else "..."))
        try:
            preds.append(_forecast_one_model(sub, combined, target_col, horizon, lag_steps, roll_windows,
                                              feature_cols, freq, train_series_full, train_feat, fut_feat,
                                              target_results, chronos_pipeline, sarimax_defaults))
        except Exception as exc:
            print(f"{sub} falló en predicción en vivo ({exc}), se omite.")
    if not preds:
        full = pd.concat([train_series_full, pd.Series(np.nan, index=pd.DatetimeIndex(fut_feat["datetime"]))])
        return baseline_seasonal(full, horizon, baseline_season_len)
    if len(preds) == 1:
        return preds[0]
    ref_index = preds[0].index
    return pd.Series(np.mean([np.asarray(p.reindex(ref_index)) for p in preds], axis=0), index=ref_index)


def forecast_demand(history: pd.DataFrame, future_weather: pd.DataFrame, target_results: dict,
                     chronos_pipeline, progress=None) -> pd.Series:
    horizon = len(future_weather)
    future_part = future_weather[["datetime"] + DEMAND_FEATURES].copy()
    future_part["demanda_mwh"] = np.nan
    combined = pd.concat(
        [history[["datetime", "demanda_mwh"] + DEMAND_FEATURES], future_part], ignore_index=True
    )
    return _forecast_target(
        "demanda_mwh", combined, horizon, DEMAND_LAGS, DEMAND_ROLLS, DEMAND_FEATURES, pd.Timedelta(hours=1),
        history.set_index("datetime")["demanda_mwh"], history[["datetime"] + DEMAND_FEATURES],
        future_weather[["datetime"] + DEMAND_FEATURES], target_results, chronos_pipeline,
        DEMAND_SARIMAX_DEFAULTS, 24 * 7, progress, "demanda",
    )


def forecast_renewable(history_daily: pd.DataFrame, future_daily: pd.DataFrame, target_results: dict,
                        chronos_pipeline, progress=None) -> pd.Series:
    horizon = len(future_daily)
    future_part = future_daily[["datetime"] + RENEWABLE_FEATURES].copy()
    future_part["renewable_pct"] = np.nan
    combined = pd.concat(
        [history_daily[["datetime", "renewable_pct"] + RENEWABLE_FEATURES], future_part], ignore_index=True
    )
    lag_steps = [1, 7, 14, 30, 365] if len(history_daily) > 400 else [1, 7, 14]
    roll_windows = [7, 30] if len(history_daily) > 400 else [7]
    return _forecast_target(
        "renewable_pct", combined, horizon, lag_steps, roll_windows, RENEWABLE_FEATURES, pd.Timedelta(days=1),
        history_daily.set_index("datetime")["renewable_pct"], history_daily[["datetime"] + RENEWABLE_FEATURES],
        future_daily[["datetime"] + RENEWABLE_FEATURES], target_results, chronos_pipeline,
        RENEWABLE_SARIMAX_DEFAULTS, 7, progress, "% renovable",
    )


def aggregate_daily(hourly_future: pd.DataFrame) -> pd.DataFrame:
    daily = (
        hourly_future.groupby(hourly_future["datetime"].dt.date)
        .agg(shortwave_radiation=("shortwave_radiation", "mean"),
             shortwave_radiation_max=("shortwave_radiation", "max"),
             direct_radiation=("direct_radiation", "mean"),
             cloud_cover=("cloud_cover", "mean"),
             wind_speed_10m=("wind_speed_10m", "mean"),
             wind_speed_10m_max=("wind_speed_10m", "max"),
             wind_speed_100m=("wind_speed_100m", "mean"),
             wind_speed_100m_max=("wind_speed_100m", "max"),
             precipitation=("precipitation", "sum"),
             surface_pressure=("surface_pressure", "mean"),
             wave_height=("wave_height", "mean"),
             doy_sin=("doy_sin", "first"),
             doy_cos=("doy_cos", "first"),
             is_weekend=("is_weekend", "max"),
             is_holiday=("is_holiday", "max"))
        .reset_index()
    )
    daily["datetime"] = pd.to_datetime(daily["datetime"])
    return daily


def aggregate_daily_history(history: pd.DataFrame) -> pd.DataFrame:
    daily = aggregate_daily(history)
    renewable = history.groupby(history["datetime"].dt.date)["renewable_pct"].first()
    daily["renewable_pct"] = daily["datetime"].dt.date.map(renewable)
    return daily


def build_forecast(progress=None) -> dict:
    """`progress`, si se pasa, es una función que recibe un string describiendo
    el paso en curso -- así la app de Streamlit puede mostrar al usuario en qué
    punto va la predicción "en tiempo real" en vez de un spinner ciego."""
    def _p(msg):
        if progress:
            progress(msg)
        print(msg)

    _p("Cargando histórico real (REE + Open-Meteo)...")
    demand_results, renewable_results = load_target_results()
    history = load_history()

    _p("Consultando previsión meteorológica real de los próximos 7 días (Open-Meteo)...")
    future_hourly = fetch_future_weather(horizon_days=7)

    def _uses_chronos(target_results: dict) -> bool:
        champ = target_results["_campeon"]
        if champ == "Chronos-2":
            return True
        if champ == "Ensemble":
            return "Chronos-2" in target_results.get("Ensemble", {}).get("compuesto_por", [])
        return False

    chronos_pipeline = None
    if _uses_chronos(demand_results) or _uses_chronos(renewable_results):
        _p("Cargando Chronos-2 (foundation model)...")
        from chronos import Chronos2Pipeline
        chronos_pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")

    demand_pred = forecast_demand(history, future_hourly, demand_results, chronos_pipeline, _p)

    history_daily = aggregate_daily_history(history)
    future_daily = aggregate_daily(future_hourly)
    renewable_pred = forecast_renewable(history_daily, future_daily, renewable_results, chronos_pipeline, _p)

    _p("Combinando demanda y % renovable en el resumen diario...")
    demand_daily_gwh = (demand_pred / 1000).groupby(demand_pred.index.date).sum()
    daily_summary = []
    for d in future_daily["datetime"].dt.date:
        gwh = float(demand_daily_gwh.get(d, float("nan")))
        pct = float(renewable_pred.get(pd.Timestamp(d), float("nan")))
        daily_summary.append({
            "date": str(d),
            "demanda_gwh": round(gwh, 1) if gwh == gwh else None,
            "renovable_%": round(pct, 1) if pct == pct else None,
            "renovable_gwh": round(gwh * pct / 100, 1) if gwh == gwh and pct == pct else None,
        })

    result = {
        "generado_en": datetime.now(timezone.utc).isoformat(),
        "historico_hasta": str(history["datetime"].max().date()),
        "modelo_demanda": demand_results["_campeon"],
        "modelo_renovable": renewable_results["_campeon"],
        "demanda_mwh": [{"datetime": str(t), "mwh": round(float(v), 1)} for t, v in demand_pred.items()],
        "renovable_pct": [{"date": str(t.date() if hasattr(t, "date") else t), "pct": round(float(v), 1)}
                           for t, v in renewable_pred.items()],
        "resumen_diario": daily_summary,
    }
    _p("Predicción lista.")
    return result


def main():
    result = build_forecast()
    with open(OUTPUT_DIR / "latest_forecast.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    print(f"Predicción generada con {result['modelo_demanda']} (demanda) y "
          f"{result['modelo_renovable']} (% renovable). Guardado en outputs/latest_forecast.json")
    for row in result["resumen_diario"]:
        print(row)


if __name__ == "__main__":
    main()
