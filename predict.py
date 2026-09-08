"""Genera la predicción a 7 días "en vivo": parte del histórico cacheado por
data.py (demanda + generación real de REE, clima histórico) y lo combina con la
previsión meteorológica real de los próximos 7 días (Open-Meteo /forecast, no el
archivo histórico) para predecir demanda y % renovable a futuro de verdad -- no
solo reproducir el holdout de evaluación de train.py.

Usa, para cada objetivo, el modelo que salió "campeón" (menor MAPE) en
outputs/metrics.json -- generado por train.py --, no siempre el mismo modelo
para todo: si SARIMAX gana en demanda y Chronos-2 gana en % renovable, aquí se
respeta esa elección en vez de forzar un único modelo para las dos cosas.

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

RENEWABLE_FEATURES = ["shortwave_radiation", "direct_radiation", "cloud_cover",
                       "wind_speed_10m", "wind_speed_100m", "precipitation",
                       "surface_pressure", "wave_height", "is_weekend", "is_holiday"]


def load_champions() -> tuple[dict, dict]:
    """Devuelve, para cada objetivo, el bloque de resultados del modelo campeón
    (incluye no solo el nombre sino los hiperparámetros/orden ya validados en
    train.py) -- así predict.py no vuelve a lanzar Optuna ni auto_arima en cada
    predicción, reutiliza lo que ya se validó offline."""
    with open(OUTPUT_DIR / "metrics.json", encoding="utf-8") as fh:
        m = json.load(fh)
    demand_info = m["demanda_mwh"][m["demanda_mwh"]["_campeon"]]
    demand_info["_nombre"] = m["demanda_mwh"]["_campeon"]
    renewable_info = m["renewable_pct"][m["renewable_pct"]["_campeon"]]
    renewable_info["_nombre"] = m["renewable_pct"]["_campeon"]
    return demand_info, renewable_info


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


def forecast_demand(history: pd.DataFrame, future_weather: pd.DataFrame, champion: dict,
                     chronos_pipeline) -> pd.Series:
    horizon = len(future_weather)
    future_part = future_weather[["datetime"] + DEMAND_FEATURES].copy()
    future_part["demanda_mwh"] = np.nan
    combined = pd.concat(
        [history[["datetime", "demanda_mwh"] + DEMAND_FEATURES], future_part], ignore_index=True
    )
    name = champion["_nombre"]

    if name == "SARIMAX":
        series = combined.set_index("datetime")["demanda_mwh"]
        exog = combined.set_index("datetime")[DEMAND_FEATURES]
        order = tuple(champion.get("orden_pdq", (2, 0, 2)))
        seasonal_order = tuple(champion.get("orden_estacional", (1, 0, 1, 24)))
        pred, _ = sarimax_auto_forecast(series, exog, horizon, 24, 24 * 120, False,
                                         order, seasonal_order)
        return pred
    if name == "XGBoost":
        pred, *_ = xgb_recursive_forecast(combined, "demanda_mwh", horizon, DEMAND_LAGS,
                                           DEMAND_ROLLS, DEMAND_FEATURES, pd.Timedelta(hours=1),
                                           fixed_params=champion.get("mejores_hiperparametros"))
        return pred
    if name == "Chronos-2":
        train_feat = history[["datetime"] + DEMAND_FEATURES]
        fut_feat = future_weather[["datetime"] + DEMAND_FEATURES]
        pred = chronos_covariate_forecast(
            history.set_index("datetime")["demanda_mwh"], train_feat, fut_feat, horizon, chronos_pipeline
        )
        return pd.Series(pred, index=pd.DatetimeIndex(future_weather["datetime"]))
    # Baseline (fallback si ninguno de los otros aplica)
    full = pd.concat([history.set_index("datetime")["demanda_mwh"],
                       pd.Series(np.nan, index=pd.DatetimeIndex(future_weather["datetime"]))])
    return baseline_seasonal(full, horizon, 24 * 7)


def forecast_renewable(history_daily: pd.DataFrame, future_daily: pd.DataFrame, champion: dict,
                        chronos_pipeline) -> pd.Series:
    horizon = len(future_daily)
    future_part = future_daily[["datetime"] + RENEWABLE_FEATURES].copy()
    future_part["renewable_pct"] = np.nan
    combined = pd.concat(
        [history_daily[["datetime", "renewable_pct"] + RENEWABLE_FEATURES], future_part],
        ignore_index=True,
    )
    lag_steps = [1, 7, 14, 30, 365] if len(history_daily) > 400 else [1, 7, 14]
    roll_windows = [7, 30] if len(history_daily) > 400 else [7]
    name = champion["_nombre"]

    if name == "SARIMAX":
        series = combined.set_index("datetime")["renewable_pct"]
        exog = combined.set_index("datetime")[RENEWABLE_FEATURES]
        order = champion.get("orden_pdq")
        seasonal_order = champion.get("orden_estacional")
        if order and seasonal_order:
            pred, _ = sarimax_auto_forecast(series, exog, horizon, 7, use_auto_search=False,
                                             fixed_order=tuple(order), fixed_seasonal_order=tuple(seasonal_order))
        else:
            pred, _ = sarimax_auto_forecast(series, exog, horizon, 7)
        return pred
    if name == "XGBoost":
        pred, *_ = xgb_recursive_forecast(combined, "renewable_pct", horizon, lag_steps,
                                           roll_windows, RENEWABLE_FEATURES, pd.Timedelta(days=1),
                                           fixed_params=champion.get("mejores_hiperparametros"))
        return pred
    if name == "Chronos-2":
        train_feat = history_daily[["datetime"] + RENEWABLE_FEATURES]
        fut_feat = future_daily[["datetime"] + RENEWABLE_FEATURES]
        pred = chronos_covariate_forecast(
            history_daily.set_index("datetime")["renewable_pct"], train_feat, fut_feat, horizon, chronos_pipeline
        )
        return pd.Series(pred, index=pd.DatetimeIndex(future_daily["datetime"]))
    full = pd.concat([history_daily.set_index("datetime")["renewable_pct"],
                       pd.Series(np.nan, index=pd.DatetimeIndex(future_daily["datetime"]))])
    return baseline_seasonal(full, horizon, 7)


def aggregate_daily(hourly_future: pd.DataFrame) -> pd.DataFrame:
    daily = (
        hourly_future.groupby(hourly_future["datetime"].dt.date)
        .agg(shortwave_radiation=("shortwave_radiation", "mean"),
             direct_radiation=("direct_radiation", "mean"),
             cloud_cover=("cloud_cover", "mean"),
             wind_speed_10m=("wind_speed_10m", "mean"),
             wind_speed_100m=("wind_speed_100m", "mean"),
             precipitation=("precipitation", "sum"),
             surface_pressure=("surface_pressure", "mean"),
             wave_height=("wave_height", "mean"),
             is_weekend=("is_weekend", "max"),
             is_holiday=("is_holiday", "max"))
        .reset_index()
    )
    daily["datetime"] = pd.to_datetime(daily["datetime"])
    return daily


def aggregate_daily_history(history: pd.DataFrame) -> pd.DataFrame:
    daily = (
        history.groupby(history["datetime"].dt.date)
        .agg(renewable_pct=("renewable_pct", "first"),
             shortwave_radiation=("shortwave_radiation", "mean"),
             direct_radiation=("direct_radiation", "mean"),
             cloud_cover=("cloud_cover", "mean"),
             wind_speed_10m=("wind_speed_10m", "mean"),
             wind_speed_100m=("wind_speed_100m", "mean"),
             precipitation=("precipitation", "sum"),
             surface_pressure=("surface_pressure", "mean"),
             wave_height=("wave_height", "mean"),
             is_weekend=("is_weekend", "max"),
             is_holiday=("is_holiday", "max"))
        .reset_index()
    )
    daily["datetime"] = pd.to_datetime(daily["datetime"])
    return daily


def build_forecast() -> dict:
    demand_champion, renewable_champion = load_champions()
    history = load_history()
    future_hourly = fetch_future_weather(horizon_days=7)

    chronos_pipeline = None
    if "Chronos-2" in (demand_champion["_nombre"], renewable_champion["_nombre"]):
        from chronos import Chronos2Pipeline
        chronos_pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")

    demand_pred = forecast_demand(history, future_hourly, demand_champion, chronos_pipeline)

    history_daily = aggregate_daily_history(history)
    future_daily = aggregate_daily(future_hourly)
    renewable_pred = forecast_renewable(history_daily, future_daily, renewable_champion, chronos_pipeline)

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
        "modelo_demanda": demand_champion["_nombre"],
        "modelo_renovable": renewable_champion["_nombre"],
        "demanda_mwh": [{"datetime": str(t), "mwh": round(float(v), 1)} for t, v in demand_pred.items()],
        "renovable_pct": [{"date": str(t.date() if hasattr(t, "date") else t), "pct": round(float(v), 1)}
                           for t, v in renewable_pred.items()],
        "resumen_diario": daily_summary,
    }
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
