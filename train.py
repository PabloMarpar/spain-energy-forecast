"""Entrena y compara 4 enfoques -- baseline estacional, SARIMAX (statsmodels vía
pmdarima), XGBoost (hiperparámetros ajustados con Optuna, recursivo) y Chronos-2
(foundation model, zero-shot, con clima/calendario como covariables) -- para
predecir a 7 días:
  1) la demanda eléctrica de España (horaria)
  2) el % de generación renovable (diario)

El holdout de evaluación son siempre los últimos 7 días, nunca un split aleatorio
(mezclar fechas al azar en series temporales es un error clásico: se estaría
evaluando con "el futuro" disponible en el entrenamiento).

Para cada objetivo se guarda, además de las métricas de los 4 modelos, cuál es
el "campeón" (menor MAPE) -- ese es el modelo que usan luego predict.py y la
app de Streamlit para ese objetivo en concreto. No tiene sentido forzar un único
modelo "ganador" para todo: si SARIMAX es mejor prediciendo demanda y Chronos-2
es mejor prediciendo % renovable, se usa cada uno donde de verdad gana.
"""

import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import matplotlib.pyplot as plt
import pmdarima as pm
import xgboost as xgb
from chronos import Chronos2Pipeline

from data import CACHE_PATH, add_calendar_features

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
optuna.logging.set_verbosity(optuna.logging.WARNING)

COLORS = {"Real": "#0b0b0b", "Baseline": "#898781", "SARIMAX": "#eb6834",
          "XGBoost": "#2a78d6", "Chronos-2": "#1baf7a", "Ensemble": "#8e44ad"}


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    """Más allá de MAE/RMSE/MAPE (las 3 "de siempre") se añaden:
    - sMAPE: como MAPE pero simétrico, no explota si `actual` se acerca a 0
      (le pasa a veces a la demanda de madrugada o a días de % renovable bajo).
    - R2: cuánta varianza del real explica la predicción (1 = perfecto, 0 = como
      predecir siempre la media, negativo = peor que la media).
    - Bias: error medio con signo. Positivo = el modelo infrapredice de media
      (el real suele quedar por encima), negativo = sobrepredice.
    """
    actual, pred = np.asarray(actual, dtype=float), np.asarray(pred, dtype=float)
    err = actual - pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mape = float(np.mean(np.abs(err / actual))) * 100
    smape = float(np.mean(2 * np.abs(err) / (np.abs(actual) + np.abs(pred) + 1e-9))) * 100
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    bias = float(np.mean(err))
    return {
        "MAE": round(mae, 3), "RMSE": round(rmse, 3), "MAPE_%": round(mape, 2),
        "sMAPE_%": round(smape, 2), "R2": round(r2, 3), "Bias": round(bias, 3),
    }


def horizon_breakdown(actual: np.ndarray, pred: np.ndarray, near_far_split: int) -> dict:
    """MAPE del tramo cercano del horizonte (día 1) frente al lejano (día 7).
    Casi todo forecaster pierde precisión cuanto más lejos predice -- cuantificar
    cuánto se degrada es más honesto que dar un único número para "a 7 días"."""
    actual, pred = np.asarray(actual, dtype=float), np.asarray(pred, dtype=float)
    near_a, near_p = actual[:near_far_split], pred[:near_far_split]
    far_a, far_p = actual[-near_far_split:], pred[-near_far_split:]
    mape_near = float(np.mean(np.abs((near_a - near_p) / near_a))) * 100
    mape_far = float(np.mean(np.abs((far_a - far_p) / far_a))) * 100
    return {"MAPE_dia1_%": round(mape_near, 2), "MAPE_dia7_%": round(mape_far, 2)}


def baseline_seasonal(series: pd.Series, horizon: int, season_len: int) -> pd.Series:
    train = series.iloc[:-horizon]
    tail = train.iloc[-season_len:]
    reps = int(np.ceil(horizon / season_len))
    preds = pd.concat([tail] * reps).iloc[:horizon]
    preds.index = series.index[-horizon:]
    return preds


# ---------------------------------------------------------------------------
# SARIMAX con orden automático (pmdarima)
# ---------------------------------------------------------------------------

def sarimax_auto_forecast(series: pd.Series, exog: pd.DataFrame, horizon: int,
                           seasonal_period: int, max_train_points: int = None,
                           use_auto_search: bool = True, fixed_order=None,
                           fixed_seasonal_order=None):
    """Por defecto usa pmdarima.auto_arima para buscar el mejor orden
    (p,d,q)(P,D,Q,s) probando varias combinaciones, en vez de que el orden salga
    de una suposición manual. Entrena sobre una ventana reciente, no todo el
    histórico -- con periodo estacional, ajustar sobre años completos de datos
    horarios es intratable en CPU; una ventana reciente sigue capturando la
    estacionalidad y es práctica habitual en producción (reajuste sobre ventana
    móvil en vez de reentrenar con todo el histórico).

    Para la serie horaria de demanda (m=24) la búsqueda automática de
    auto_arima resultó intratable en CPU (cada candidato con estacionalidad de
    24 horas tarda minutos, y evalúa varios) -- ahí se usa `use_auto_search=False`
    con un orden fijo (elegido a mano previamente y confirmado que funciona bien)
    ajustado directamente con `pmdarima.ARIMA`, sin búsqueda. Sigue siendo la
    misma librería, solo sin el coste del stepwise search para esta serie en
    concreto -- se documenta como decisión de ingeniería, no se esconde."""
    train_y = series.iloc[:-horizon]
    train_x = exog.iloc[:-horizon]
    if max_train_points:
        train_y = train_y.iloc[-max_train_points:]
        train_x = train_x.iloc[-max_train_points:]
    future_x = exog.iloc[-horizon:]

    if use_auto_search:
        model = pm.auto_arima(
            train_y.values, X=train_x.values, seasonal=True, m=seasonal_period,
            stepwise=True, suppress_warnings=True, error_action="ignore",
            max_p=2, max_q=2, max_P=1, max_Q=1, maxiter=100,
        )
    else:
        model = pm.ARIMA(order=fixed_order, seasonal_order=fixed_seasonal_order,
                          suppress_warnings=True)
        model.fit(train_y.values, X=train_x.values)

    forecast = model.predict(n_periods=horizon, X=future_x.values)
    order_str = f"{model.order}x{model.seasonal_order}"
    return pd.Series(np.asarray(forecast), index=series.index[-horizon:]), order_str


# ---------------------------------------------------------------------------
# XGBoost DIRECTO multi-horizonte (con "h" como feature), Optuna
# ---------------------------------------------------------------------------
#
# Antes esto era un forecast RECURSIVO: un modelo a 1 paso, cuya propia
# predicción se realimentaba como "lag" del siguiente paso. Con un horizonte de
# 168 (demanda) o 7 (renovables) eso encadena mucho error y, peor, tiende a
# "aplanarse": en cuanto el modelo predice un valor cercano a la media, ese
# valor suavizado pasa a ser el lag del siguiente paso, y el siguiente, y el
# siguiente -- se vio clarísimo en la comparativa de renovables (la predicción
# apenas se movía semana a semana mientras el real oscilaba mucho más), y la
# importancia de variables lo confirmaba: `lag_1` acaparaba ~60% de la
# importancia, todo el clima junto no llegaba al 15%.
#
# Ahora es DIRECTO: un único modelo aprende a predecir cualquier paso h=1..H
# por delante, con `h` como variable más, a partir de lags/medias móviles
# calculados SIEMPRE con datos reales del origen (nunca con predicciones
# encadenadas) más el clima/calendario del instante futuro. El dataset de
# entrenamiento se construye tomando un origen de cada `origin_stride` pasos
# (no todas las horas) para que cada fila se parezca a como se usa el modelo
# de verdad -- una predicción de horizonte completo por vez, no una por hora.

def _lag_roll_features(series: pd.Series, lag_steps: list, roll_windows: list) -> pd.DataFrame:
    feats = {f"lag_{l}": series.shift(l) for l in lag_steps}
    for w in roll_windows:
        # shift(1) antes del rolling: la media solo puede usar valores anteriores
        # al instante que se predice, nunca el propio valor (fuga de información).
        feats[f"roll_{w}"] = series.shift(1).rolling(w).mean()
    return pd.DataFrame(feats)


def build_direct_frame(df: pd.DataFrame, target_col: str, feature_cols: list, lag_steps: list,
                        roll_windows: list, horizon: int, origin_stride: int):
    lag_roll = _lag_roll_features(df[target_col], lag_steps, roll_windows)
    lag_roll_names = list(lag_roll.columns)
    valid = lag_roll.notna().all(axis=1)

    n = len(df)
    origins = [o for o in range(n - horizon) if valid.iloc[o]][::origin_stride]

    rows = []
    for o in origins:
        base = lag_roll.iloc[o].to_dict()
        for h in range(1, horizon + 1):
            row = dict(base)
            row["h"] = h
            t = o + h
            for col in feature_cols:
                row[col] = df[col].iloc[t]
            row["_target"] = df[target_col].iloc[t]
            rows.append(row)
    frame = pd.DataFrame(rows)
    feature_names = lag_roll_names + ["h"] + feature_cols
    return frame, feature_names


def tune_xgb_params_direct(frame: pd.DataFrame, feature_names: list, n_trials: int = 40) -> dict:
    """Como un tuning normal de Optuna, pero la validación se separa por ORIGEN
    (no por fila): todas las filas de un mismo origen (una por cada h) comparten
    los mismos lags, así que si se mezclaran entre train y validación la
    validación dejaría de ser honesta -- se valida con los últimos orígenes."""
    n_h = frame["h"].nunique()
    n_origins = len(frame) // n_h
    n_val_origins = max(int(n_origins * 0.15), 3)
    n_val_rows = n_val_origins * n_h

    frame_tr, frame_val = frame.iloc[:-n_val_rows], frame.iloc[-n_val_rows:]
    X_tr, y_tr = frame_tr[feature_names], frame_tr["_target"]
    X_val, y_val = frame_val[feature_names], frame_val["_target"]

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 5.0, log=True),
        }
        model = xgb.XGBRegressor(**params, random_state=42)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_val)
        return float(np.sqrt(np.mean((y_val.values - pred) ** 2)))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def xgb_direct_forecast(df: pd.DataFrame, target_col: str, horizon: int, lag_steps: list,
                         roll_windows: list, feature_cols: list, origin_stride: int,
                         n_trials: int = 40, fixed_params: dict = None):
    """Si se pasa `fixed_params` (los hiperparámetros ya validados y guardados en
    outputs/metrics.json por un entrenamiento previo) se usan directamente y no
    se vuelve a lanzar Optuna -- así es como predict.py sirve predicciones en
    segundos en vez de repetir una búsqueda de 40 pruebas en cada petición."""
    train_df = df.iloc[:-horizon].reset_index(drop=True)
    frame, feature_names = build_direct_frame(train_df, target_col, feature_cols, lag_steps,
                                               roll_windows, horizon, origin_stride)

    best_params = fixed_params or tune_xgb_params_direct(frame, feature_names, n_trials)
    model = xgb.XGBRegressor(**best_params, random_state=42)
    model.fit(frame[feature_names], frame["_target"])

    # Predicción real: un único origen (el último punto de train_df, con datos
    # 100% reales) y h=1..horizon -- nunca se realimenta una predicción propia.
    base = _lag_roll_features(train_df[target_col], lag_steps, roll_windows).iloc[-1].to_dict()
    future_meta = df.iloc[-horizon:].reset_index(drop=True)
    rows = []
    for h in range(1, horizon + 1):
        row = dict(base)
        row["h"] = h
        for col in feature_cols:
            row[col] = future_meta[col].iloc[h - 1]
        rows.append(row)
    preds = model.predict(pd.DataFrame(rows)[feature_names])
    index = df.iloc[-horizon:].set_index("datetime").index
    return pd.Series(preds, index=index), model, feature_names, best_params


# ---------------------------------------------------------------------------
# Chronos-2 con covariables (clima + calendario)
# ---------------------------------------------------------------------------

def _to_uniform_utc(datetime_col: pd.Series) -> pd.Series:
    """Chronos-2 necesita inferir una frecuencia uniforme sobre la columna de
    fecha. En hora local de Madrid el cambio de hora (DST) rompe esa uniformidad
    (un salto de 2 horas en primavera, uno repetido en otoño) aunque no falte
    ningún dato -- se pasa a UTC, que no tiene cambios de hora, antes de
    quitarle el timezone. La serie diaria de renovables ya llega sin timezone
    (a nivel de día el DST no afecta), así que ahí no hace falta convertir."""
    dt = pd.to_datetime(datetime_col)
    if dt.dt.tz is not None:
        dt = dt.dt.tz_convert("UTC").dt.tz_localize(None)
    return dt


def chronos_covariate_forecast(train_series: pd.Series, train_features: pd.DataFrame,
                                future_features: pd.DataFrame, horizon: int, pipeline) -> np.ndarray:
    context_df = train_features.copy()
    context_df["target"] = train_series.values
    context_df["datetime"] = _to_uniform_utc(context_df["datetime"])
    context_df["id"] = "es"

    future_df = future_features.copy()
    future_df["datetime"] = _to_uniform_utc(future_df["datetime"])
    future_df["id"] = "es"

    pred = pipeline.predict_df(
        context_df, future_df=future_df, prediction_length=horizon,
        quantile_levels=[0.1, 0.5, 0.9], id_column="id", timestamp_column="datetime", target="target",
    )
    return pred["0.5"].to_numpy()


# ---------------------------------------------------------------------------
# Gráficas
# ---------------------------------------------------------------------------

def plot_comparison(actual: pd.Series, forecasts: dict, title: str, ylabel: str, path: Path):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(actual.index, actual.values, color=COLORS["Real"], linewidth=2, label="Real")
    for name, series in forecasts.items():
        ax.plot(actual.index, np.asarray(series)[: len(actual)], color=COLORS[name], linewidth=1.6,
                 linestyle="--", label=name)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


GENERATION_BUCKETS = {
    "Solar": ["tech_solar_fv_pct", "tech_solar_termica_pct"],
    "Eólica": ["tech_eolica_pct"],
    "Hidráulica": ["tech_hidraulica_pct"],
    "Otras renovables": ["tech_otras_renovables_pct", "tech_residuos_renovables_pct"],
    "Nuclear": ["tech_nuclear_pct"],
    "No renovable (resto)": ["tech_carbon_pct", "tech_fuel_gas_pct", "tech_turbina_vapor_pct",
                              "tech_ciclo_combinado_pct", "tech_cogeneracion_pct",
                              "tech_residuos_no_renovables_pct"],
}
GENERATION_COLORS = {"Solar": "#f4b400", "Eólica": "#4285f4", "Hidráulica": "#1a73e8",
                      "Otras renovables": "#34a853", "Nuclear": "#9c27b0", "No renovable (resto)": "#5f6368"}


def plot_generation_mix(df: pd.DataFrame, path: Path, days: int = 730):
    """Stacked-area del mix de generación por tecnología -- el estilo de gráfica
    estándar del sector para "cuánto pone cada fuente" (el mismo que usan REE,
    ENTSO-E o cualquier operador del sistema), no solo el % renovable agregado."""
    daily = df.groupby(df["datetime"].dt.date).first(numeric_only=True).tail(days)
    x = pd.to_datetime(daily.index)
    values, labels, colors = [], [], []
    for name, cols in GENERATION_BUCKETS.items():
        present = [c for c in cols if c in daily.columns]
        if not present:
            continue
        values.append(daily[present].sum(axis=1).values)
        labels.append(name)
        colors.append(GENERATION_COLORS[name])
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.stackplot(x, values, labels=labels, colors=colors, alpha=0.92)
    ax.set_title(f"Mix de generación eléctrica en España (% diario, últimos {days // 365} años)")
    ax.set_ylabel("%")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.14))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(model, feature_names, path: Path, title: str):
    importance = model.feature_importances_
    order = np.argsort(importance)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(np.array(feature_names)[order][::-1], importance[order][::-1], color="#2a78d6")
    ax.set_title(title)
    ax.set_xlabel("Importancia")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Comparativa para un target concreto
# ---------------------------------------------------------------------------

def run_comparison(df: pd.DataFrame, target_col: str, horizon: int, baseline_season_len: int,
                    lag_steps: list, roll_windows: list, feature_cols: list,
                    chronos_pipeline, label: str, ylabel: str, near_far_split: int,
                    origin_stride: int, sarimax_max_train: int = None, sarimax_period: int = 7,
                    sarimax_auto_search: bool = True, sarimax_fixed_order=None,
                    sarimax_fixed_seasonal_order=None, xgb_trials: int = 40) -> dict:
    data_ = df.dropna(subset=[target_col] + feature_cols).reset_index(drop=True)
    series = data_.set_index("datetime")[target_col]
    actual_holdout = series.iloc[-horizon:]

    print(f"\n=== {label}: {len(series)} puntos, horizonte {horizon} ===")

    results, forecasts = {}, {}

    forecasts["Baseline"] = baseline_seasonal(series, horizon, baseline_season_len).values
    results["Baseline"] = metrics(actual_holdout, forecasts["Baseline"])
    results["Baseline"].update(horizon_breakdown(actual_holdout.values, forecasts["Baseline"], near_far_split))
    print("Baseline:", results["Baseline"])
    baseline_mape = results["Baseline"]["MAPE_%"]

    exog = data_.set_index("datetime")[feature_cols]
    try:
        sarimax_pred, order_str = sarimax_auto_forecast(
            series, exog, horizon, sarimax_period, sarimax_max_train,
            sarimax_auto_search, sarimax_fixed_order, sarimax_fixed_seasonal_order,
        )
        forecasts["SARIMAX"] = sarimax_pred.values
        results["SARIMAX"] = metrics(actual_holdout, sarimax_pred)
        results["SARIMAX"].update(horizon_breakdown(actual_holdout.values, sarimax_pred.values, near_far_split))
        results["SARIMAX"]["orden"] = order_str
        order_tuple, seasonal_tuple = order_str.split("x")
        results["SARIMAX"]["orden_pdq"] = [int(v) for v in order_tuple.strip("()").split(",")]
        results["SARIMAX"]["orden_estacional"] = [int(v) for v in seasonal_tuple.strip("()").split(",")]
        results["SARIMAX"]["mejora_vs_baseline_%"] = round(
            (baseline_mape - results["SARIMAX"]["MAPE_%"]) / baseline_mape * 100, 1)
        print(f"SARIMAX (orden {order_str}):", results["SARIMAX"])
    except Exception as exc:
        print(f"SARIMAX falló ({exc}), se omite de la comparativa.")

    xgb_pred, xgb_model, xgb_features, xgb_params = xgb_direct_forecast(
        data_, target_col, horizon, lag_steps, roll_windows, feature_cols, origin_stride, xgb_trials
    )
    forecasts["XGBoost"] = xgb_pred.values
    results["XGBoost"] = metrics(actual_holdout, xgb_pred)
    results["XGBoost"].update(horizon_breakdown(actual_holdout.values, xgb_pred.values, near_far_split))
    results["XGBoost"]["mejores_hiperparametros"] = xgb_params
    results["XGBoost"]["mejora_vs_baseline_%"] = round(
        (baseline_mape - results["XGBoost"]["MAPE_%"]) / baseline_mape * 100, 1)
    print("XGBoost (Optuna):", results["XGBoost"])

    try:
        train_features = data_.iloc[:-horizon][["datetime"] + feature_cols]
        future_features = data_.iloc[-horizon:][["datetime"] + feature_cols]
        chronos_pred = chronos_covariate_forecast(
            series.iloc[:-horizon], train_features, future_features, horizon, chronos_pipeline
        )
        forecasts["Chronos-2"] = np.asarray(chronos_pred)
        results["Chronos-2"] = metrics(actual_holdout, np.asarray(chronos_pred))
        results["Chronos-2"].update(horizon_breakdown(actual_holdout.values, np.asarray(chronos_pred), near_far_split))
        results["Chronos-2"]["mejora_vs_baseline_%"] = round(
            (baseline_mape - results["Chronos-2"]["MAPE_%"]) / baseline_mape * 100, 1)
        print("Chronos-2 (zero-shot + covariables):", results["Chronos-2"])
    except Exception as exc:
        print(f"Chronos-2 falló ({exc}), se omite de la comparativa.")

    # Ensemble = media simple de los modelos "de verdad" (no el baseline). Promediar
    # modelos con errores no perfectamente correlacionados suele reducir el error
    # total -- es una técnica estándar, no un intento de inflar el resultado: se
    # reporta con sus métricas como uno más, y solo "gana" si de verdad mejora.
    non_baseline = {k: v for k, v in forecasts.items() if k != "Baseline"}
    if len(non_baseline) >= 2:
        ensemble_pred = np.mean([np.asarray(v)[: len(actual_holdout)] for v in non_baseline.values()], axis=0)
        forecasts["Ensemble"] = ensemble_pred
        results["Ensemble"] = metrics(actual_holdout, ensemble_pred)
        results["Ensemble"].update(horizon_breakdown(actual_holdout.values, ensemble_pred, near_far_split))
        results["Ensemble"]["compuesto_por"] = list(non_baseline.keys())
        results["Ensemble"]["mejora_vs_baseline_%"] = round(
            (baseline_mape - results["Ensemble"]["MAPE_%"]) / baseline_mape * 100, 1)
        print("Ensemble (media):", results["Ensemble"])

    champion = min(results.keys(), key=lambda name: results[name]["MAPE_%"])
    results["_campeon"] = champion
    print(f"Campeón para {label}: {champion} (MAPE {results[champion]['MAPE_%']}%)")

    plot_comparison(actual_holdout, forecasts, f"{label}: real vs. predicción (últimos 7 días)",
                     ylabel, OUTPUT_DIR / f"{target_col}_comparativa.png")
    plot_feature_importance(xgb_model, xgb_features, OUTPUT_DIR / f"{target_col}_importancia.png",
                             f"Importancia de variables -- XGBoost ({label})")

    # Predicción punto a punto del holdout (no solo las métricas agregadas) --
    # es lo que usa la app de Streamlit para el gráfico interactivo real vs.
    # predicción; con esto se puede hacer zoom, ocultar modelos, etc. en vez de
    # depender de una imagen estática.
    holdout_df = pd.DataFrame({"datetime": actual_holdout.index, "real": actual_holdout.values})
    for name, values in forecasts.items():
        holdout_df[name] = np.asarray(values)[: len(actual_holdout)]
    holdout_df.to_csv(OUTPUT_DIR / f"{target_col}_predicciones_holdout.csv", index=False)

    return results


def main():
    df = pd.read_csv(CACHE_PATH, parse_dates=["datetime"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert("Europe/Madrid")

    # Rejilla horaria completa: huecos puntuales (horas sin publicar todavía en la
    # API de REE) rompían el espaciado uniforme si simplemente se descartaban esas
    # filas -- eso desalinea los lags (24h/168h) y hace que Chronos-2 ni siquiera
    # pueda inferir la frecuencia de la serie. Se rellena la rejilla y se
    # interpolan los huecos (como mucho 6 horas seguidas), y las columnas de
    # calendario se recalculan desde cero para las horas que faltaban.
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    df = df.set_index("datetime").asfreq("h")
    df[numeric_cols] = df[numeric_cols].interpolate(limit=6)
    df = df.reset_index()
    df = add_calendar_features(df)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    plot_generation_mix(df, OUTPUT_DIR / "generation_mix.png")

    print("Cargando Chronos-2 (foundation model, se descarga la primera vez)...")
    chronos_pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")

    all_results = {}

    demand_features = ["temperature_2m", "relative_humidity_2m", "shortwave_radiation",
                        "wind_speed_10m", "wind_speed_100m", "is_weekend", "is_holiday"]
    all_results["demanda_mwh"] = run_comparison(
        df, "demanda_mwh", horizon=24 * 7, baseline_season_len=24 * 7,
        lag_steps=[24, 48, 168, 336], roll_windows=[24, 168],
        feature_cols=demand_features,
        chronos_pipeline=chronos_pipeline, label="Demanda eléctrica (MWh/h)", ylabel="MWh",
        near_far_split=24, origin_stride=24, sarimax_max_train=24 * 120, sarimax_auto_search=False,
        sarimax_fixed_order=(2, 0, 2), sarimax_fixed_seasonal_order=(1, 0, 1, 24),
    )

    daily = (
        df.groupby(df["datetime"].dt.date)
        .agg(renewable_pct=("renewable_pct", "first"),
             shortwave_radiation=("shortwave_radiation", "mean"),
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
    # doy_sin/doy_cos (ciclo anual) son, con diferencia, las variables que más
    # deberían pesar aquí: el % solar depende sobre todo de en qué época del año
    # estamos, mucho más estable año a año que la meteorología día a día -- con
    # 5 años de histórico ya hay suficientes ciclos anuales completos para que el
    # modelo pueda aprovecharlas.
    renewable_features = ["shortwave_radiation", "shortwave_radiation_max", "direct_radiation",
                           "cloud_cover", "wind_speed_10m", "wind_speed_10m_max",
                           "wind_speed_100m", "wind_speed_100m_max", "precipitation",
                           "surface_pressure", "wave_height", "doy_sin", "doy_cos",
                           "is_weekend", "is_holiday"]
    roll_windows_renewable = [7, 30] if len(daily) > 400 else [7]
    lag_steps_renewable = [1, 7, 14, 30, 365] if len(daily) > 400 else [1, 7, 14]
    all_results["renewable_pct"] = run_comparison(
        daily, "renewable_pct", horizon=7, baseline_season_len=7,
        lag_steps=lag_steps_renewable, roll_windows=roll_windows_renewable,
        feature_cols=renewable_features,
        chronos_pipeline=chronos_pipeline, label="% Generación renovable", ylabel="%",
        near_far_split=1, origin_stride=1, sarimax_period=7,
    )

    with open(OUTPUT_DIR / "metrics.json", "w") as fh:
        json.dump(all_results, fh, indent=2, ensure_ascii=False)
    print("\nGuardado outputs/metrics.json y gráficas comparativas.")
    print(f"Campeones -> demanda: {all_results['demanda_mwh']['_campeon']}, "
          f"% renovable: {all_results['renewable_pct']['_campeon']}")


if __name__ == "__main__":
    main()
