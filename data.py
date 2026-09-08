"""Descarga demanda eléctrica, mix de generación (% renovable) y clima de España
(REE + Open-Meteo) y construye un dataset horario unificado para predecir demanda
y % de generación renovable a 7 días.

Fuentes (todas gratuitas, sin API key):
- REE apidatos: demanda y estructura de generación por tecnología.
- Open-Meteo: radiación solar, viento, precipitación, temperatura.
- Open-Meteo Marine: oleaje, cerca de Mutriku (País Vasco), único punto con
  generación undimotriz real en España -- se prueba como variable exploratoria.
"""

import time
from datetime import date, timedelta
from pathlib import Path

import holidays
import pandas as pd
import requests

CACHE_PATH = Path("energy_weather_hourly.csv")

REE_DEMANDA_URL = "https://apidatos.ree.es/es/datos/demanda/evolucion"
REE_GENERACION_URL = "https://apidatos.ree.es/es/datos/generacion/estructura-generacion"
METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"

# Punto representativo de España peninsular para clima (Madrid, centroide aproximado)
LAT, LON = 40.4168, -3.7038
# Punto costero cerca de Mutriku (País Vasco), la única planta undimotriz real de
# España, para la variable exploratoria de oleaje ("fuerza del mar")
MARINE_LAT, MARINE_LON = 43.30, -2.38

RENEWABLE_KEYWORDS = ["solar", "eólic", "eolic", "hidrául", "hidraul", "hidroeólica", "renovable", "geotérmica"]

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; portfolio-research-script/1.0)"}


def _get_ree(url: str, params: dict, retries: int = 2, base_delay: float = 4.0) -> dict | None:
    """GET a un endpoint de REE. apidatos.ree.es devuelve a veces un 400 genérico
    ("datos no disponibles en este momento, inténtelo más tarde") que no es un error
    real de los parámetros, sino un fallo transitorio del backend -- se reintenta un
    par de veces y, si sigue fallando, se devuelve None para reintentarlo en una
    segunda pasada al final (en vez de tirar toda la descarga de 2 años por un mes)."""
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=30)
            r.raise_for_status()
            r.encoding = "utf-8"
            return r.json()
        except requests.exceptions.HTTPError as exc:
            wait = base_delay * (2 ** attempt)
            print(f"  aviso: {exc} -- reintentando en {wait:.0f}s ({attempt + 1}/{retries})")
            time.sleep(wait)
    return None


def _date_chunks(start: date, end: date, days: int = 14):
    """Trocea en bloques de `days` días -- se comprobó en vivo que el endpoint horario
    de REE falla con bloques de 1 mes completo pero funciona bien con 2 semanas."""
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=days), end)
        yield cur, nxt
        cur = nxt


def fetch_ree_demanda(start: date, end: date) -> pd.DataFrame:
    rows = []
    pending = list(_date_chunks(start, end))
    for _pass in range(2):  # 2ª pasada: reintenta solo los meses que fallaron en la 1ª
        still_pending = []
        for chunk_start, chunk_end in pending:
            params = {
                "start_date": f"{chunk_start}T00:00",
                "end_date": f"{chunk_end}T00:00",
                "time_trunc": "hour",
                "geo_trunc": "electric_system",
                "geo_limit": "peninsular",
                "geo_ids": "8741",
            }
            data = _get_ree(REE_DEMANDA_URL, params)
            if data is None:
                still_pending.append((chunk_start, chunk_end))
                continue
            values = data["included"][0]["attributes"]["values"]
            rows.extend({"datetime": v["datetime"], "demanda_mwh": v["value"]} for v in values)
            time.sleep(1.0)
        pending = still_pending
        if not pending:
            break
        print(f"  {len(pending)} meses de demanda fallaron, reintentando tras una pausa...")
        time.sleep(20)
    if pending:
        print(f"  aviso: {len(pending)} meses de demanda no se pudieron descargar tras 2 pasadas: {pending}")
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert("Europe/Madrid")
    return df.drop_duplicates("datetime").sort_values("datetime").reset_index(drop=True)


def fetch_ree_generacion(start: date, end: date) -> pd.DataFrame:
    """Devuelve % de generación renovable por DÍA (esta serie de REE es diaria, no horaria).

    Usa el campo `percentage` que ya trae cada tecnología (fracción sobre el total),
    en vez de recalcularlo a partir de los MWh -- así no hay que preocuparse por la
    fila-resumen "Generación total" que la API incluye junto a las tecnologías reales.
    """
    rows = []
    pending = list(_date_chunks(start, end))
    for _pass in range(2):
        still_pending = []
        for chunk_start, chunk_end in pending:
            params = {
                "start_date": f"{chunk_start}T00:00",
                "end_date": f"{chunk_end}T00:00",
                "time_trunc": "day",
                "geo_trunc": "electric_system",
                "geo_limit": "peninsular",
                "geo_ids": "8741",
            }
            data = _get_ree(REE_GENERACION_URL, params)
            if data is None:
                still_pending.append((chunk_start, chunk_end))
                continue
            for tech in data["included"]:
                tech_name = tech.get("type", "")
                if "total" in tech_name.lower():
                    continue  # fila-resumen, no es una tecnología real
                is_renewable = any(k in tech_name.lower() for k in RENEWABLE_KEYWORDS)
                for v in tech["attributes"]["values"]:
                    rows.append({
                        "date": v["datetime"][:10],
                        "tech": tech_name,
                        "percentage": v.get("percentage") or 0.0,
                        "is_renewable": is_renewable,
                    })
            time.sleep(1.0)
        pending = still_pending
        if not pending:
            break
        print(f"  {len(pending)} meses de generación fallaron, reintentando tras una pausa...")
        time.sleep(20)
    if pending:
        print(f"  aviso: {len(pending)} meses de generación no se pudieron descargar tras 2 pasadas: {pending}")
    df = pd.DataFrame(rows)
    daily_pct = (
        df.groupby(["date", "is_renewable"])["percentage"].sum()
        .unstack("is_renewable", fill_value=0)
    )
    daily_pct = daily_pct.rename(columns={True: "renewable_frac", False: "nonrenewable_frac"})
    for col in ("renewable_frac", "nonrenewable_frac"):
        if col not in daily_pct.columns:
            daily_pct[col] = 0.0
    daily_pct["renewable_pct"] = daily_pct["renewable_frac"] * 100
    return daily_pct.reset_index()[["date", "renewable_pct"]]


def _fetch_open_meteo(url: str, lat: float, lon: float, hourly_vars: str,
                       start: date = None, end: date = None, forecast_days: int = None) -> pd.DataFrame:
    params = {"latitude": lat, "longitude": lon, "hourly": hourly_vars, "timezone": "Europe/Madrid"}
    if forecast_days is not None:
        params["forecast_days"] = forecast_days
    else:
        params["start_date"] = str(start)
        params["end_date"] = str(end)
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    hourly = r.json()["hourly"]
    df = pd.DataFrame(hourly)
    # ambiguous/nonexistent="NaT": el cambio de hora (primavera se salta una hora,
    # otoño la repite) deja un par de horas al año sin una localización única --
    # se marcan NaT y se descartan solas al hacer merge/dropna, son un puñado sobre
    # un dataset de ~17.500 filas.
    df["datetime"] = pd.to_datetime(df["time"]).dt.tz_localize(
        "Europe/Madrid", ambiguous="NaT", nonexistent="NaT"
    )
    return df.drop(columns="time").dropna(subset=["datetime"])


WEATHER_VARS = (
    "shortwave_radiation,direct_radiation,cloud_cover,wind_speed_10m,wind_speed_100m,"
    "precipitation,temperature_2m,relative_humidity_2m,surface_pressure"
)


def fetch_weather(start: date = None, end: date = None, forecast_days: int = None) -> pd.DataFrame:
    """Variables y su porqué:
    - shortwave_radiation / direct_radiation / cloud_cover: generación solar --
      shortwave es el mejor proxy de irradiancia total sobre panel, direct_radiation
      distingue cielo despejado de difuso, cloud_cover es la señal más directa de
      "va a hacer sol o no" para quien lea el modelo.
    - wind_speed_10m / wind_speed_100m: generación eólica -- 100m es la altura real
      de buje de un aerogenerador moderno, más fiel que el estándar meteorológico
      de 10m; se dejan ambas para que el modelo elija.
    - precipitation: generación hidroeléctrica (embalses).
    - temperature_2m / relative_humidity_2m: demanda -- calefacción/AC explican
      picos de consumo que el calendario solo no captura; la humedad modula cómo
      de agobiante se siente el calor (más uso de AC a igual temperatura).
    - surface_pressure: proxy barato de frentes/borrascas, que traen viento y lluvia
      a la vez -- variable exploratoria, igual que el oleaje.
    """
    url = METEO_FORECAST_URL if forecast_days else METEO_ARCHIVE_URL
    return _fetch_open_meteo(url, LAT, LON, WEATHER_VARS, start, end, forecast_days)


def fetch_marine(start: date = None, end: date = None, forecast_days: int = None) -> pd.DataFrame:
    return _fetch_open_meteo(MARINE_URL, MARINE_LAT, MARINE_LON, "wave_height,swell_wave_height",
                              start, end, forecast_days)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    years = range(df["datetime"].dt.year.min(), df["datetime"].dt.year.max() + 1)
    es_holidays = holidays.Spain(years=years)
    df["hour"] = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["is_holiday"] = df["datetime"].dt.date.astype("O").apply(lambda d: d in es_holidays).astype(int)
    return df


def build_dataset(start: date, end: date, forecast_days: int = None) -> pd.DataFrame:
    print(f"Descargando demanda REE ({start} a {end})...")
    demanda = fetch_ree_demanda(start, end) if forecast_days is None else pd.DataFrame()

    print("Descargando mix de generación REE (diario)...")
    generacion = fetch_ree_generacion(start, end) if forecast_days is None else pd.DataFrame()

    print("Descargando clima (Open-Meteo)...")
    weather = fetch_weather(start, end, forecast_days)

    print("Descargando oleaje (Open-Meteo Marine)...")
    marine = fetch_marine(start, end, forecast_days)

    df = weather.merge(marine, on="datetime", how="left")
    if not demanda.empty:
        df = df.merge(demanda, on="datetime", how="left")
    if not generacion.empty:
        df["date_str"] = df["datetime"].dt.strftime("%Y-%m-%d")
        df = df.merge(generacion, left_on="date_str", right_on="date", how="left").drop(columns=["date_str", "date"])

    df = add_calendar_features(df)
    return df


if __name__ == "__main__":
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=365 * 5)
    df = build_dataset(start, end)
    df.to_csv(CACHE_PATH, index=False)
    print(f"\nGuardado {CACHE_PATH} con {len(df)} filas.")
    print(f"Rango de fechas: {df['datetime'].min()} a {df['datetime'].max()}")
    print(f"Huecos en demanda_mwh: {df['demanda_mwh'].isna().sum()}")
    print(f"Huecos en renewable_pct: {df['renewable_pct'].isna().sum()}")
    print(df.head())
