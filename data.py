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
import numpy as np
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

# Madrid no es zona eólica -- los parques de verdad están concentrados en el valle
# del Ebro (Aragón, viento de cierzo), la meseta norte (Castilla y León) y Galicia.
# Medir el viento en Madrid para explicar generación eólica nacional es medir en el
# sitio equivocado. Se añaden estos 3 puntos, representativos de las regiones con
# más potencia eólica instalada de España, como variable específica para eso.
WIND_REGIONS = {
    "aragon": (41.65, -0.88),
    "castilla_leon": (42.34, -3.70),
    "galicia": (43.36, -8.41),
}

# La demanda nacional tampoco depende solo de Madrid -- son las 5 áreas
# metropolitanas más pobladas, ponderadas por población aproximada (fuente:
# cifras de población por provincia/comunidad, redondeadas) para que Madrid y
# Barcelona pesen lo que de verdad pesan en el consumo eléctrico nacional y
# Bilbao no cuente lo mismo que ellas.
DEMAND_CITIES = {
    "madrid": (40.4168, -3.7038, 6_700_000),
    "barcelona": (41.3851, 2.1734, 5_700_000),
    "valencia": (39.4699, -0.3763, 2_600_000),
    "sevilla": (37.3891, -5.9845, 1_950_000),
    "bilbao": (43.2630, -2.9350, 1_150_000),
}
_DEMAND_POP_TOTAL = sum(pop for _, _, pop in DEMAND_CITIES.values())

REE_PRECIO_URL = "https://apidatos.ree.es/es/datos/mercados/precios-mercados-tiempo-real"

# Población aproximada por comunidad autónoma (millones, redondeado) -- para
# ponderar qué fracción del país está de fiesta regional un día dado, en vez de
# tratar un festivo autonómico pequeño igual que uno nacional.
SPAIN_REGION_POPULATION = {
    "AN": 8.5, "CT": 7.7, "MD": 6.7, "VC": 5.1, "GA": 2.7, "CL": 2.4, "PV": 2.2,
    "CN": 2.2, "CM": 2.0, "MC": 1.5, "AR": 1.3, "IB": 1.2, "EX": 1.05, "AS": 1.0,
    "NC": 0.66, "CB": 0.58, "RI": 0.32,
}
_REGION_POP_TOTAL = sum(SPAIN_REGION_POPULATION.values())

RENEWABLE_KEYWORDS = ["solar", "eólic", "eolic", "hidrául", "hidraul", "hidroeólica", "renovable", "geotérmica"]

# Nombre de tecnología (tal cual lo da REE) -> columna corta, para poder guardar el
# desglose completo del mix (no solo el agregado renovable/no renovable) y montar
# gráficas de "generation mix" tipo stacked-area, el estándar visual del sector.
TECH_SLUG = {
    "hidráulica": "hidraulica", "nuclear": "nuclear", "carbón": "carbon",
    "fuel + gas": "fuel_gas", "turbina de vapor": "turbina_vapor",
    "ciclo combinado": "ciclo_combinado", "eólica": "eolica",
    "solar fotovoltaica": "solar_fv", "solar térmica": "solar_termica",
    "otras renovables": "otras_renovables", "cogeneración": "cogeneracion",
    "residuos no renovables": "residuos_no_renovables", "residuos renovables": "residuos_renovables",
}

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
    """Devuelve % de generación renovable por DÍA (esta serie de REE es diaria, no
    horaria -- se probó en vivo pedir `time_trunc=hour` a este endpoint en concreto
    y REE lo rechaza siempre con 400, a diferencia de demanda que sí lo soporta).

    Además del agregado renovable/no renovable, guarda el % de cada tecnología por
    separado (`tech_<nombre>_pct`) -- no solo para tener más variables, sino porque
    es lo que hace falta para una gráfica de mix de generación tipo stacked-area
    (el estándar visual del sector: cuánto pone cada tecnología en cada momento),
    y porque solar y eólica tienen dinámicas muy distintas (solar es un ciclo anual
    y diario muy predecible, eólica es mucho más errática) que se pierden al
    mirar solo el agregado.

    IMPORTANTE -- bug real de la API descubierto en vivo: el campo `percentage`
    que trae cada tecnología NO es la fracción sobre el total (aunque el nombre
    del campo lo sugiere). Se comprobó pidiendo un día suelto: sumando el
    `percentage` de las 13 tecnologías reales (sin la fila "Generación total")
    da exactamente 0.5000000, la mitad de lo que debería (1.0) -- pero sumando
    los `value` (MWh) de esas mismas 13 tecnologías da EXACTAMENTE el mismo MWh
    que la fila "Generación total". Es decir: `percentage` viene escalado a la
    mitad por lo que sea (posible peculiaridad interna de REE), pero `value` es
    fiable. Por eso aquí el % de cada tecnología se calcula a mano como
    `value_tecnología / value_total`, ignorando el campo `percentage` de la API
    -- de lo contrario el % renovable habría salido sistemáticamente a la mitad
    de su valor real (p.ej. un 46% real habría salido como ~23%).
    """
    rows = []
    totals = {}
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
                    for v in tech["attributes"]["values"]:
                        totals[v["datetime"][:10]] = v["value"]
                    continue  # fila-resumen: se guarda como denominador, no como fila de tecnología
                is_renewable = any(k in tech_name.lower() for k in RENEWABLE_KEYWORDS)
                slug = TECH_SLUG.get(tech_name.lower().strip(), tech_name.lower().replace(" ", "_"))
                for v in tech["attributes"]["values"]:
                    rows.append({
                        "date": v["datetime"][:10],
                        "slug": slug,
                        "value_mwh": v["value"],
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
    df["total_mwh"] = df["date"].map(totals)

    daily_mwh = (
        df.groupby(["date", "is_renewable"])["value_mwh"].sum()
        .unstack("is_renewable", fill_value=0)
    )
    daily_mwh = daily_mwh.rename(columns={True: "renewable_mwh", False: "nonrenewable_mwh"})
    for col in ("renewable_mwh", "nonrenewable_mwh"):
        if col not in daily_mwh.columns:
            daily_mwh[col] = 0.0
    daily_mwh["total_mwh"] = daily_mwh.index.map(totals)
    # En ~0.3% de los días la suma de tecnologías supera muy ligeramente (100-103%)
    # el valor de "Generación total" que publica REE -- redondeos independientes
    # entre ambas cifras en la fuente, no un error de este script. Se recorta a 100.
    daily_mwh["renewable_pct"] = (daily_mwh["renewable_mwh"] / daily_mwh["total_mwh"] * 100).clip(upper=100)

    df["percentage"] = df["value_mwh"] / df["total_mwh"] * 100
    tech_pivot = df.pivot_table(index="date", columns="slug", values="percentage", aggfunc="sum", fill_value=0.0)
    tech_pivot.columns = [f"tech_{c}_pct" for c in tech_pivot.columns]

    return daily_mwh.reset_index()[["date", "renewable_pct"]].merge(
        tech_pivot.reset_index(), on="date", how="left"
    )


def _fetch_open_meteo(url: str, lat: float, lon: float, hourly_vars: str,
                       start: date = None, end: date = None, forecast_days: int = None) -> pd.DataFrame:
    # Open-Meteo devuelve el viento en km/h por defecto, no en m/s -- se pide
    # explícitamente m/s para que `wind_power_proxy` (potencia ~ velocidad^3,
    # con el corte físico pensado en m/s) tenga sentido, y para que cualquiera
    # que lea el dataset no tenga que adivinar la unidad.
    params = {"latitude": lat, "longitude": lon, "hourly": hourly_vars, "timezone": "Europe/Madrid",
              "wind_speed_unit": "ms"}
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


def fetch_wind_regions(start: date = None, end: date = None, forecast_days: int = None) -> pd.DataFrame:
    """Viento en las 3 regiones con más potencia eólica instalada de España (no en
    Madrid, que no es zona eólica) -- da una media (cuánto viento hay "en general")
    y un máximo (si al menos una región está soplando fuerte, aunque las otras no)
    entre las tres, más una variable física: la potencia de un aerogenerador escala
    aprox. con el cubo de la velocidad del viento (hasta el punto en que ya genera a
    potencia nominal), así que `v_media^3` (recortado a 12 m/s, la zona típica antes
    de saturar) es una variable mucho más cercana a "cuánta energía se puede sacar"
    que la velocidad en bruto -- un árbol de decisión puede aproximar esa curva a
    base de cortes, pero dársela ya calculada le ahorra tener que reconstruirla.
    """
    url = METEO_FORECAST_URL if forecast_days else METEO_ARCHIVE_URL
    per_region = []
    for name, (lat, lon) in WIND_REGIONS.items():
        region_df = _fetch_open_meteo(url, lat, lon, "wind_speed_10m,wind_speed_100m",
                                       start, end, forecast_days)
        region_df = region_df.rename(columns={"wind_speed_10m": f"wind10_{name}",
                                                "wind_speed_100m": f"wind100_{name}"})
        per_region.append(region_df.set_index("datetime"))
        time.sleep(0.5)
    merged = pd.concat(per_region, axis=1, join="inner")
    wind100_cols = [f"wind100_{name}" for name in WIND_REGIONS]
    wind10_cols = [f"wind10_{name}" for name in WIND_REGIONS]
    out = pd.DataFrame(index=merged.index)
    out["wind_national_mean_100m"] = merged[wind100_cols].mean(axis=1)
    out["wind_national_max_100m"] = merged[wind100_cols].max(axis=1)
    out["wind_national_std_100m"] = merged[wind100_cols].std(axis=1)
    out["wind_national_mean_10m"] = merged[wind10_cols].mean(axis=1)
    out["wind_power_proxy"] = out["wind_national_mean_100m"].clip(upper=12) ** 3
    return out.reset_index()


def fetch_demand_climate(start: date = None, end: date = None, forecast_days: int = None) -> pd.DataFrame:
    """Temperatura ponderada por población de las 5 áreas metropolitanas más
    grandes de España, no solo Madrid -- mismo razonamiento que con el viento:
    la demanda nacional depende de cuánto frío/calor hace donde vive la gente
    de verdad, no en un único punto arbitrario. De la temperatura ponderada se
    derivan HDD/CDD (grados-día de calefacción/refrigeración): `HDD = max(0,
    18-T)`, `CDD = max(0, T-18)`, el estándar del sector para modelar demanda
    sensible al clima -- representan directamente "cuánta calefacción/aire
    acondicionado hace falta" y capturan el doble pico anual de España (verano
    por AC, invierno por calefacción) mucho mejor que la temperatura en bruto
    o que un único seno/coseno anual (que solo tiene un pico, no dos)."""
    url = METEO_FORECAST_URL if forecast_days else METEO_ARCHIVE_URL
    per_city = []
    for name, (lat, lon, _pop) in DEMAND_CITIES.items():
        city_df = _fetch_open_meteo(url, lat, lon, "temperature_2m,relative_humidity_2m",
                                     start, end, forecast_days)
        city_df = city_df.rename(columns={"temperature_2m": f"temp_{name}",
                                           "relative_humidity_2m": f"rh_{name}"})
        per_city.append(city_df.set_index("datetime"))
        time.sleep(0.5)
    merged = pd.concat(per_city, axis=1, join="inner")
    out = pd.DataFrame(index=merged.index)
    temp_national = sum(merged[f"temp_{name}"] * pop for name, (_, _, pop) in DEMAND_CITIES.items()) / _DEMAND_POP_TOTAL
    rh_national = sum(merged[f"rh_{name}"] * pop for name, (_, _, pop) in DEMAND_CITIES.items()) / _DEMAND_POP_TOTAL
    out["temperature_national"] = temp_national
    out["humidity_national"] = rh_national
    out["hdd"] = (18 - temp_national).clip(lower=0)
    out["cdd"] = (temp_national - 18).clip(lower=0)
    return out.reset_index()


def fetch_ree_price(start: date, end: date) -> pd.DataFrame:
    """Precio spot horario del mercado eléctrico (REE apidatos, sin token --
    endpoint distinto al de generación, no tiene el límite de 14 días de los
    otros). Variable exploratoria: la relación con la demanda es más
    "consecuencia" que "causa" (el precio sube cuando la demanda ya es alta,
    no al revés), así que aquí solo se usa como *lag* (precio de ayer, de hace
    una semana) -- nunca como "precio futuro conocido", porque en producción
    el precio del día siguiente en adelante no se conoce con certeza más allá
    de un día (mercado diario), y usar el precio real futuro del holdout como
    si se conociera de antemano sería una fuga de información sutil."""
    rows = []
    pending = list(_date_chunks(start, end))
    for _pass in range(2):
        still_pending = []
        for chunk_start, chunk_end in pending:
            params = {"start_date": f"{chunk_start}T00:00", "end_date": f"{chunk_end}T00:00",
                      "time_trunc": "hour"}
            data = _get_ree(REE_PRECIO_URL, params)
            if data is None:
                still_pending.append((chunk_start, chunk_end))
                continue
            for serie in data["included"]:
                if "spot" not in serie.get("type", "").lower():
                    continue
                for v in serie["attributes"]["values"]:
                    rows.append({"datetime": v["datetime"], "precio_eur_mwh": v["value"]})
            time.sleep(1.0)
        pending = still_pending
        if not pending:
            break
        time.sleep(20)
    if pending:
        print(f"  aviso: {len(pending)} tramos de precio no se pudieron descargar: {pending}")
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert("Europe/Madrid")
    return df.drop_duplicates("datetime").sort_values("datetime").reset_index(drop=True)


def _regional_holiday_weight_map(years) -> dict:
    """% de la población española de fiesta autonómica cada día -- una fiesta en
    La Rioja (0.6% de la población) no debería pesar igual que una en Andalucía
    o Cataluña (18% y 16%). Se calcula una vez por rango de años (cachear años
    repetidos sería una micro-optimización que no hace falta a este tamaño)."""
    weight_by_date = {}
    for region, pop in SPAIN_REGION_POPULATION.items():
        try:
            region_holidays = holidays.Spain(years=years, subdiv=region)
        except NotImplementedError:
            continue
        weight = pop / _REGION_POP_TOTAL
        for d in region_holidays:
            weight_by_date[d] = weight_by_date.get(d, 0.0) + weight
    return weight_by_date


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    years = range(df["datetime"].dt.year.min(), df["datetime"].dt.year.max() + 1)
    es_holidays = holidays.Spain(years=years)
    regional_weight = _regional_holiday_weight_map(years)
    df["hour"] = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["is_holiday"] = df["datetime"].dt.date.astype("O").apply(lambda d: d in es_holidays).astype(int)
    df["regional_holiday_pct"] = (
        df["datetime"].dt.date.astype("O").apply(lambda d: regional_weight.get(d, 0.0)) * 100
    )
    # Ciclo anual como seno/coseno (día 365 y día 1 quedan "cerca" el uno del otro,
    # cosa que un entero de día-del-año 1-365 no representa bien). Es la variable
    # que más explica el % de generación solar: en España hay mucha más luz e
    # irradiancia en junio que en diciembre, con un patrón que se repite cada año
    # de forma mucho más fiable que la meteorología día a día.
    doy = df["datetime"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return df


def trim_incomplete_trailing_days(df: pd.DataFrame, min_hours: int = 20,
                                   check_cols: tuple = ("demanda_mwh",)) -> pd.DataFrame:
    """Descubierto en vivo: `data.py` pide datos hasta "ayer" (`date.today() - 1`),
    pero REE no siempre tiene el día anterior completo cuando se descarga -- a
    veces solo hay 1-2 horas publicadas. Ese día a medias no sale como NaN (que
    se filtraría solo): sale con un valor real pero calculado sobre casi ninguna
    hora, lo que contamina tanto el entrenamiento como, si cae dentro del
    holdout de evaluación, la propia "verdad" contra la que se mide cada modelo
    (así fue como se detectó: un "desplome" de % renovable a final del holdout
    que en realidad era un día con 1 sola hora de demanda publicada). Se recortan
    del final los días que no lleguen a `min_hours` horas reales en ninguna de
    `check_cols`."""
    dates = sorted(df["datetime"].dt.date.unique())
    while dates:
        day_mask = df["datetime"].dt.date == dates[-1]
        if all(df.loc[day_mask, c].notna().sum() >= min_hours for c in check_cols):
            break
        dates.pop()
    if not dates:
        return df
    return df[df["datetime"].dt.date <= dates[-1]].reset_index(drop=True)


def build_dataset(start: date, end: date, forecast_days: int = None) -> pd.DataFrame:
    print(f"Descargando demanda REE ({start} a {end})...")
    demanda = fetch_ree_demanda(start, end) if forecast_days is None else pd.DataFrame()

    print("Descargando mix de generación REE (diario)...")
    generacion = fetch_ree_generacion(start, end) if forecast_days is None else pd.DataFrame()

    print("Descargando clima (Open-Meteo)...")
    weather = fetch_weather(start, end, forecast_days)

    print("Descargando oleaje (Open-Meteo Marine)...")
    marine = fetch_marine(start, end, forecast_days)

    print("Descargando viento en las 3 regiones eólicas (Aragón, Castilla y León, Galicia)...")
    wind_regions = fetch_wind_regions(start, end, forecast_days)

    print("Descargando clima ponderado por población en 5 áreas metropolitanas...")
    demand_climate = fetch_demand_climate(start, end, forecast_days)

    df = (weather.merge(marine, on="datetime", how="left")
          .merge(wind_regions, on="datetime", how="left")
          .merge(demand_climate, on="datetime", how="left"))
    if not demanda.empty:
        df = df.merge(demanda, on="datetime", how="left")
    if not generacion.empty:
        df["date_str"] = df["datetime"].dt.strftime("%Y-%m-%d")
        df = df.merge(generacion, left_on="date_str", right_on="date", how="left").drop(columns=["date_str", "date"])
    if forecast_days is None:
        print("Descargando precio spot horario (REE)...")
        precio = fetch_ree_price(start, end)
        df = df.merge(precio, on="datetime", how="left")

    df = add_calendar_features(df)
    if forecast_days is None:
        before = df["datetime"].dt.date.nunique()
        df = trim_incomplete_trailing_days(df)
        after = df["datetime"].dt.date.nunique()
        if after < before:
            print(f"  recortados {before - after} día(s) incompletos al final (REE aún no los había publicado del todo)")
    return df


if __name__ == "__main__":
    # 10 años en vez de 5 -- el GRU entrenado desde cero necesita más ejemplos
    # que los modelos anteriores para no sobreajustar. Aviso honesto: para %
    # renovable esto mete años en los que apenas había solar/eólica instalada
    # en España (el mix ha cambiado mucho desde 2016), así que esa parte del
    # histórico antiguo es menos representativa del sistema actual -- se
    # documenta en el README, no se esconde.
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=365 * 10)
    df = build_dataset(start, end)
    df.to_csv(CACHE_PATH, index=False)
    print(f"\nGuardado {CACHE_PATH} con {len(df)} filas.")
    print(f"Rango de fechas: {df['datetime'].min()} a {df['datetime'].max()}")
    print(f"Huecos en demanda_mwh: {df['demanda_mwh'].isna().sum()}")
    print(f"Huecos en renewable_pct: {df['renewable_pct'].isna().sum()}")
    print(df.head())
