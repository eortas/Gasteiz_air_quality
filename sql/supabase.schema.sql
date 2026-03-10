-- ============================================================
-- SCHEMA COMPLETO — Vitoria Air Quality Intelligence
-- vgasteiz-traffic-air-quality
--
-- Tablas:
--   1. traffic_sensors          — inventario de sensores de tráfico
--   2. traffic_measurements     — mediciones de tráfico (últimos 90 días, 1h)
--   3. air_stations             — inventario de estaciones de calidad del aire
--   4. air_measurements         — mediciones de calidad del aire (formato largo)
--   5. weather_measurements     — mediciones meteorológicas (Open-Meteo)
--
-- Vistas:
--   v_traffic_geo               — tráfico + coordenadas + período ZBE
--   v_air_quality               — aire pivotado (formato ancho por contaminante)
--   v_air_quality_geo           — aire pivotado + coordenadas estaciones
--   v_weather                   — meteorología con nombres limpios
--   v_combined_hourly           — cruce tráfico + aire + meteo por hora (analítica)
--
-- Fuentes de datos:
--   Tráfico     : API Ayuntamiento Vitoria-Gasteiz (agregado a 1h)
--   Calidad aire: API Kunak (Red Municipal Vitoria)
--   Meteorología: Open-Meteo Historical Weather API (sin API key)
--
-- Estrategia Supabase:
--   - Todas las tablas: ventana móvil últimos 90 días
--   - Histórico completo: CSVs comprimidos en Supabase Storage
--       csv-traffic / csv-air / csv-weather
--
-- Ejecutar completo en SQL Editor de Supabase
-- ============================================================


-- ════════════════════════════════════════════════════════════
-- 1. TRÁFICO
-- ════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS traffic_sensors (
    code        TEXT PRIMARY KEY,
    name        TEXT,
    provider    TEXT,
    lat         DOUBLE PRECISION,
    lon         DOUBLE PRECISION,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Granularidad: 1 hora (agregado desde 15 min originales de la API)
--   volume   : suma de vehículos en la hora
--   occupancy: media del % de ocupación en la hora
--   load     : media de la carga en la hora
CREATE TABLE IF NOT EXISTS traffic_measurements (
    id          BIGSERIAL PRIMARY KEY,
    code        TEXT        NOT NULL REFERENCES traffic_sensors(code),
    start_date  TIMESTAMPTZ NOT NULL,   -- inicio de la hora (ej. 08:00)
    end_date    TIMESTAMPTZ NOT NULL,   -- fin de la hora   (ej. 09:00)
    volume      REAL,
    occupancy   REAL,
    load        REAL,
    created_at  TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE (code, start_date)
);

CREATE INDEX IF NOT EXISTS idx_traffic_code       ON traffic_measurements(code);
CREATE INDEX IF NOT EXISTS idx_traffic_start_date ON traffic_measurements(start_date);
CREATE INDEX IF NOT EXISTS idx_traffic_date_code  ON traffic_measurements(start_date, code);

-- Vista: tráfico con coordenadas y período ZBE
CREATE OR REPLACE VIEW v_traffic_geo AS
SELECT
    m.start_date,
    m.end_date,
    m.code,
    s.name,
    s.lat,
    s.lon,
    m.volume,
    m.occupancy,
    m.load,
    EXTRACT(YEAR  FROM m.start_date) AS year,
    EXTRACT(MONTH FROM m.start_date) AS month,
    EXTRACT(DOW   FROM m.start_date) AS day_of_week,  -- 0=domingo, 6=sábado
    EXTRACT(HOUR  FROM m.start_date) AS hour,
    CASE
        WHEN m.start_date >= '2025-09-01'                            THEN 'post_zbe'
        WHEN m.start_date >= '2020-03-15'
         AND m.start_date <= '2020-06-21'                            THEN 'covid_lockdown'
        ELSE 'pre_zbe'
    END AS period
FROM traffic_measurements m
JOIN traffic_sensors s ON s.code = m.code;


-- ════════════════════════════════════════════════════════════
-- 2. CALIDAD DEL AIRE (Kunak — Red Municipal Vitoria)
-- ════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS air_stations (
    id      INTEGER PRIMARY KEY,
    nombre  TEXT,
    lat     DOUBLE PRECISION,
    lon     DOUBLE PRECISION
);

-- Coordenadas aproximadas — verificar con inventario oficial Kunak
INSERT INTO air_stations (id, nombre, lat, lon) VALUES
    (1247, 'FUEROS',    42.84738, -2.67187),
    (1248, 'BEATO',     42.84521, -2.67891),
    (2085, 'LANDAZURI', 42.85012, -2.68234),
    (2086, 'PAUL',      42.84103, -2.66891),
    (3460, 'HUETOS',    42.83512, -2.65432),
    (3461, 'ZUMABIDE',  42.86234, -2.69123)
ON CONFLICT (id) DO NOTHING;

-- Formato largo: una fila por (timestamp, estacion, contaminante)
CREATE TABLE IF NOT EXISTS air_measurements (
    id              BIGSERIAL PRIMARY KEY,
    timestamp       TIMESTAMPTZ NOT NULL,
    estacion_id     INTEGER     NOT NULL REFERENCES air_stations(id),
    estacion        TEXT        NOT NULL,
    contaminante    TEXT        NOT NULL,
    valor           REAL,
    unidad          TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE (timestamp, estacion_id, contaminante)
);

CREATE INDEX IF NOT EXISTS idx_air_timestamp    ON air_measurements(timestamp);
CREATE INDEX IF NOT EXISTS idx_air_estacion     ON air_measurements(estacion_id);
CREATE INDEX IF NOT EXISTS idx_air_contaminante ON air_measurements(contaminante);
CREATE INDEX IF NOT EXISTS idx_air_est_cont_ts  ON air_measurements(estacion_id, contaminante, timestamp);

-- Vista pivotada: una fila por (timestamp, estacion) con columnas por contaminante
CREATE OR REPLACE VIEW v_air_quality AS
SELECT
    timestamp,
    estacion_id,
    estacion,
    MAX(CASE WHEN contaminante = 'NO2'         THEN valor END) AS no2,
    MAX(CASE WHEN contaminante = 'PM10'        THEN valor END) AS pm10,
    MAX(CASE WHEN contaminante = 'PM2.5'       THEN valor END) AS pm25,
    MAX(CASE WHEN contaminante = 'ICA'         THEN valor END) AS ica,
    MAX(CASE WHEN contaminante = 'temperatura' THEN valor END) AS temperatura,
    MAX(CASE WHEN contaminante = 'humedad'     THEN valor END) AS humedad,
    MAX(CASE WHEN contaminante = 'presion'     THEN valor END) AS presion,
    MAX(CASE WHEN contaminante = 'viento_vel'  THEN valor END) AS viento_vel,
    MAX(CASE WHEN contaminante = 'viento_dir'  THEN valor END) AS viento_dir,
    CASE
        WHEN timestamp >= '2025-09-01'                               THEN 'post_zbe'
        WHEN timestamp >= '2020-03-15' AND timestamp <= '2020-06-21' THEN 'covid_lockdown'
        ELSE 'pre_zbe'
    END AS period
FROM air_measurements
GROUP BY timestamp, estacion_id, estacion;

-- Vista con coordenadas incluidas
CREATE OR REPLACE VIEW v_air_quality_geo AS
SELECT
    a.*,
    s.lat,
    s.lon
FROM v_air_quality a
JOIN air_stations s ON s.id = a.estacion_id;


-- ════════════════════════════════════════════════════════════
-- 3. METEOROLOGÍA (Open-Meteo — Vitoria-Gasteiz)
-- ════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS weather_measurements (
    id                                BIGSERIAL PRIMARY KEY,
    timestamp                         TIMESTAMPTZ NOT NULL,
    timestamp_local                   TEXT,

    -- Temperatura
    temperature_2m                    REAL,
    apparent_temperature              REAL,
    dewpoint_2m                       REAL,

    -- Humedad
    relative_humidity_2m              REAL,
    vapour_pressure_deficit           REAL,

    -- Precipitación
    precipitation                     REAL,
    rain                              REAL,
    snowfall                          REAL,
    snow_depth                        REAL,

    -- Presión
    pressure_msl                      REAL,
    surface_pressure                  REAL,

    -- Viento
    wind_speed_10m                    REAL,
    wind_direction_10m                REAL,
    wind_gusts_10m                    REAL,

    -- Radiación y cielo
    cloud_cover                       REAL,
    visibility                        REAL,
    is_day                            REAL,
    sunshine_duration                 REAL,
    weather_code                      REAL,

    -- Capa límite (dispersión contaminantes)
    boundary_layer_height             REAL,

    -- Metadatos geográficos
    latitude                          REAL,
    longitude                         REAL,
    elevation_m                       REAL,

    -- Resúmenes diarios
    daily_weather_code                REAL,
    daily_temperature_2m_max          REAL,
    daily_temperature_2m_min          REAL,
    daily_temperature_2m_mean         REAL,
    daily_apparent_temperature_max    REAL,
    daily_apparent_temperature_min    REAL,
    daily_precipitation_sum           REAL,
    daily_rain_sum                    REAL,
    daily_snowfall_sum                REAL,
    daily_precipitation_hours         REAL,
    daily_wind_speed_10m_max          REAL,
    daily_wind_gusts_10m_max          REAL,
    daily_wind_direction_10m_dominant REAL,
    daily_shortwave_radiation_sum     REAL,

    created_at                        TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE (timestamp)
);

CREATE INDEX IF NOT EXISTS idx_weather_timestamp ON weather_measurements(timestamp);

-- Vista meteorología con nombres limpios y período ZBE
CREATE OR REPLACE VIEW v_weather AS
SELECT
    timestamp,
    timestamp_local,
    temperature_2m              AS temperature,
    apparent_temperature,
    dewpoint_2m,
    relative_humidity_2m        AS humidity,
    precipitation,
    rain,
    snowfall,
    pressure_msl                AS pressure,
    wind_speed_10m              AS wind_speed,
    wind_direction_10m          AS wind_direction,
    wind_gusts_10m              AS wind_gust,
    cloud_cover,
    visibility,
    is_day,
    sunshine_duration,
    weather_code,
    boundary_layer_height,
    daily_temperature_2m_max    AS temp_max,
    daily_temperature_2m_min    AS temp_min,
    daily_precipitation_sum,
    daily_wind_speed_10m_max,
    CASE
        WHEN timestamp >= '2025-09-01'                                 THEN 'post_zbe'
        WHEN timestamp >= '2020-03-15' AND timestamp <= '2020-06-21'   THEN 'covid_lockdown'
        ELSE 'pre_zbe'
    END AS period
FROM weather_measurements;


-- ════════════════════════════════════════════════════════════
-- 4. VISTA ANALÍTICA COMBINADA
-- Cruce horario tráfico + aire + meteo
-- Base para el modelo predictivo
-- ════════════════════════════════════════════════════════════

CREATE OR REPLACE VIEW v_combined_hourly AS
SELECT
    -- Tiempo
    date_trunc('hour', a.timestamp)                 AS hour,
    a.period,

    -- Calidad del aire (media de todas las estaciones por hora)
    ROUND(AVG(a.no2)::NUMERIC,  2)                  AS no2_avg,
    ROUND(AVG(a.pm10)::NUMERIC, 2)                  AS pm10_avg,
    ROUND(AVG(a.pm25)::NUMERIC, 2)                  AS pm25_avg,
    ROUND(AVG(a.ica)::NUMERIC,  1)                  AS ica_avg,

    -- Meteorología
    ROUND(AVG(w.temperature)::NUMERIC,          1)  AS temperature,
    ROUND(AVG(w.humidity)::NUMERIC,             1)  AS humidity,
    ROUND(AVG(w.wind_speed)::NUMERIC,           1)  AS wind_speed,
    ROUND(AVG(w.wind_direction)::NUMERIC,       0)  AS wind_direction,
    ROUND(SUM(w.precipitation)::NUMERIC,        1)  AS precipitation,
    ROUND(AVG(w.cloud_cover)::NUMERIC,          1)  AS cloud_cover,
    ROUND(AVG(w.visibility)::NUMERIC,           0)  AS visibility,
    ROUND(AVG(w.boundary_layer_height)::NUMERIC, 0) AS boundary_layer_height,

    -- Tráfico (suma de volumen de todos los sensores por hora)
    ROUND(SUM(t.volume)::NUMERIC,    0)             AS total_volume,
    ROUND(AVG(t.occupancy)::NUMERIC, 2)             AS avg_occupancy,

    -- Variables temporales (features para ML)
    EXTRACT(DOW   FROM a.timestamp)                 AS day_of_week,
    EXTRACT(HOUR  FROM a.timestamp)                 AS hour_of_day,
    EXTRACT(MONTH FROM a.timestamp)                 AS month

FROM v_air_quality a
LEFT JOIN v_weather w
    ON date_trunc('hour', w.timestamp) = date_trunc('hour', a.timestamp)
LEFT JOIN v_traffic_geo t
    ON date_trunc('hour', t.start_date) = date_trunc('hour', a.timestamp)
GROUP BY
    date_trunc('hour', a.timestamp),
    a.period,
    EXTRACT(DOW   FROM a.timestamp),
    EXTRACT(HOUR  FROM a.timestamp),
    EXTRACT(MONTH FROM a.timestamp);