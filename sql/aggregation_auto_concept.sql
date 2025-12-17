/* =========================================================
   Conceptual SQL: Hourly aggregation of vehicle crossings
   Purpose: Feature engineering for ML models
   Note: This query illustrates aggregation logic only.
   ========================================================= */

WITH hours AS (
    SELECT generate_series(0, 23) AS hour
),
dates AS (
    SELECT generate_series(:start_date::date, :end_date::date, '1 day') AS data
),
pctf AS (
    SELECT DISTINCT pctf
    FROM dim_pctf
    -- Example exclusion of non-road checkpoints
    WHERE pctf_type = 'ROAD'
)

SELECT
    p.pctf,
    d.data,
    EXTRACT(ISODOW FROM d.data) AS day_of_week,
    h.hour,

    -- Total vehicle flow
    SUM(CASE WHEN f.direction = 'IN'  THEN 1 ELSE 0 END) AS vehicles_in,
    SUM(CASE WHEN f.direction = 'OUT' THEN 1 ELSE 0 END) AS vehicles_out,

    -- Vehicle categories
    SUM(CASE WHEN f.direction = 'IN'  AND f.vehicle_type IN ('CAR') THEN 1 ELSE 0 END) AS car_in,
    SUM(CASE WHEN f.direction = 'OUT' AND f.vehicle_type IN ('CAR') THEN 1 ELSE 0 END) AS car_out,

    SUM(CASE WHEN f.direction = 'IN'  AND f.vehicle_type IN ('BUS') THEN 1 ELSE 0 END) AS bus_in,
    SUM(CASE WHEN f.direction = 'OUT' AND f.vehicle_type IN ('BUS') THEN 1 ELSE 0 END) AS bus_out,

    SUM(CASE WHEN f.direction = 'IN'  AND f.vehicle_type IN ('LIGHT_TRUCK') THEN 1 ELSE 0 END) AS light_truck_in,
    SUM(CASE WHEN f.direction = 'OUT' AND f.vehicle_type IN ('LIGHT_TRUCK') THEN 1 ELSE 0 END) AS light_truck_out,

    SUM(CASE WHEN f.direction = 'IN'  AND f.vehicle_type IN ('TRUCK') THEN 1 ELSE 0 END) AS truck_in,
    SUM(CASE WHEN f.direction = 'OUT' AND f.vehicle_type IN ('TRUCK') THEN 1 ELSE 0 END) AS truck_out,

    COUNT(f.event_id) AS total_vehicles

FROM pctf p
CROSS JOIN dates d
CROSS JOIN hours h
LEFT JOIN fact_crossings f
    ON f.pctf = p.pctf
   AND f.crossing_date = d.data
   AND EXTRACT(HOUR FROM f.crossing_time) = h.hour
   AND f.is_driver = TRUE

GROUP BY
    p.pctf, d.data, day_of_week, h.hour

ORDER BY
    p.pctf, d.data, h.hour;