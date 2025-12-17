/* =========================================================
   Conceptual SQL: Hourly aggregation of people crossings
   Purpose:
     Construction of ML-ready datasets for analysis and
     prediction of human border traffic.

   Notes:
   - This query illustrates aggregation and feature engineering logic only
   - No production schemas, real checkpoint identifiers or infrastructure
   - All data are assumed to be aggregated and anonymized
   ========================================================= */

WITH hours AS (
    SELECT generate_series(0, 23) AS hour
),
dates AS (
    SELECT generate_series(:start_date::date, :end_date::date, '1 day') AS data
),
checkpoints AS (
    SELECT DISTINCT checkpoint_id
    FROM dim_checkpoints
    WHERE checkpoint_type = 'ROAD'
)

SELECT
    c.checkpoint_id,
    d.data,
    EXTRACT(ISODOW FROM d.data) AS day_of_week,
    h.hour,

    -- Total people flow
    SUM(CASE WHEN f.direction = 'IN'  THEN 1 ELSE 0 END) AS people_in,
    SUM(CASE WHEN f.direction = 'OUT' THEN 1 ELSE 0 END) AS people_out,
    COUNT(f.event_id) AS total_people,

    -- Citizenship groups (aggregated)
    SUM(CASE WHEN f.citizenship_group = 'MD' THEN 1 ELSE 0 END) AS md,
    SUM(CASE WHEN f.citizenship_group = 'RO' THEN 1 ELSE 0 END) AS ro,
    SUM(CASE WHEN f.citizenship_group = 'UA' THEN 1 ELSE 0 END) AS ua,
    SUM(CASE WHEN f.citizenship_group = 'OTHER' THEN 1 ELSE 0 END) AS other

FROM checkpoints c
CROSS JOIN dates d
CROSS JOIN hours h
LEFT JOIN fact_people_crossings f
    ON f.checkpoint_id = c.checkpoint_id
   AND f.crossing_date = d.data
   AND EXTRACT(HOUR FROM f.crossing_time) = h.hour

GROUP BY
    c.checkpoint_id,
    d.data,
    day_of_week,
    h.hour

ORDER BY
    c.checkpoint_id,
    d.data,
    h.hour;