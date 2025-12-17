/*
CONCEPT: Hourly vehicle traffic dataset (synthetic dataset is exported later as CSV)

Goal:
- Produce a complete hourly panel: (PTF × date × hour)
- Aggregate counts by direction and by vehicle category
- Keep logic portable (schema/table names are placeholders)

Assumptions:
- fact_crossings: transactional fact table with at least:
  id, pctf, cross_date (date), cross_time (time/timestamp), direction, vehicle_type, is_driver
- direction values: 'INTRARE' / 'IESIRE'
- vehicle_type codes are mapped to categories:
  Cars:        ('B','B1','B2')
  Buses:       ('D','D1','DE','D1E')
  Light trucks:('C1','C1E')
  Trucks:      ('C','CE')
*/

WITH hours AS (
    SELECT generate_series(0, 23) AS hour
),
dates AS (
    SELECT generate_series(DATE '2022-12-01', DATE '2025-11-30', INTERVAL '1 day')::date AS dt
),
pctf AS (
    SELECT DISTINCT pctf
    FROM fact_crossings
    WHERE pctf NOT IN ('EXAMPLE_PCTF_1','EXAMPLE_PCTF_2')
)
SELECT
    p.pctf,
    d.dt AS date,
    EXTRACT(ISODOW FROM d.dt) AS day_of_week,
    h.hour,

    /* Direction totals */
    SUM(CASE WHEN f.direction = 'INTRARE' THEN 1 ELSE 0 END) AS in_count,
    SUM(CASE WHEN f.direction = 'IESIRE'  THEN 1 ELSE 0 END) AS out_count,

    /* Vehicle categories - IN */
    SUM(CASE WHEN f.direction = 'INTRARE' AND f.vehicle_type IN ('B','B1','B2') THEN 1 ELSE 0 END) AS car_in,
    SUM(CASE WHEN f.direction = 'INTRARE' AND f.vehicle_type IN ('D','D1','DE','D1E') THEN 1 ELSE 0 END) AS bus_in,
    SUM(CASE WHEN f.direction = 'INTRARE' AND f.vehicle_type IN ('C1','C1E') THEN 1 ELSE 0 END) AS light_trucks_in,
    SUM(CASE WHEN f.direction = 'INTRARE' AND f.vehicle_type IN ('C','CE') THEN 1 ELSE 0 END) AS trucks_in,

    /* Vehicle categories - OUT */
    SUM(CASE WHEN f.direction = 'IESIRE'  AND f.vehicle_type IN ('B','B1','B2') THEN 1 ELSE 0 END) AS car_out,
    SUM(CASE WHEN f.direction = 'IESIRE'  AND f.vehicle_type IN ('D','D1','DE','D1E') THEN 1 ELSE 0 END) AS bus_out,
    SUM(CASE WHEN f.direction = 'IESIRE'  AND f.vehicle_type IN ('C1','C1E') THEN 1 ELSE 0 END) AS light_trucks_out,
    SUM(CASE WHEN f.direction = 'IESIRE'  AND f.vehicle_type IN ('C','CE') THEN 1 ELSE 0 END) AS trucks_out,

    /* Total vehicles (all directions) */
    COUNT(f.id) AS total

FROM pctf p
CROSS JOIN dates d
CROSS JOIN hours h
LEFT JOIN fact_crossings f
    ON f.pctf = p.pctf
   AND f.cross_date = d.dt
   AND EXTRACT(HOUR FROM f.cross_time) = h.hour
   AND f.is_driver = TRUE     -- include only vehicle records (driver=vehicle indicator)
GROUP BY
    p.pctf, d.dt, day_of_week, h.hour
ORDER BY
    p.pctf, d.dt, h.hour;
