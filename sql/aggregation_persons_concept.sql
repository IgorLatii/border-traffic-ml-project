/*
CONCEPT: Hourly person traffic dataset (synthetic dataset is exported later as CSV)

Goal:
- Produce a complete hourly panel: (PCTF × date × hour)
- Aggregate counts by direction
- Aggregate citizenship into groups:
  MD, RO, UA, EU (excluding RO), OTHER (non-EU excluding MD/RO/UA)

Assumptions:
- fact_crossings: transactional fact table with at least:
  id, pctf, cross_date (date), cross_time (time/timestamp), direction, citizenship_full
- cet_tara: reference table with at least:
  cet_code (ISO-3), cet_full_name, is_eu (boolean)
- Join is done by normalized full citizenship name (adapt as needed).
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

    --Direction totals
    SUM(CASE WHEN f.direction = 'INTRARE' THEN 1 ELSE 0 END) AS in_count,
    SUM(CASE WHEN f.direction = 'IESIRE'  THEN 1 ELSE 0 END) AS out_count,
    COUNT(f.id) AS total,

    --Core citizenship groups (examples use full names; adapt to your coding)
    SUM(CASE WHEN f.citizenship_full = 'MOLDOVA' THEN 1 ELSE 0 END) AS md,
    SUM(CASE WHEN f.citizenship_full = 'ROMANIA' THEN 1 ELSE 0 END) AS ro,
    SUM(CASE WHEN f.citizenship_full = 'UKRAINE' THEN 1 ELSE 0 END) AS ua,

    --EU group (excluding MD/RO/UA explicitly)
    SUM(
        CASE
            WHEN r.is_eu = TRUE
             AND f.citizenship_full NOT IN ('MOLDOVA','ROMANIA','UKRAINE')
            THEN 1 ELSE 0
        END
    ) AS eu,

    --Non-EU group (excluding MD/RO/UA explicitly)
    SUM(
        CASE
            WHEN (r.is_eu = FALSE OR r.is_eu IS NULL)
             AND f.citizenship_full NOT IN ('MOLDOVA','ROMANIA','UKRAINE')
            THEN 1 ELSE 0
        END
    ) AS other

FROM pctf p
CROSS JOIN dates d
CROSS JOIN hours h
LEFT JOIN fact_crossings f
    ON f.pctf = p.pctf
   AND f.cross_date = d.dt
   AND EXTRACT(HOUR FROM f.cross_time) = h.hour
LEFT JOIN cet_tara r
    ON UPPER(TRIM(f.citizenship_full)) = UPPER(TRIM(r.cet_full_name))
GROUP BY
    p.pctf, d.dt, day_of_week, h.hour
ORDER BY
    p.pctf, d.dt, h.hour;