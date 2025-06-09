DROP TABLE IF EXISTS agg_vitals;
CREATE TABLE agg_vitals AS

WITH timetoend_vitals AS (
    SELECT 
        vitals.*,
        EXTRACT(EPOCH FROM (vitals.end_time - vitals.chart_time)) / 60 AS timetoend
    FROM vitals_spo2 AS vitals
),
weighted_vitals AS (
    SELECT vitals.*,
        CASE 
            WHEN timetoend > 10000 THEN 0
            ELSE EXP(-0.01 * timetoend)
        END AS w
    FROM timetoend_vitals AS vitals
),
agg AS (
    SELECT wv.stay_id,
        -- Exponential weighted averages with fallback to NULL
        COALESCE(SUM(spo2 * w) / NULLIF(SUM(CASE WHEN spo2 IS NOT NULL THEN w ELSE 0 END), 0), NULL) AS spo2_ewa,
        COALESCE(SUM(respiration * w) / NULLIF(SUM(CASE WHEN respiration IS NOT NULL THEN w ELSE 0 END), 0), NULL) AS respiration_ewa,
        COALESCE(SUM(temperature * w) / NULLIF(SUM(CASE WHEN temperature IS NOT NULL THEN w ELSE 0 END), 0), NULL) AS temperature_ewa,
        COALESCE(SUM(heartrate * w) / NULLIF(SUM(CASE WHEN heartrate IS NOT NULL THEN w ELSE 0 END), 0), NULL) AS heartrate_ewa
    FROM weighted_vitals AS wv
    INNER JOIN patient_first_vent as pat
    ON pat.stay_id = wv.stay_id
    GROUP BY 
        wv.stay_id
)

SELECT * 
FROM agg
WHERE spo2_ewa IS NOT NULL
ORDER BY 
    agg.stay_id ASC; 