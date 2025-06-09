DROP TABLE IF EXISTS agg_matches;
CREATE TABLE agg_matches AS

WITH timetoend_matches AS (
    SELECT matches.stay_id,
        sao2_time AS chart_time,
        pat.end_time,
        EXTRACT(EPOCH FROM (pat.end_time - sao2_time)) / 60 AS timetoend,
        spo2 - sao2 AS discrepancy
    FROM matches
    INNER JOIN patient_first_vent AS pat
    ON pat.stay_id = matches.stay_id
    WHERE 
        sao2_time >= pat.start_time AND
        matches.deltatime <= 5 AND   -- strict
        -- matches.deltatime <= 90 AND   -- lenient
        sao2_time <= pat.end_time
),
weighted_matches AS (
    SELECT matches.*,
        CASE 
            WHEN timetoend > 10000 THEN 0
            ELSE EXP(-0.01 * timetoend)
        END AS w
    FROM timetoend_matches AS matches
),
agg AS (
    SELECT wm.stay_id,
        -- Exponential weighted averages with fallback to NULL
        COALESCE(SUM(discrepancy * w) / NULLIF(SUM(CASE WHEN discrepancy IS NOT NULL THEN w ELSE 0 END), 0), NULL) AS discrepancy_ewa
    FROM weighted_matches AS wm
    INNER JOIN patient_first_vent AS pat
    ON pat.stay_id = wm.stay_id
    GROUP BY 
        wm.stay_id
)

SELECT * 
FROM agg
WHERE discrepancy_ewa IS NOT NULL
ORDER BY agg.stay_id ASC;