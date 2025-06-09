DROP TABLE IF EXISTS agg_matches;
CREATE TABLE agg_matches AS

WITH timetoend_matches AS (
    SELECT matches.patientunitstayid,
        sao2_offset AS chartoffset,
        pat.endtime,
        pat.endtime - sao2_offset AS timetoend,
        spo2 - sao2 AS discrepancy
    FROM matches
    INNER JOIN patient_first_vent AS pat
    ON pat.patientunitstayid = matches.patientunitstayid
    WHERE 
        sao2_offset >= 0 AND
        matches.deltatime <= 5 AND   -- strict
        -- matches.deltatime <= 90 AND   -- lenient
        sao2_offset <= pat.endtime
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
    SELECT wm.patientunitstayid,
        -- Exponential weighted averages with fallback to NULL
        COALESCE(SUM(discrepancy * w) / NULLIF(SUM(CASE WHEN discrepancy IS NOT NULL THEN w ELSE 0 END), 0), NULL) AS discrepancy_ewa
    FROM weighted_matches AS wm
    INNER JOIN patient_first_vent AS pat
    ON pat.patientunitstayid = wm.patientunitstayid
    GROUP BY 
        wm.patientunitstayid
)

SELECT * 
FROM agg
WHERE discrepancy_ewa IS NOT NULL
ORDER BY agg.patientunitstayid ASC;