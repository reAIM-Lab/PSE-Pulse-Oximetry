DROP TABLE IF EXISTS agg_abgs;
CREATE TABLE agg_abgs AS

WITH timetoend_abgs AS (
    SELECT abgs.*,
        abgs.endtime - abgs.chartoffset AS timetoend
    FROM abgs
),
weighted_abgs AS (
    SELECT abgs.*,
        CASE 
            WHEN timetoend > 10000 THEN 0
            ELSE EXP(-0.01 * timetoend)
        END AS w
    FROM timetoend_abgs AS abgs
),
agg AS (
    SELECT wa.patientunitstayid,
        -- Exponential weighted averages with fallback to NULL
        COALESCE(SUM(ph * w) / NULLIF(SUM(CASE WHEN ph IS NOT NULL THEN w ELSE 0 END), 0), NULL) AS ph_ewa,
        COALESCE(SUM(paco2 * w) / NULLIF(SUM(CASE WHEN paco2 IS NOT NULL THEN w ELSE 0 END), 0), NULL) AS paco2_ewa,
        COALESCE(SUM(pao2 * w) / NULLIF(SUM(CASE WHEN pao2 IS NOT NULL THEN w ELSE 0 END), 0), NULL) AS pao2_ewa,
        COALESCE(SUM(sao2 * w) / NULLIF(SUM(CASE WHEN sao2 IS NOT NULL THEN w ELSE 0 END), 0), NULL) AS sao2_ewa,
        COALESCE(SUM(carboxyhemoglobin * w) / NULLIF(SUM(CASE WHEN carboxyhemoglobin IS NOT NULL THEN w ELSE 0 END), 0), NULL) AS carboxyhemoglobin_ewa,
        COALESCE(SUM(methemoglobin * w) / NULLIF(SUM(CASE WHEN methemoglobin IS NOT NULL THEN w ELSE 0 END), 0), NULL) AS methemoglobin_ewa,
        COALESCE(SUM(lactate * w) / NULLIF(SUM(CASE WHEN lactate IS NOT NULL THEN w ELSE 0 END), 0), NULL) AS lactate_ewa,
        COALESCE(SUM(creatinine * w) / NULLIF(SUM(CASE WHEN creatinine IS NOT NULL THEN w ELSE 0 END), 0), NULL) AS creatinine_ewa,
        COALESCE(SUM(potassium * w) / NULLIF(SUM(CASE WHEN potassium IS NOT NULL THEN w ELSE 0 END), 0), NULL) AS potassium_ewa,
        COALESCE(SUM(sodium * w) / NULLIF(SUM(CASE WHEN sodium IS NOT NULL THEN w ELSE 0 END), 0), NULL) AS sodium_ewa,
        COALESCE(SUM(chloride * w) / NULLIF(SUM(CASE WHEN chloride IS NOT NULL THEN w ELSE 0 END), 0), NULL) AS chloride_ewa,
        COALESCE(SUM(glucose * w) / NULLIF(SUM(CASE WHEN glucose IS NOT NULL THEN w ELSE 0 END), 0), NULL) AS glucose_ewa,
        COALESCE(SUM(hct * w) / NULLIF(SUM(CASE WHEN hct IS NOT NULL THEN w ELSE 0 END), 0), NULL) AS hct_ewa,
        COALESCE(SUM(hgb * w) / NULLIF(SUM(CASE WHEN hgb IS NOT NULL THEN w ELSE 0 END), 0), NULL) AS hgb_ewa,
        COALESCE(SUM(bicarbonate * w) / NULLIF(SUM(CASE WHEN bicarbonate IS NOT NULL THEN w ELSE 0 END), 0), NULL) AS bicarbonate_ewa,
        COALESCE(SUM(co2 * w) / NULLIF(SUM(CASE WHEN co2 IS NOT NULL THEN w ELSE 0 END), 0), NULL) AS co2_ewa
    FROM weighted_abgs AS wa
    INNER JOIN patient_first_vent as pat
    ON pat.patientunitstayid = wa.patientunitstayid
    GROUP BY 
        wa.patientunitstayid
)

SELECT * 
FROM agg
WHERE sao2_ewa IS NOT NULL
ORDER BY agg.patientunitstayid ASC; 