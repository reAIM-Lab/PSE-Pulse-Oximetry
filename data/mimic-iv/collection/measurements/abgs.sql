DROP TABLE IF EXISTS abgs CASCADE;
CREATE TABLE abgs AS

WITH raw_abgs AS (
    SELECT * FROM (
        SELECT 
            subject_id,
            hadm_id,
            charttime AS chart_time,
            AVG(CASE WHEN label = 'pH' THEN value ELSE NULL END) AS ph,
            AVG(CASE WHEN label = 'pCO2' THEN value ELSE NULL END) AS paco2,
            AVG(CASE WHEN label = 'pO2' THEN value ELSE NULL END) AS pao2,
            AVG(CASE 
                WHEN label = 'Oxygen Saturation' AND value BETWEEN 70 AND 100 
                THEN value 
                ELSE NULL 
            END) AS sao2,
            AVG(CASE WHEN label = 'Carboxyhemoglobin' THEN value ELSE NULL END) AS carboxyhemoglobin,
            AVG(CASE WHEN label = 'Methemoglobin' THEN value ELSE NULL END) AS methemoglobin,
            AVG(CASE WHEN label = 'Lactate' THEN value ELSE NULL END) AS lactate,
            AVG(CASE WHEN label = 'Creatinine' THEN value ELSE NULL END) AS creatinine,
            AVG(CASE WHEN label = 'Potassium, Whole Blood' THEN value ELSE NULL END) AS potassium,
            AVG(CASE WHEN label = 'Sodium, Whole Blood' THEN value ELSE NULL END) AS sodium,
            AVG(CASE WHEN label = 'Chloride, Whole Blood' THEN value ELSE NULL END) AS chloride,
            AVG(CASE WHEN label = 'Glucose' THEN value ELSE NULL END) AS glucose,
            AVG(CASE WHEN label = 'Hematocrit, Calculated' THEN value ELSE NULL END) AS hct,
            AVG(CASE WHEN label = 'Hemoglobin' THEN value ELSE NULL END) AS hgb,
            AVG(CASE WHEN label = 'Calculated Bicarbonate, Whole Blood' THEN value ELSE NULL END) AS bicarbonate,
            AVG(CASE WHEN label = 'Calculated Total CO2' THEN value ELSE NULL END) AS co2
        FROM filtered_labs
        GROUP BY subject_id, hadm_id, charttime
    ) AS measurements
    WHERE 
        subject_id IS NOT NULL AND
        hadm_id IS NOT NULL AND
        NOT (
            ph IS NULL AND
            paco2 IS NULL AND
            pao2 IS NULL AND
            sao2 IS NULL AND
            carboxyhemoglobin IS NULL AND
            methemoglobin IS NULL AND
            lactate IS NULL AND
            creatinine IS NULL AND
            potassium IS NULL AND
            sodium IS NULL AND
            chloride IS NULL AND
            glucose IS NULL AND
            hct IS NULL AND
            hgb IS NULL AND
            bicarbonate IS NULL AND
            co2 IS NULL
        )
)
SELECT 
    pat.stay_id,
    pat.start_time,
    pat.end_time,
    raw_abgs.chart_time,
    raw_abgs.ph,
    raw_abgs.paco2,
    raw_abgs.pao2,
    raw_abgs.sao2,
    raw_abgs.carboxyhemoglobin,
    raw_abgs.methemoglobin,
    raw_abgs.lactate,
    raw_abgs.creatinine,
    raw_abgs.potassium,
    raw_abgs.sodium,
    raw_abgs.chloride,
    raw_abgs.glucose,
    raw_abgs.hct,
    raw_abgs.hgb,
    raw_abgs.bicarbonate,
    raw_abgs.co2
FROM raw_abgs
INNER JOIN patient_first_vent AS pat
    ON raw_abgs.subject_id = pat.subject_id AND
        raw_abgs.hadm_id = pat.hadm_id
WHERE
    raw_abgs.chart_time >= pat.start_time AND
    raw_abgs.chart_time <= pat.end_time
ORDER BY 
    pat.stay_id,
    raw_abgs.chart_time ASC;