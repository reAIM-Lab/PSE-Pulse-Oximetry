DROP TABLE IF EXISTS abgs CASCADE;
CREATE TABLE abgs AS

SELECT * FROM (
    SELECT 
        lab.patientunitstayid,
        labresultoffset AS chartoffset,
        pat.endtime,
        -- take average of measurements for each timestamp
        AVG(CASE WHEN labname = 'pH' THEN labresult ELSE NULL END) AS ph,
        AVG(CASE WHEN labname = 'paCO2' THEN labresult ELSE NULL END) AS paco2,
        AVG(CASE WHEN labname = 'paO2' THEN labresult ELSE NULL END) AS pao2,
        -- AVG(CASE WHEN labname = 'O2 Sat (%)' THEN labresult ELSE NULL END) AS sao2,
        AVG(CASE 
                WHEN labname = 'O2 Sat (%)' AND labresult BETWEEN 70 AND 100 THEN labresult 
                ELSE NULL 
            END) AS sao2,
        AVG(CASE WHEN labname = 'Carboxyhemoglobin' THEN labresult ELSE NULL END) AS carboxyhemoglobin,
        AVG(CASE WHEN labname = 'Methemoglobin' THEN labresult ELSE NULL END) AS methemoglobin,
        AVG(CASE WHEN labname = 'lactate' THEN labresult ELSE NULL END) AS lactate,
        AVG(CASE WHEN labname = 'creatinine' THEN labresult ELSE NULL END) AS creatinine,
        AVG(CASE WHEN labname = 'potassium' THEN labresult ELSE NULL END) AS potassium,
        AVG(CASE WHEN labname = 'sodium' THEN labresult ELSE NULL END) AS sodium,
        AVG(CASE WHEN labname = 'chloride' THEN labresult ELSE NULL END) AS chloride,
        AVG(CASE WHEN labname = 'glucose' THEN labresult ELSE NULL END) AS glucose,
        AVG(CASE WHEN labname = 'Hct' THEN labresult ELSE NULL END) AS hct,
        AVG(CASE WHEN labname = 'Hgb' THEN labresult ELSE NULL END) AS hgb,
        AVG(CASE WHEN labname = 'bicarbonate' THEN labresult ELSE NULL END) AS bicarbonate,
        AVG(CASE WHEN labname = 'Total CO2' THEN labresult ELSE NULL END) AS co2
    FROM lab
    INNER JOIN patient_first_vent AS pat
        ON pat.patientunitstayid = lab.patientunitstayid
    WHERE 
        lab.labresultoffset >= 0 AND
        lab.labresultoffset <= pat.endtime
    GROUP BY 
        lab.patientunitstayid,
        lab.labresultoffset,
        pat.endtime
) AS abgs
WHERE 
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
ORDER BY 
    patientunitstayid ASC,
    chartoffset ASC;