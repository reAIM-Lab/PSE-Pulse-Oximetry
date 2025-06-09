DROP TABLE IF EXISTS vitals_spo2 CASCADE;
CREATE TABLE vitals_spo2 AS

SELECT
    vp.patientunitstayid,
    vp.observationoffset AS chartoffset,
    pat.endtime,
    CASE 
        WHEN vp.sao2 BETWEEN 70 AND 100 THEN vp.sao2
        ELSE NULL
    END AS spo2, -- really spo2, since it's taken with the vitals
    vp.respiration,
    CASE
        WHEN vp.temperature < 45 THEN vp.temperature * 9.0 / 5 + 32
        ELSE vp.temperature
    END AS temperature, -- convert C -> F
    vp.heartRate AS heartrate
FROM vitalperiodic AS vp
INNER JOIN patient_first_vent AS pat
    ON vp.patientunitstayid = pat.patientunitstayid
WHERE
    vp.observationoffset >= 0 AND
    vp.observationoffset <= pat.endtime
    AND NOT (
        vp.sao2 IS NULL AND
        vp.respiration IS NULL AND
        vp.heartRate IS NULL
    )