DROP TABLE IF EXISTS binned_vitals CASCADE;
CREATE TABLE binned_vitals AS

WITH vitals AS (
    SELECT v.*,
        v.endtime - v.chartoffset AS timetoend,
        FLOOR((v.endtime - v.chartoffset) / 60.0) AS bucket
    FROM vitals_spo2 AS v
)

SELECT
    patientunitstayid,
    bucket,
    MIN(timetoend) AS timetoend,  -- for decay weight
    AVG(spo2) AS spo2,
    AVG(respiration) AS respiration,
    AVG(temperature) AS temperature,
    AVG(heartrate) AS heartrate
FROM vitals
GROUP BY patientunitstayid, bucket
