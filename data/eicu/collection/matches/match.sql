DROP TABLE IF EXISTS matches CASCADE;
CREATE TABLE matches AS

SELECT 
    abgs.patientunitstayid,
    abgs.sao2,
    vitals_matched.spo2,
    abgs.chartoffset AS sao2_offset,
    vitals_matched.chartoffset AS spo2_offset,
    abgs.chartoffset - vitals_matched.chartoffset AS deltatime
FROM (
    SELECT 
        abgs.patientunitstayid,
        abgs.chartoffset,
        abgs.sao2
    FROM abgs
    WHERE abgs.sao2 IS NOT NULL
) AS abgs

LEFT JOIN LATERAL (
    SELECT 
        vitals.patientunitstayid,
        vitals.chartoffset,
        vitals.spo2
    FROM vitals_spo2 AS vitals
    WHERE 
        abgs.patientunitstayid = vitals.patientunitstayid
        AND vitals.chartoffset <= abgs.chartoffset
        AND vitals.spo2 IS NOT NULL
    ORDER BY vitals.chartoffset DESC
    LIMIT 1
) AS vitals_matched on TRUE
ORDER BY
    abgs.patientunitstayid ASC,
    abgs.chartoffset ASC,
    vitals_matched.chartoffset ASC;