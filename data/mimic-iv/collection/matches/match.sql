DROP TABLE IF EXISTS matches CASCADE;
CREATE TABLE matches AS

SELECT 
    abgs.stay_id,
    abgs.sao2,
    vitals_matched.spo2,
    abgs.chart_time AS sao2_time,
    vitals_matched.chart_time AS spo2_time,
    EXTRACT(EPOCH FROM (abgs.chart_time - vitals_matched.chart_time)) / 60 AS deltatime
FROM (
    SELECT 
        abgs.stay_id,
        abgs.chart_time,
        abgs.sao2
    FROM abgs
    WHERE abgs.sao2 IS NOT NULL
) AS abgs

LEFT JOIN LATERAL (
    SELECT 
        vitals.stay_id,
        vitals.chart_time,
        vitals.spo2
    FROM vitals_spo2 AS vitals
    WHERE 
        abgs.stay_id = vitals.stay_id
        AND vitals.chart_time <= abgs.chart_time
        AND vitals.spo2 IS NOT NULL
    ORDER BY vitals.chart_time DESC
    LIMIT 1
) AS vitals_matched on TRUE
ORDER BY
    abgs.stay_id ASC,
    abgs.chart_time ASC,
    vitals_matched.chart_time ASC;