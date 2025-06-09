DROP TABLE IF EXISTS vitals_spo2 CASCADE;
CREATE TABLE vitals_spo2 AS

WITH raw_vitals AS (
    SELECT * FROM (
        SELECT
            chart.subject_id,
            chart.hadm_id,
            chart.stay_id,
            chart.charttime AS chart_time,
            AVG(CASE 
                WHEN items.label = 'O2 saturation pulseoxymetry' AND chart.value BETWEEN 70 AND 100 
                THEN chart.value 
                ELSE NULL 
            END) AS spo2,
            AVG(CASE WHEN items.label = 'Respiratory Rate (Total)' THEN chart.value ELSE NULL END) AS respiration,
            AVG(CASE
                    WHEN items.label IN ('Temperature Fahrenheit', 'Temperature Celsius') AND chart.value IS NOT NULL
                    THEN
                        CASE
                        WHEN items.label = 'Temperature Fahrenheit' THEN (chart.value - 32) * 5/9
                        WHEN items.label = 'Temperature Celsius' THEN chart.value
                        END
                    ELSE NULL
            END) AS temperature,
            AVG(CASE WHEN items.label = 'Heart Rate' THEN chart.value ELSE NULL END) AS heartrate
        FROM filtered_charts AS chart
        INNER JOIN mimiciv_icu.d_items AS items
        ON chart.itemid = items.itemid
        GROUP BY
            chart.subject_id,
            chart.hadm_id,
            chart.stay_id,
            chart.charttime
    ) AS measurements
    WHERE NOT (
        spo2 IS NULL AND
        respiration IS NULL AND
        temperature IS NULL AND
        heartrate IS NULL
    )
)
SELECT 
    pat.stay_id,
    pat.start_time,
    pat.end_time,
    raw_vitals.chart_time,
    raw_vitals.spo2,
    raw_vitals.respiration,
    raw_vitals.temperature,
    raw_vitals.heartrate
FROM raw_vitals
INNER JOIN patient_first_vent AS pat
    ON raw_vitals.subject_id = pat.subject_id AND
        raw_vitals.hadm_id = pat.hadm_id AND
        raw_vitals.stay_id = pat.stay_id
WHERE
    raw_vitals.chart_time >= pat.start_time AND
    raw_vitals.chart_time <= pat.end_time
ORDER BY 
    pat.stay_id,
    raw_vitals.chart_time ASC;