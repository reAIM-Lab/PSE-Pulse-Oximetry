DROP TABLE IF EXISTS filtered_charts CASCADE;
CREATE TABLE filtered_charts AS

SELECT
    chart.subject_id,
    chart.hadm_id,
    chart.stay_id,
    chart.charttime,
    items.label,
    chart.itemid,
    chart.value::FLOAT AS value
FROM mimiciv_icu.chartevents AS chart
JOIN mimiciv_icu.d_items AS items
    ON chart.itemid = items.itemid
WHERE items.label IN (
    'O2 saturation pulseoxymetry',
    'Respiratory Rate (Total)',
    'Temperature Fahrenheit',
    'Temperature Celsius',
    'Heart Rate'
)
AND chart.value ~ '^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$'