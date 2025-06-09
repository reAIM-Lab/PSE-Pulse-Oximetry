DROP TABLE IF EXISTS filtered_labs CASCADE;
CREATE TABLE filtered_labs AS

SELECT
    lab.subject_id,
    lab.hadm_id,
    lab.charttime,
    dlab.label,
    lab.itemid,
    lab.value::FLOAT AS value
FROM mimiciv_hosp.labevents AS lab
JOIN mimiciv_hosp.d_labitems AS dlab
    ON lab.itemid = dlab.itemid
WHERE dlab.label IN (
    'pH',
    'pCO2',
    'pO2',
    'Oxygen Saturation',
    'Carboxyhemoglobin',
    'Methemoglobin',
    'Lactate',
    'Creatinine',
    'Potassium, Whole Blood',
    'Sodium, Whole Blood',
    'Chloride, Whole Blood',
    'Glucose',
    'Hematocrit, Calculated',
    'Hemoglobin',
    'Calculated Bicarbonate, Whole Blood',
    'Calculated Total CO2'
)
AND lab.value ~ '^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$'