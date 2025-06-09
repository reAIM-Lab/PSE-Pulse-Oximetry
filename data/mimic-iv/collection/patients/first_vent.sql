DROP TABLE IF EXISTS patient_first_vent CASCADE;
CREATE TABLE patient_first_vent AS

WITH patient_wvent AS (
  SELECT 
    icu.subject_id,
    icu.hadm_id,
    icu.stay_id,
    icu.intime,
    icu.outtime,
    icu.los,
    adm.race,
    pat.gender,
    pat.anchor_age + (EXTRACT(YEAR FROM icu.intime) - pat.anchor_year) AS age,
    vent.starttime AS vent_start,
    vent.endtime AS vent_end,
    CASE 
      WHEN vent.ventilation_status LIKE 'InvasiveVent' THEN 1 
      ELSE 0 
    END AS invasive_vent
  FROM mimiciv_icu.icustays AS icu
  LEFT JOIN mimiciv_hosp.admissions AS adm
    ON icu.subject_id = adm.subject_id AND icu.hadm_id = adm.hadm_id
  LEFT JOIN mimiciv_hosp.patients AS pat
    ON icu.subject_id = pat.subject_id
  INNER JOIN mimiciv_derived.ventilation AS vent
    ON icu.stay_id = vent.stay_id
  WHERE icu.los > 0.5
    AND (adm.deathtime IS NULL OR icu.outtime < adm.deathtime)
    AND pat.anchor_age >= 18
    AND adm.race IS NOT NULL
    AND icu.intime <= vent.starttime
    AND vent.endtime <= icu.outtime
),

patient_wvent_num AS (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY stay_id ORDER BY vent_start ASC) AS vent_num
  FROM patient_wvent
)

SELECT 
  subject_id,
  hadm_id,
  stay_id,
  intime AS start_time,
  outtime AS end_time,
  los,
  race,
  gender,
  age,
  vent_start,
  vent_end,
  EXTRACT(EPOCH FROM (vent_end - vent_start)) / 60 AS vent_duration,
  invasive_vent
FROM patient_wvent_num
WHERE vent_num = 1
ORDER BY subject_id ASC, hadm_id ASC, stay_id ASC, vent_start ASC;