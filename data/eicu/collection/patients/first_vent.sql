DROP TABLE IF EXISTS patient_first_vent CASCADE;
CREATE TABLE patient_first_vent AS

WITH patients_wvent AS (
    SELECT 
        p.patientunitstayid,
        p.gender,
        p.age,
        p.ethnicity,
        v.vent_start,
        v.vent_end,
        v.vent_duration,
        v.oxygen_therapy_type,
        -- Compute endtime based on oxygen_therapy_type
        CASE 
            WHEN v.oxygen_therapy_type = 4 THEN v.vent_end
            ELSE p.unitdischargeoffset
        END AS endtime,
        -- Compute vented flag
        CASE 
            WHEN v.oxygen_therapy_type = 4 THEN TRUE
            ELSE FALSE
        END AS vented

    FROM (
        SELECT * 
        FROM patient AS pt
        WHERE -- LOWER(pt.hospitaldischargestatus) LIKE '%alive%'
            pt.patientunitstayid IS NOT NULL
            AND pt.gender IS NOT NULL
            AND pt.age IS NOT NULL
            AND pt.ethnicity IS NOT NULL
    ) AS p
    LEFT JOIN (
        SELECT * 
        FROM ventilation
        WHERE oxygen_therapy_type IN (0, 1, 3, 4) 
            AND ventnum = 1
    ) AS v
    ON p.patientunitstayid = v.icustay_id
)

SELECT *
FROM patients_wvent
WHERE endtime >= 1440
ORDER BY patientunitstayid ASC;