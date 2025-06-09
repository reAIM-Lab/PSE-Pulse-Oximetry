COPY (
    SELECT 
        p.patientunitstayid,
        p.gender,
        p.age,
        p.ethnicity,
        c.final_charlson_score as charlson,
        p.endtime,
        p.vented,
        p.vent_start,
        p.vent_end,
        p.vent_duration,
        p.oxygen_therapy_type
    FROM patient_first_vent as p
    LEFT JOIN charlson AS c
    ON p.patientunitstayid = c.patientunitstayid
) TO 'insert_absolute_path_here' DELIMITER ',' CSV HEADER;