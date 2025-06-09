COPY (
    SELECT 
        p.subject_id,
        p.hadm_id,
        p.stay_id,
        p.gender,
        p.age,
        p.race,
        p.start_time,
        p.end_time,
        p.vent_start,
        p.vent_end,
        p.vent_duration,
        p.invasive_vent,
        c.charlson_comorbidity_index as charlson
    FROM patient_first_vent as p
    LEFT JOIN mimiciv_derived.charlson AS c
        ON p.subject_id = c.subject_id
        AND p.hadm_id = c.hadm_id
) TO 'insert_absolute_path_here' DELIMITER ',' CSV HEADER;