COPY (
    SELECT 
        oasis.stay_id,
        oasis.oasis,
        oasis.oasis_prob,
        oasis.age_score AS age_oasis,
        oasis.gcs_score AS gcs_oasis,
        oasis.mbp_score AS map_oasis,
        oasis.mechvent_score AS vent_oasis,
        oasis.temp_score AS temperature_oasis,
        oasis.heart_rate_score AS heartrate_oasis,
        oasis.preiculos_score AS pre_icu_los_oasis,
        oasis.urineoutput_score AS urineoutput_oasis,
        oasis.resp_rate_score AS respiratoryrate_oasis,
        oasis.electivesurgery_score AS electivesurgery_oasis
    FROM mimiciv_derived.oasis AS oasis
    INNER JOIN patient_first_vent as pat
    ON oasis.stay_id = pat.stay_id
    ORDER BY oasis.stay_id ASC
) TO 'insert_absolute_path_here' DELIMITER ',' CSV HEADER;