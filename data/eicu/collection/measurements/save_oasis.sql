COPY (
    SELECT oasis.* 
    FROM pivoted_oasis AS oasis
    INNER JOIN patient_first_vent as pat
    ON oasis.patientunitstayid = pat.patientunitstayid
    ORDER BY oasis.patientunitstayid ASC
) TO 'insert_absolute_path_here' DELIMITER ',' CSV HEADER;