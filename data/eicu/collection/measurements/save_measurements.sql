COPY (
    SELECT abgs.*,
        vitals.spo2_ewa,
        vitals.respiration_ewa,
        vitals.temperature_ewa,
        vitals.heartrate_ewa
    FROM agg_abgs AS abgs
    INNER JOIN agg_vitals as vitals
    ON abgs.patientunitstayid = vitals.patientunitstayid
    WHERE
        abgs.sao2_ewa IS NOT NULL
        AND vitals.spo2_ewa IS NOT NULL
    ORDER BY abgs.patientunitstayid ASC
) TO 'insert_absolute_path_here' DELIMITER ',' CSV HEADER;