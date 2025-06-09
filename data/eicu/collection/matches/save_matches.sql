COPY (
    SELECT *
    FROM agg_matches
    ORDER BY patientunitstayid ASC
) TO 'insert_absolute_path_here' DELIMITER ',' CSV HEADER;