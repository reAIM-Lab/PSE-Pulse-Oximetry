COPY (
    SELECT *
    FROM agg_matches
    ORDER BY stay_id ASC
) TO 'insert_absolute_path_here' DELIMITER ',' CSV HEADER;