DROP TABLE IF EXISTS ventilation CASCADE;
CREATE TABLE ventilation (
	icustay_id INT, 
	ventnum INT,
	vent_end INT,
	vent_start INT,
    oxygen_therapy_type INT,
	supp_oxygen BOOLEAN, 
	vent_duration FLOAT,
    vent_start_first INT
);
\COPY ventilation(icustay_id, ventnum, vent_end, vent_start, oxygen_therapy_type, supp_oxygen, vent_duration, vent_start_first) FROM './collection/patients/ventilation.csv' DELIMITER ',' CSV HEADER;