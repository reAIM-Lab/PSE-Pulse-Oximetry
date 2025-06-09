DROP TABLE IF EXISTS charlson CASCADE;
CREATE TABLE charlson (
    patientunitstayid INT,
    mets6 INT, 
    aids6 INT,
    liver3 INT,
    stroke2 INT,
    renal2 INT,
    dm INT,
    cancer2 INT,
    leukemia2 INT,
    lymphoma2 INT,
    mi1 INT,
    chf1 INT,
    pvd1 INT,
    tia1 INT,
    dementia1 INT,
    copd1 INT,
    ctd1 INT,
    pud1 INT,
    liver1 INT,
    age_score_charlson INT,
    final_charlson_score INT
);
\COPY charlson(patientunitstayid, mets6, aids6, liver3, stroke2, renal2, dm, cancer2, leukemia2, lymphoma2, mi1, chf1, pvd1, tia1, dementia1, copd1, ctd1, pud1, liver1, age_score_charlson, final_charlson_score) FROM './collection/patients/charlson.csv' DELIMITER ',' CSV HEADER;