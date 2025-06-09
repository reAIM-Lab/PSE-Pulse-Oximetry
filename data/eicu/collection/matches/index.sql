DROP INDEX IF EXISTS vitals_idx;
CREATE INDEX vitals_idx
ON vitals_spo2 (patientunitstayid, chartoffset DESC)
WHERE spo2 IS NOT NULL;