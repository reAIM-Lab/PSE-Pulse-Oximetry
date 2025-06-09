#!/bin/bash

DB_NAME="eicu"
DB_USER="kyz2005"

# patients + ventilation
echo "PATIENTS"
psql -U "$DB_USER" -d "$DB_NAME" -f ./collection/patients/ventilation.sql
psql -U "$DB_USER" -d "$DB_NAME" -f ./collection/patients/charlson.sql
psql -U "$DB_USER" -d "$DB_NAME" -f ./collection/patients/first_vent.sql
psql -U "$DB_USER" -d "$DB_NAME" -f ./collection/patients/save_patients.sql

# measurements
echo
echo "MEASUREMENTS"
psql -U "$DB_USER" -d "$DB_NAME" -f ./collection/measurements/abgs.sql
psql -U "$DB_USER" -d "$DB_NAME" -f ./collection/measurements/agg_abgs.sql
psql -U "$DB_USER" -d "$DB_NAME" -f ./collection/measurements/vitals_spo2.sql
psql -U "$DB_USER" -d "$DB_NAME" -f ./collection/measurements/bin_vitals.sql
psql -U "$DB_USER" -d "$DB_NAME" -f ./collection/measurements/agg_vitals.sql
psql -U "$DB_USER" -d "$DB_NAME" -f ./collection/measurements/save_measurements.sql
psql -U "$DB_USER" -d "$DB_NAME" -f ./collection/measurements/save_oasis.sql

# matches
echo
echo "MATCHES"
psql -U "$DB_USER" -d "$DB_NAME" -f ./collection/matches/index.sql
psql -U "$DB_USER" -d "$DB_NAME" -f ./collection/matches/match.sql
psql -U "$DB_USER" -d "$DB_NAME" -f ./collection/matches/agg_matches.sql
psql -U "$DB_USER" -d "$DB_NAME" -f ./collection/matches/save_matches.sql

echo "All queries executed."