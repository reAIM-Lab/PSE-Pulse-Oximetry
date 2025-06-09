import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PATIENTS_PATH = './collection/patients.csv'
MEASUREMENTS_PATH = './collection/measurements.csv'
MATCHES_PATH = './collection/matches.csv'
OASIS_PATH = './collection/oasis.csv'

patients = pd.read_csv(PATIENTS_PATH)
measurements = pd.read_csv(MEASUREMENTS_PATH)
matches = pd.read_csv(MATCHES_PATH)
oasis = pd.read_csv(OASIS_PATH)

##### PART 1 ######
patients = patients[
    (patients['ethnicity'] == 'African American') |
    (patients['ethnicity'] == 'Asian') |
    (patients['ethnicity'] == 'Caucasian') |
    (patients['ethnicity'] == 'Hispanic')
].copy()
patients.loc[:, 'ethnicity'] = patients['ethnicity'].map(
    {'African American': 0, 'Asian': 1, 'Caucasian': 2, 'Hispanic': 3}
)
patients.loc[:, 'age'] = patients['age'].replace({'> 89': 90})
patients.loc[:, 'gender'] = patients['gender'].map({'Female': 0, 'Male': 1})
patients.loc[:, 'vented'] = patients['vented'].map({'f': 0, 't': 1})
cohort = patients.merge(oasis, how='inner', on=['patientunitstayid'])
print(cohort.head())
print('Total stays with OASIS:', len(cohort))

##### PART 2 ######
cohort = cohort.merge(measurements, how='inner', on=['patientunitstayid'])
cohort = cohort.loc[(~cohort['sao2_ewa'].isna()) & (~cohort['spo2_ewa'].isna())]
print('Total stays with OASIS + SaO2/SpO2:', len(cohort))

##### PART 3 ######
cohort = cohort.merge(matches, how='inner', on=['patientunitstayid'])
cohort = cohort.loc[(~cohort['discrepancy_ewa'].isna())]
assert len(cohort) == len(set(cohort['patientunitstayid']))
cohort = cohort.drop(columns=['patientunitstayid', 'oxygen_therapy_type', 'oasis_prob'])
print('Total stays with OASIS + SaO2/SpO2 + Δ:', len(cohort))
print('Columns:', list(cohort.columns))
print()

##### PRINT RESULTS #####
ethnicities = ['African American', 'Asian', 'Caucasian', 'Hispanic']
for i,e in enumerate(ethnicities):
    df = cohort[cohort['ethnicity'] == i]
    df_y1 = df[df['vented'] == 1]
    print(f'***** {e} *****')
    print('Total patients', len(df))
    print('Total vented patients', len(df_y1))
    print('Rate of ventilation', np.round(len(df_y1) / len(df), 3))
    print('Avg duration of ventilation', np.round(np.mean(df_y1['vent_duration']), 3))
    print()

cohort = cohort.astype(float)
cohort.to_csv('cohort.csv', index=False)