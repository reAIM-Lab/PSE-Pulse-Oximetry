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
races = patients['race'].unique()
race_dict = {}
for r in races:
    if 'BLACK' in r or 'AFRICAN AMERICAN' in r:
        race_dict[r] = 0
    elif 'ASIAN' in r:
        race_dict[r] = 1
    elif 'WHITE' in r:
        race_dict[r] = 2
    elif 'HISPANIC' in r or 'LATINO' in r:
        race_dict[r] = 3
races = list(race_dict.keys())
patients = patients[patients['race'].isin(races)].copy()
patients.loc[:, 'race'] = patients['race'].map(race_dict)
patients.loc[:, 'gender'] = patients['gender'].map({'F': 0, 'M': 1})
patients = patients.drop(columns=['subject_id', 'hadm_id'])
cohort = patients.merge(oasis, how='inner', on=['stay_id'])
print('Total stays with OASIS:', len(cohort))

##### PART 2 ######
cohort = cohort.merge(measurements, how='inner', on=['stay_id'])
cohort = cohort.loc[(~cohort['sao2_ewa'].isna()) & (~cohort['spo2_ewa'].isna())]
print('Total stays with OASIS + SaO2/SpO2:', len(cohort))

##### PART 3 ######
cohort = cohort.merge(matches, how='inner', on=['stay_id'])
cohort = cohort.loc[(~cohort['discrepancy_ewa'].isna())]
assert len(cohort) == len(set(cohort['stay_id']))
cohort = cohort.drop(columns=['stay_id', 'oasis_prob'])
print('Total stays with OASIS + SaO2/SpO2 + Δ:', len(cohort))

##### PART 4 ######
cohort['start_time'] = pd.to_datetime(cohort['start_time'])
cohort['end_time'] = pd.to_datetime(cohort['end_time'])
cohort['vent_start'] = pd.to_datetime(cohort['vent_start'])
cohort['vent_end'] = pd.to_datetime(cohort['vent_end'])
cohort['vent_start'] = (cohort['vent_start'] - cohort['start_time']).dt.total_seconds() // 60
cohort['vent_end'] = (cohort['vent_end'] - cohort['start_time']).dt.total_seconds() // 60
cohort = cohort.drop(columns=['start_time', 'end_time'])
cohort = cohort.rename(columns={'invasive_vent': 'vented'})
print('Columns:', list(cohort.columns))
print()

##### PRINT RESULTS #####
races = ['African American', 'Asian', 'Caucasian', 'Hispanic']
for i,r in enumerate(races):
    df = cohort[cohort['race'] == i]
    df_y1 = df[df['vented'] == 1]
    print(f'***** {r} *****')
    print('Total patients', len(df))
    print('Total vented patients', len(df_y1))
    print('Rate of ventilation', np.round(len(df_y1) / len(df), 3))
    print('Avg duration of ventilation', np.round(np.mean(df_y1['vent_duration']), 3))
    print()

cohort = cohort.astype(float)
cohort.to_csv('cohort.csv', index=False)
print('Saved to cohort.csv')