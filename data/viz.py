import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

fontsize = 24
titlesize = 24
labelsize = 16
markersize = 10
mpl.rcParams.update({
    "font.size": fontsize,
    "axes.labelsize": labelsize,
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "font.weight": "bold",
    "xtick.labelsize": labelsize,
    "ytick.labelsize": labelsize,
    "axes.spines.top": False,
    "axes.spines.right": False
})
plt.rc('font', family='serif')

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

age_bins = [17, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, float('inf')]
age_labels = list(range(len(age_bins) - 1))

make_dir('./viz/')
eicu_df = pd.read_csv('eicu/cohort.csv')
if 'ethnicity' in eicu_df.columns:
    eicu_df = eicu_df.rename(columns={'ethnicity': 'race'})
eicu_df = eicu_df[(eicu_df['spo2_ewa'] >= 70) & (eicu_df['spo2_ewa'] <= 100)]
eicu_df = eicu_df[(eicu_df['sao2_ewa'] >= 70) & (eicu_df['sao2_ewa'] <= 100)]
eicu_df = eicu_df[eicu_df['age'] >= 18]
for c in eicu_df.columns:
    if c.endswith('_ewa'):
        eicu_df.rename(columns={c: c[:-4]}, inplace=True)
eicu_df['age_group'] = pd.cut(
    eicu_df['age'], bins=age_bins, labels=age_labels, right=True
).astype(int)
print('eICU samples:', len(eicu_df))

mimic_df = pd.read_csv('mimic-iv/cohort.csv')
mimic_df['vent_duration'] /= 60
if 'ethnicity' in mimic_df.columns:
    mimic_df = mimic_df.rename(columns={'ethnicity': 'race'})
mimic_df = mimic_df[(mimic_df['spo2_ewa'] >= 70) & (mimic_df['spo2_ewa'] <= 100)]
mimic_df = mimic_df[(mimic_df['sao2_ewa'] >= 70) & (mimic_df['sao2_ewa'] <= 100)]
for c in mimic_df.columns:
    if c.endswith('_ewa'):
        mimic_df.rename(columns={c: c[:-4]}, inplace=True)
mimic_df['age_group'] = pd.cut(
    mimic_df['age'], bins=age_bins, labels=age_labels, right=True
).astype(int)
print('MIMIC-IV samples:', len(mimic_df))

combined_df = pd.concat([eicu_df, mimic_df], axis=0)
print('Total samples:', len(combined_df))

# MIMIC
black_novent_mimic = len(mimic_df[(mimic_df['race'] == 0) & (mimic_df['vented'] == 0)])
asian_novent_mimic = len(mimic_df[(mimic_df['race'] == 1) & (mimic_df['vented'] == 0)])
white_novent_mimic = len(mimic_df[(mimic_df['race'] == 2) & (mimic_df['vented'] == 0)])
hispanic_novent_mimic = len(mimic_df[(mimic_df['race'] == 3) & (mimic_df['vented'] == 0)])
black_wvent_mimic = len(mimic_df[(mimic_df['race'] == 0) & (mimic_df['vented'] == 1)])
asian_wvent_mimic = len(mimic_df[(mimic_df['race'] == 1) & (mimic_df['vented'] == 1)])
white_wvent_mimic = len(mimic_df[(mimic_df['race'] == 2) & (mimic_df['vented'] == 1)])
hispanic_wvent_mimic = len(mimic_df[(mimic_df['race'] == 3) & (mimic_df['vented'] == 1)])
# eICU
black_novent_eicu = len(eicu_df[(eicu_df['race'] == 0) & (eicu_df['vented'] == 0)])
asian_novent_eicu = len(eicu_df[(eicu_df['race'] == 1) & (eicu_df['vented'] == 0)])
white_novent_eicu = len(eicu_df[(eicu_df['race'] == 2) & (eicu_df['vented'] == 0)])
hispanic_novent_eicu = len(eicu_df[(eicu_df['race'] == 3) & (eicu_df['vented'] == 0)])
black_wvent_eicu = len(eicu_df[(eicu_df['race'] == 0) & (eicu_df['vented'] == 1)])
asian_wvent_eicu = len(eicu_df[(eicu_df['race'] == 1) & (eicu_df['vented'] == 1)])
white_wvent_eicu = len(eicu_df[(eicu_df['race'] == 2) & (eicu_df['vented'] == 1)])
hispanic_wvent_eicu = len(eicu_df[(eicu_df['race'] == 3) & (eicu_df['vented'] == 1)])

races = ['White', 'Black', 'Hispanic', 'Asian']
x = np.arange(len(races))
width = 0.3

# Reordered data: [White, Black, Hispanic, Asian]
mimic_novent = [white_novent_mimic, black_novent_mimic, hispanic_novent_mimic, asian_novent_mimic]
mimic_wvent = [white_wvent_mimic, black_wvent_mimic, hispanic_wvent_mimic, asian_wvent_mimic]
eicu_novent = [white_novent_eicu, black_novent_eicu, hispanic_novent_eicu, asian_novent_eicu]
eicu_wvent = [white_wvent_eicu, black_wvent_eicu, hispanic_wvent_eicu, asian_wvent_eicu]

fig, ax = plt.subplots(figsize=(6, 6))

# Plot eICU on the left
bar1 = ax.bar(x - width/2, eicu_novent, width, label='eICU No Vent', color='lightblue')
bar1_top = ax.bar(x - width/2, eicu_wvent, width,
                  bottom=eicu_novent, label='eICU With Vent',
                  color='lightblue', hatch='//')

# Plot MIMIC on the right
bar2 = ax.bar(x + width/2, mimic_novent, width, label='MIMIC No Vent', color='lightcoral')
bar2_top = ax.bar(x + width/2, mimic_wvent, width,
                  bottom=mimic_novent, label='MIMIC With Vent',
                  color='lightcoral', hatch='//')

# Horizontal separators
for i in range(len(races)):
    # eICU
    x_eicu = x[i] - width/2
    y_eicu = eicu_novent[i]
    ax.hlines(y=y_eicu, xmin=x_eicu - width/2, xmax=x_eicu + width/2, colors='black', linewidth=1)

    # MIMIC
    x_mimic = x[i] + width/2
    y_mimic = mimic_novent[i]
    ax.hlines(y=y_mimic, xmin=x_mimic - width/2, xmax=x_mimic + width/2, colors='black', linewidth=1)

# Labels and style
ax.set_xticks(x)
ax.set_xticklabels(races)
ax.set_ylabel('Number of Patients')
ax.set_xlabel('Race')
ax.tick_params(axis='y')
ax.legend(fontsize=labelsize)

plt.tight_layout()
plt.yscale('log')
plt.savefig('viz/samples.jpg', dpi=300, bbox_inches='tight')

#################################################################

black_mimic = mimic_df[(mimic_df['race'] == 0)]
asian_mimic = mimic_df[(mimic_df['race'] == 1)]
white_mimic = mimic_df[(mimic_df['race'] == 2)]
hispanic_mimic = mimic_df[(mimic_df['race'] == 3)]
black_eicu = eicu_df[(eicu_df['race'] == 0)]
asian_eicu = eicu_df[(eicu_df['race'] == 1)]
white_eicu = eicu_df[(eicu_df['race'] == 2)]
hispanic_eicu = eicu_df[(eicu_df['race'] == 3)]
labels = ['White', 'Black', 'Hispanic', 'Asian']
eicu_dfs = [white_eicu, black_eicu, hispanic_eicu, asian_eicu]
mimic_dfs = [white_mimic, black_mimic, hispanic_mimic, asian_mimic]
cutoff = combined_df['vent_duration'].max()

fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
for i in range(2):
    if i == 0:
        dfs = eicu_dfs
    else:
        dfs = mimic_dfs
    for j in range(len(labels)):
        sns.kdeplot(
            dfs[j], x='vent_duration', ax=ax[i], log_scale=True, linewidth=3,
            label=labels[j], clip=(0, cutoff), bw_adjust=1.5,
        )
ax[0].set_ylabel('eICU\nDensity')
ax[1].set_ylabel('MIMIC-IV\nDensity')
ax[1].legend(fontsize=labelsize)
ax[0].set_yticks([0, 0.5, 1], labels=['0.0', '0.5', '1.0'])
ax[1].set_yticks([0, 0.5, 1], labels=['0.0', '0.5', '1.0'])
ax[1].set_xlabel('Duration of invasive ventilation (hrs)')
plt.tight_layout()
plt.savefig('viz/duration.jpg', dpi=300, bbox_inches='tight')

#################################################################

race_labels = {0: 'Black', 1: 'Asian', 2: 'White', 3: 'Hispanic'}
gender_labels = {0: 'Female', 1: 'Male'}
age_labels = {0: '≤20', 15: '>90'}
for i in range(1,15):
    age_labels[i] = f'{5*(i+3)+1}-{5*(i+4)}'

for x in ['charlson', 'pre_icu_los_oasis', 'gender', 'age_group']:
    if x == 'charlson':
        xlabel = 'Charlson Comorbidity Index'
    elif x == 'pre_icu_los_oasis':
        xlabel = 'pre-ICU OASIS Score'
    elif x == 'gender':
        xlabel = 'Sex'
    elif x == 'age_group':
        xlabel = 'Age'
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for i in range(2):
        if i == 0:
            df = eicu_df
        else:
            df = mimic_df
        df = df.dropna(subset=[x])
        frequency = df.groupby([x, 'race']).size().unstack(fill_value=0)
        proportions = frequency.div(frequency.sum(axis=0), axis=1).reset_index()
        melted = proportions.melt(id_vars=x, var_name='race', value_name='proportion')
        melted['race'] = melted['race'].map(race_labels)
        sns.barplot(
            data=melted, x=x, y='proportion', hue='race', 
            hue_order=['White', 'Black', 'Hispanic', 'Asian'],
            palette='Set2', ax=ax[i])
    ax[0].set_ylabel('eICU\nProportion')
    ax[1].set_ylabel('MIMIC-IV\nProportion')
    if x == 'charlson':
        ax[0].legend(title=None, fontsize=labelsize)
    else:
        ax[0].get_legend().remove()
    ax[1].get_legend().remove()
    if x == 'pre_icu_los_oasis':
        ax[0].set_yticks([0.0, 0.2, 0.4], labels=['0.0', '0.2', '0.4'])
        ax[1].set_yticks([0.0, 0.2, 0.4], labels=['0.0', '0.2', '0.4'])

    r = combined_df[x].dropna().unique().astype(int)
    r.sort()
    if x == 'gender':
        labels = [gender_labels[n] for n in r]
    elif x == 'age_group':
        labels = [age_labels[n] for n in r]
    else:
        labels = [str(n) for n in r]
    if x == 'age_group':
        fp = 'viz/age.jpg'
        ax[1].set_xticks(r, labels=labels, rotation=45)
    else:
        if x == 'gender':
            fp = 'viz/sex.jpg'
        if x == 'pre_icu_los_oasis':
            fp = 'viz/oasis.jpg'
        else:
            fp = f'viz/{x}.jpg'
        ax[1].set_xticks(r, labels=labels)
    ax[1].set_xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(fp, dpi=300, bbox_inches='tight')

#######################################################################

fig = plt.figure(figsize=(8,8))
gs = GridSpec(5,5)

ax_scatter = fig.add_subplot(gs[1:5, 0:4])
plt.subplots_adjust(wspace=0.2, hspace=0.2)

df_norm = eicu_df[
    ~((eicu_df['sao2'] < 88) & (eicu_df['spo2'] >= 88))
]
df_hypo = eicu_df[
    (eicu_df['sao2'] < 88) & (eicu_df['spo2'] >= 88)
]

sns.scatterplot(
    df_norm, x='spo2', y='sao2', marker='o', 
    color='tab:blue', alpha=0.2, ax=ax_scatter
)
sns.scatterplot(
    df_hypo, x='spo2', y='sao2', marker='o', 
    color='tab:red', alpha=0.2, ax=ax_scatter
)
ax_scatter.scatter([], [], marker='o', color='tab:blue', alpha=1, label='Norm')
ax_scatter.scatter([], [], marker='o', color='tab:red', alpha=1, label='H.H.')
ax_scatter.set_xlabel('SpO2 (%)')
ax_scatter.set_ylabel('SaO2 (%)')
ax_scatter.set_xlim([69, 102])
ax_scatter.set_ylim([69, 102])
ax_scatter.grid(False)

ax_hist_x = fig.add_subplot(gs[0,0:4])
sns.kdeplot(
    x=eicu_df['spo2'], clip=(70, 100), ax=ax_hist_x, 
    linewidth=2, color='green', fill=True)
ax_hist_x.yaxis.set_ticks([0, 0.2], labels=['0', '0.2'])
ax_hist_x.xaxis.set_ticklabels([])
ax_hist_x.grid(False)
ax_hist_x.set_xlabel('')
ax_hist_x.set_ylim([0,.35])
ax_hist_x.axvline(
    x=eicu_df['spo2'].mean(), linewidth=1.5,
    linestyle='--', color='green', 
)
ax_hist_x.text(
    eicu_df['spo2'].mean() - 4.4, 0.24,
    f'SpO2\nMean\n{eicu_df["spo2"].mean():.2f}', 
    color='green', fontsize=12, va='center', ha='left', 
    backgroundcolor='white', 
)

ax_hist_y = fig.add_subplot(gs[1:5, 4])
sns.kdeplot(
    y=eicu_df['sao2'], clip=(70, 100), ax=ax_hist_y, 
    linewidth=2, color='orange', fill=True)
ax_hist_y.xaxis.set_ticks([0, 0.2], labels=['0', '0.2'])
ax_hist_y.yaxis.set_ticklabels([])
ax_hist_y.grid(False)
ax_hist_y.set_ylabel('')
ax_hist_y.set_xlim([0,.35])
ax_hist_y.axhline(
    y=eicu_df['sao2'].mean(), xmin=0, xmax=1,
    linewidth=1.5, linestyle='--', color='orange',
)
ax_hist_y.text(
    0.15, eicu_df['sao2'].mean() - 2.7, 
    f'SaO2\nMean\n{eicu_df["sao2"].mean():.2f}', 
    color='orange', fontsize=12, va='center', ha='left', 
    backgroundcolor='white', 
)

fig.legend(loc='upper right', bbox_to_anchor=(0.915, 0.85), ncol=1, fontsize=14)
plt.savefig(f'viz/hidden_hypox.jpg', dpi=300, bbox_inches='tight')