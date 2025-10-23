import os
import ast
import numpy as np
import pandas as pd
import pickle as pk
import seaborn as sns
import matplotlib as mpl
from scipy.stats import sem
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.cm import ScalarMappable
from datasets import load_eicu, load_mimic

fontsize = 24
titlesize = 24
labelsize = 16
markersize = 10
linestyle = '--'
sns.set_style("ticks")
sns.set_style("whitegrid")
markers = ['o', 'X', 's', '^']
colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']
bar_colors = ['mediumturquoise', 'salmon', 'yellowgreen', 'goldenrod', 'mediumpurple']
plt.rcParams['hatch.linewidth'] = 2

# plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}'
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

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save(fig, path, tight=True):
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')

def myplot(ax, results_df, x, y, marker, color, label):
    sns.lineplot(
        data=results_df, x=x, y=y, errorbar=('ci', 95),
        marker=marker, markersize=markersize, linestyle=linestyle,
        color=color, linewidth=3, label=label, ax=ax
    )

def multiplot(ax, results_df, x, ys, labels):
    for i in range(len(ys)):
        myplot(ax, results_df, x, ys[i], markers[i], colors[i], labels[i])

def plot_query(results_df, N, fig, ax, title='', root=None, SN=False):
    suffix = '(SN)' if SN else '(non-SN)'
    sn_ = 'sn_' if SN else ''
    ys = [
        f'EYx0_{sn_}Error%', 
        f'EYx1_{sn_}Error%', 
        f'EYx1WVx0_{sn_}Error%',
        f'EYx1Vx0Wx1_{sn_}Error%',
    ]
    labels = [
        r'$\mathbb{E}[Y_{x_0}]$', 
        r'$\mathbb{E}[Y_{x_1}]$', 
        r'$\mathbb{E}[Y_{x_1, M_{x_0}}]$',
        r'$\mathbb{E}[Y_{x_1, V_{x_0, W_{x_1}}}]$',
    ]
    ax.set_xscale('log')
    multiplot(ax, results_df, 'n', ys, labels)
    ax.set_xticks(N, labels=[str(n) for n in N])
    ax.set_xlabel('Sample Size', fontsize=fontsize)
    ax.set_ylabel('Relative error (%)', fontsize=fontsize)
    ax.set_title(f'{title} queries {suffix}', fontsize=titlesize)

    plt.rc('text', usetex=True)
    ax.legend(ncols=2, fontsize=fontsize)
    save(fig, f'{root}/query_{sn_}error.jpg')
    plt.rc('text', usetex=False)
    ax.clear()

def plot_VDE_nuisance(results_df, N, fig, ax, title='', root=None):
    ys = [
        f'Epi3_vde', 
        f'Epi2_vde', 
        f'Epi1_vde',
    ]
    labels = [
        r'$\mathbb{E}[\hat{\pi}^3_0]$', 
        r'$\mathbb{E}[\hat{\pi}^2_0]$', 
        r'$\mathbb{E}[\hat{\pi}^1_0]$',
    ]
    ax.set_xscale('log')
    multiplot(ax, results_df, 'n', ys, labels)
    ax.axhline(y=1, color='black', linestyle='--')
    ax.set_xticks(N, labels=[str(n) for n in N])
    ax.set_xlabel('Sample Size', fontsize=fontsize)
    ax.set_ylabel('Empirical mean', fontsize=fontsize)
    ax.set_title(f'{title} nuisance (VDE)', fontsize=titlesize)
    
    plt.rc('text', usetex=True)
    ax.legend(ncols=3, fontsize=fontsize)
    save(fig, f'{root}/vde_nuisance.jpg')
    plt.rc('text', usetex=False)
    ax.clear()

def plot_synthetic(results_df, title='', root=None):
    assert root is not None
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    N = results_df['n']
    N = list(N.unique())
    N.sort()
    
    for SN in [False, True]:
        plot_query(results_df, N, fig, ax, title, root, SN)
    plot_VDE_nuisance(results_df, N, fig, ax, title, root)

def plot_estimates_bar(eicu_df, mimic_df, fig, ax, y_label='', root=None, SN=False):
    sn = '_sn' if SN else ''
    eicu_TE = eicu_df[f'Estimated_TE{sn}']
    eicu_NDE = eicu_df[f'Estimated_NDE{sn}']
    eicu_NIE = eicu_df[f'Estimated_NIE{sn}']
    eicu_NIE_ = eicu_df[f'Estimated_NIE*{sn}']
    eicu_VDE = eicu_df[f'Estimated_VDE{sn}']
    mimic_TE = mimic_df[f'Estimated_TE{sn}']
    mimic_NDE = mimic_df[f'Estimated_NDE{sn}']
    mimic_NIE = mimic_df[f'Estimated_NIE{sn}']
    mimic_NIE_ = mimic_df[f'Estimated_NIE*{sn}']
    mimic_VDE = mimic_df[f'Estimated_VDE{sn}']
    eicu_effects = np.vstack([eicu_TE, eicu_NDE, eicu_NIE, eicu_NIE_, eicu_VDE])
    mimic_effects = np.vstack([mimic_TE, mimic_NDE, mimic_NIE, mimic_NIE_, mimic_VDE])
    if y_label == 'Rate of ventilation (%)':
        eicu_effects = eicu_effects * 100
        mimic_effects = mimic_effects * 100
    eicu_means = np.mean(eicu_effects, axis=1)
    mimic_means = np.mean(mimic_effects, axis=1)
    eicu_stde = sem(eicu_effects, axis=1)
    mimic_stde = sem(mimic_effects, axis=1)
    if y_label == 'Rate of ventilation (%)':
        y = 'rate'
    elif y_label == 'Duration of ventilation (hrs)':
        y = 'duration'
    print(f'--- eICU ventilation {y} VDE {"(SN) " if SN else ""}---')
    print('mean = {:.{p}f}, 95% CI = ({:.{p}f}, {:.{p}f})'.format(
        eicu_means[-1],
        eicu_means[-1] - 1.96 * eicu_stde[-1],
        eicu_means[-1] + 1.96 * eicu_stde[-1],
        p = 1 + (y == 'rate'), 
    ))
    print(f'--- MIMIC-IV ventilation {y} VDE {"(SN) " if SN else ""}---')
    print('mean = {:.{p}f}, 95% CI = ({:.{p}f}, {:.{p}f})'.format(
        mimic_means[-1],
        mimic_means[-1] - 1.96 * mimic_stde[-1],
        mimic_means[-1] + 1.96 * mimic_stde[-1],
        p = 1 + (y == 'rate'), 
    ))
    
    width = 0.45
    x = np.arange(len(eicu_means))
    p2 = 0.3 if y_label == 'Rate of ventilation (%)' else 1.0
    eicu_bars = ax.bar(
        x - width/2, eicu_means, width, yerr=1.96*eicu_stde, 
        color=bar_colors, hatch='+', capsize=12,
    )
    mimic_bars = ax.bar(
        x + width/2, mimic_means, width, yerr=1.96*mimic_stde, 
        color=bar_colors, hatch='X', capsize=12, 
    )
    for bar_set in [eicu_bars, mimic_bars]:
        for bar in bar_set:
            height = bar.get_height()
            placement = 0.0 if height < 0 else -p2
            if y_label == 'Rate of ventilation (%)':
                text = f'{height:.2f}'
            else:
                text = f'{height:.1f}'
            ax.text(
                bar.get_x() + bar.get_width() / 2, placement, 
                text, ha='center', va='bottom', fontsize=12, fontweight='bold'
            )
    ax.bar(0, 0, color='gray', hatch='+', hatch_linewidth=5, label='eICU')
    ax.bar(0, 0, color='gray', hatch='X', hatch_linewidth=5, label='MIMIC-IV')
    ax.set_xticks(x)
    ax.set_xticklabels([
        'TE',
        'NDE',
        'NIE',
        'NIE*',
        'VDE',
    ], fontsize=16, rotation=0)
    ax.set_axisbelow(True)
    ax.set_ylabel(y_label, fontsize=fontsize)
    ax.tick_params(axis='y', which='major', labelsize=labelsize)
    ax.legend(fontsize=labelsize, handleheight=2)
    save(fig, f'{root}/effects{sn}.jpg')
    ax.clear()

def plot_cond_VDE_scatter(eicu_df, mimic_df, filter_vent, cond, root=None, SN=False):
    assert cond in ['spo2', 'sao2']
    sn = '_sn' if SN else ''
    cmap = 'viridis'

    def get_bin(o2, discrepancy):
        o2_bin = np.ceil(o2 / 5) * 5
        if o2_bin == 70:
            o2_bin = 75
        discrepancy_bin = np.ceil(discrepancy / 10) * 10
        if discrepancy_bin == -30:
            discrepancy_bin = -20
        return np.round(o2_bin), np.round(discrepancy_bin)

    eicu_cond_VDE = {}
    mimic_cond_VDE = {}
    eicu_cond_VDEs = eicu_df[f'Estimated_discrepancy/{cond}_conditional_VDE{sn}']
    mimic_cond_VDEs = eicu_df[f'Estimated_discrepancy/{cond}_conditional_VDE{sn}']
    dat_cond_VDE = [eicu_cond_VDE, mimic_cond_VDE]
    dat_cond_VDEs = [eicu_cond_VDEs, mimic_cond_VDEs]
    for cond_VDE, cond_VDEs in zip(dat_cond_VDE, dat_cond_VDEs):
        for c in cond_VDEs:
            c = ast.literal_eval(c).copy()
            for k,v in c.items():
                if k not in cond_VDE:
                    cond_VDE[k] = [v]
                else:
                    cond_VDE[k].append(v)
        for k,v in cond_VDE.items():
            cond_VDE[k] = np.mean(v)
            if not filter_vent:
                cond_VDE[k] *= 100

    all_colors = []
    eicu_data = load_eicu()
    mimic_data = load_mimic()
    if filter_vent:
        eicu_data = eicu_data[eicu_data['vented'] == 1]
        mimic_data = mimic_data[mimic_data['vented'] == 1]
    for idx, data in enumerate([eicu_data, mimic_data]):
        o2 = np.array(data[cond])
        discrepancy = np.array(data['discrepancy'])
        o2_bin = [get_bin(o2[i], discrepancy[i]) for i in range(len(o2))]
        o2_bin_ct = {i:o2_bin.count(i) for i in set(o2_bin)}
        o2_color = np.array([dat_cond_VDE[idx][b] for b in o2_bin])
        o2_color_idxs = [i for i in range(len(o2)) if o2_bin_ct[o2_bin[i]] >= 10]
        all_colors.append(o2_color[o2_color_idxs])
    all_colors = np.concatenate(all_colors)
    color_min, color_max = min(all_colors), max(all_colors)
    norm = Normalize(vmin=color_min, vmax=color_max)

    fig = plt.figure(figsize=(8, 6))
    gs = GridSpec(3, 1, height_ratios=[1, 1, 0.05], hspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax = [ax1, ax2]
    for idx, data in enumerate([eicu_data, mimic_data]):
        o2 = np.array(data[cond])
        discrepancy = np.array(data['discrepancy'])
        o2_bin = [get_bin(o2[i], discrepancy[i]) for i in range(len(o2))]
        o2_bin_ct = {i:o2_bin.count(i) for i in set(o2_bin)}
        o2_color = np.array([cond_VDE[b] for b in o2_bin])
        o2_color_idxs = [i for i in range(len(o2)) if o2_bin_ct[o2_bin[i]] >= 10]
        o2_outlier_idxs = [i for i in range(len(o2)) if o2_bin_ct[o2_bin[i]] < 10]
        ax[idx].scatter(
            o2[o2_color_idxs], discrepancy[o2_color_idxs], 
            c=o2_color[o2_color_idxs], s=0, alpha=1.0, 
            marker='o', cmap=cmap, norm=norm
        )
        ax[idx].scatter(
            o2[o2_color_idxs], discrepancy[o2_color_idxs], 
            c=o2_color[o2_color_idxs], s=16, alpha=0.3,
            marker='o', cmap=cmap, norm=norm
        )
        dat = 'eICU' if idx == 0 else 'MIMIC-IV'
        ax[idx].scatter(
            o2[o2_outlier_idxs], discrepancy[o2_outlier_idxs], 
            c='gray', s=16, alpha=0.3, marker='o', label=dat
        )
    if cond == 'spo2':
        ax[1].set_xlabel('SpO2 (%)', fontsize=fontsize)
    else:
        ax[1].set_xlabel('SaO2 (%)', fontsize=fontsize)
    cmap = ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(cmap, ax=ax, orientation='vertical', fraction=0.04)
    ax[0].tick_params(axis='both', which='major', labelsize=labelsize)
    ax[1].tick_params(axis='both', which='major', labelsize=labelsize)

    xticks = [70, 75, 80, 85, 90, 95, 100]
    if not filter_vent:
        ax[0].set_title('Rate of ventilation (%)', fontsize=fontsize)
    else:
        ax[0].set_title('Duration of ventilation (hrs)', fontsize=fontsize)
    ax[0].set_xticks(xticks, [])
    ax[1].set_xticks(xticks, [str(x) for x in xticks])
    fig.supylabel('Δ SpO2 - SaO2 (%)', fontsize=fontsize, fontweight='bold', x=-0.01)
    for i in range(2):
        ax[i].set_yticks(
            [-30, -20, -10, 0, 10, 20, 30], 
            ['-30', '-20', '-10', '0', '10', '20', '30']
        )
        ax[i].set_xlim(68, 102)
        ax[i].set_ylim(-34, 34)
        legend = ax[i].legend(handlelength=0, handletextpad=0, fontsize=labelsize)
        for handle in legend.legend_handles:
            handle.set_alpha(0.0)
    save(fig, f'{root}/{cond}_cond_effects{sn}.jpg', tight=False)
    cb.remove()
    ax.clear()

def plot_real(eicu_rate_df, eicu_duration_df, mimic_rate_df, mimic_duration_df, root=None):
    assert root is not None
    fig, ax = plt.subplots(1, 1, figsize=(8, 6.5))
    for SN in [False, True]:
        plot_estimates_bar(
            eicu_rate_df, mimic_rate_df, fig, ax, 
            'Rate of ventilation (%)', 
            root+'/rate', SN=SN
        )
        plot_estimates_bar(
            eicu_duration_df, mimic_duration_df, fig, ax, 
            'Duration of ventilation (hrs)', 
            root+'/duration', SN=SN
        )
        plot_cond_VDE_scatter(
            eicu_rate_df, mimic_rate_df, False,
            'spo2', root+'/rate', SN=SN
        )
        plot_cond_VDE_scatter(
            eicu_duration_df, mimic_duration_df, True,
            'spo2', root+'/duration', SN=SN
        )

if __name__ == '__main__':
    make_dir('./results/synthetic_plots')
    make_dir('./results/synthetic_plots/binary')
    make_dir('./results/synthetic_plots/continuous')
    make_dir('./results/semisynthetic_plots')
    make_dir('./results/real_plots')
    for dataset in ['eicu', 'mimic']:
        make_dir(f'./results/semisynthetic_plots/{dataset}')
    for Y in ['rate', 'duration']:
        make_dir(f'./results/real_plots/{Y}')

    print('Plotting synthetic experiments')
    synth_binary_df = pd.read_csv('results/synthetic_binary_B=100_K=5_C=0.0001.csv')
    synth_continuous_df = pd.read_csv('results/synthetic_continuous_B=100_K=5_C=0.0001.csv')
    plot_synthetic(synth_binary_df, title='Synthetic binary', root='results/synthetic_plots/binary')
    plot_synthetic(synth_continuous_df, title='Synthetic continuous', root='results/synthetic_plots/continuous')
    
    print('Plotting semi-synthetic experiments')
    semisynth_eicu_df = pd.read_csv('results/semisynthetic_eicu_B=100_K=5_C=0.0001.csv')
    semisynth_mimic_df = pd.read_csv('results/semisynthetic_mimic_B=100_K=5_C=0.0001.csv')
    plot_synthetic(semisynth_eicu_df, title='Semi-synthetic eICU', root='results/semisynthetic_plots/eicu')
    plot_synthetic(semisynth_mimic_df, title='Semi-synthetic MIMIC-IV', root='results/semisynthetic_plots/mimic')
    
    print('Plotting real-world experiments')
    real_eicu_rate_df = pd.read_csv('results/real_eicu_vented_B=500_K=5_C=0.0001.csv')
    real_mimic_rate_df = pd.read_csv('results/real_mimic_vented_B=500_K=5_C=0.0001.csv')
    real_eicu_duration_df = pd.read_csv('results/real_eicu_vent_duration_B=500_K=5_C=0.0001.csv')
    real_mimic_duration_df = pd.read_csv('results/real_mimic_vent_duration_B=500_K=5_C=0.0001.csv')
    plot_real(
        real_eicu_rate_df, real_eicu_duration_df,
        real_mimic_rate_df, real_mimic_duration_df,
        root='results/real_plots',
    )