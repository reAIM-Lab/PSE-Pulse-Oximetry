import os
import numpy as np
import pickle as pk
from datasets import (
    sample_synth_binary, 
    sample_synth_continuous,
    sample_semisynth,
    fit_semisynth,
    load_mimic,
    load_eicu,
    load_vars
)
from estimation import estimate_effects

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def run_synth(dataset, dims, bootstraps=30, K=5, clip=0.0, n_mc=1000000):
    assert dataset in ['binary', 'continuous']
    X = 'X'
    Y = 'Y'
    w_dim = dims['W']
    v_dim = dims['V']
    z_dim = dims['Z']
    W_cols = [f'W{i+1}' for i in range(w_dim)] if w_dim > 1 else ['W']
    V_cols = [f'V{i+1}' for i in range(v_dim)] if v_dim > 1 else ['V']
    Z_cols = [f'Z{i+1}' for i in range(z_dim)] if z_dim > 1 else ['Z']
    sampler = sample_synth_binary if dataset == 'binary' else sample_synth_continuous
    N = [1000, 2000, 4000, 8000, 16000, 32000]
    results_df = estimate_effects(
        sampler, None, X, Y, 
        W_cols, V_cols, Z_cols, True,
        N, bootstraps=bootstraps, K=K, clip=clip, 
        n_mc=n_mc, sample_kwargs={'dims':dims},
        hparam_path=f'hparams/synthetic_{dataset}.json',
    )
    save_path = f'synthetic_{dataset}'
    save_path += f'_B={bootstraps}'
    save_path += f'_K={K}_C={clip}'
    results_df.to_csv(f'results/{save_path}.csv', index=False)

def run_semisynth(dataset, bootstraps=30, K=5, clip=0.0, n_mc=1000000, sample_kwargs=None):
    assert dataset in ['eicu', 'mimic']
    sample_kwargs['dataset'] = dataset
    X = vars['X']
    Y = 'vented'
    W_cols = vars['W']
    V_cols = vars['V']
    Z_cols = vars['Z']
    sample_kwargs['Y'] = Y
    N = [1000, 2000, 4000, 8000, 16000, 32000]
    results_df = estimate_effects(
        sample_semisynth, None, X, Y, 
        W_cols, V_cols, Z_cols, True,
        N, bootstraps=bootstraps, K=K, clip=clip, 
        n_mc=n_mc, sample_kwargs=sample_kwargs,
        hparam_path=f'hparams/semisynthetic_{dataset}.json',
    )
    save_path = f'semisynthetic_{dataset}'
    save_path += f'_B={bootstraps}'
    save_path += f'_K={K}_C={clip}'
    results_df.to_csv(f'results/{save_path}.csv', index=False)

def run_real(dataset, Y, bootstraps=30, K=5, clip=0.0):
    assert dataset in ['eicu', 'mimic']
    assert Y in ['vented', 'vent_duration']
    vars = load_vars()
    X = vars['X']
    W_cols = vars['W']
    V_cols = vars['V']
    Z_cols = vars['Z']
    Y_binary = (Y == 'vented')
    df = load_eicu() if dataset == 'eicu' else load_mimic()
    df = df[df['vented'] == 1] if Y == 'vent_duration' else df
    results_df = estimate_effects(
        None, df, X, Y, W_cols, V_cols, Z_cols, 
        Y_binary, [len(df)], bootstraps=bootstraps, K=K, clip=clip,
        hparam_path=f'hparams/real_{dataset}_{Y}.json',
    )
    save_path = f'real_{dataset}_{Y}'
    save_path += f'_B={bootstraps}'
    save_path += f'_K={K}_C={clip}'
    results_df.to_csv(f'results/{save_path}.csv', index=False)

if __name__ == '__main__':
    make_dir('./results/')
    make_dir('./hparams/')
    print('Running synthetic experiments')
    dims = {
        'binary': {'Z': 1, 'X': 1, 'W': 1, 'V': 1, 'Y': 1},
        'continuous': {'Z': 3, 'X': 1, 'W': 10, 'V': 3, 'Y': 1},
    }
    for dataset in ['binary', 'continuous']:
        print('-----', dataset, '-----')
        run_synth(
            dataset, dims[dataset], bootstraps=100, 
            K=5, clip=1e-4, n_mc=1000000
        )

    print('\nRunning semi-synthetic experiments')
    for dataset in ['eicu', 'mimic']:
        print('-----', dataset, '-----')
        vars, dtypes = fit_semisynth(dataset, force_refit=False)
        sample_kwargs = {'vars': vars, 'dtypes': dtypes}
        run_semisynth(
            dataset, bootstraps=100, K=5, clip=1e-4, 
            n_mc=1000000, sample_kwargs=sample_kwargs
        )

    print('\nReal-world')
    for dataset in ['eicu', 'mimic']:
        for Y in ['vented', 'vent_duration']:
            print('-----', dataset, Y, '-----')
            run_real(dataset, Y, bootstraps=500, K=5, clip=1e-4)