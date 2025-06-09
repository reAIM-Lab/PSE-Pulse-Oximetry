import os
import json
import cupy as cp
import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm
import statsmodels.api as sm
from sklearn.model_selection import KFold, GridSearchCV

HPARAMS = {
    'n_estimators': [20, 50, 100, 200],     # Number of boosting rounds
    'max_depth': [3, 4, 5, 6],             # Maximum depth of a tree
    'reg_lambda': [0.5, 1, 2, 5],          # L2 regularization term on weights
}

def estimate_effects(
        sample, all_df, X, Y, W_cols, V_cols, Z_cols, Y_binary,
        N, clip=0.0, K=5, bootstraps=30, n_mc=1000, 
        sample_kwargs=None, hparam_path=None,
    ):
    assert (sample is not None) or (all_df is not None)
    gt_effects = None
    if sample is None:
        N = [len(all_df)]
    if all_df is None:
        mc_df, gt_effects = compute_gt_effects(sample, n_mc, X, Y, 42, sample_kwargs)
        EYx0, EYx1, EYx1WVx0, EYx1Vx0Wx1, TV, TE, NDE, NIE, VDE = gt_effects
        print(f'{Y}[x0]      : {EYx0:.4f}')
        print(f'{Y}[x1]      : {EYx1:.4f}')
        print(f'{Y}[x1WVx0]  : {EYx1WVx0:.4f}')
        print(f'{Y}[x1Vx0Wx1]: {EYx1Vx0Wx1:.4f}')
        print('*'*40)
        print(f'Total variation (TV)         : {TV:.4f}')
        print(f'Total effect (TE)            : {TE:.4f}')
        print(f'Natural direct effect (NDE)  : {NDE:.4f}')
        print(f'Natural indirect effect (NIE): {NIE:.4f}')
        print(f'V-specific effect (VDE)      : {VDE:.4f}')
        print('*'*40)

    model_list = [
        'px_z', 'px_wz',
        'px_vz', 'px_wvz',
        'y_zx0', 'y_zx1',
        'y_wvzx0', 'y_wvzx1',
        'mu2_ne', 'mu1_ne',
        'mu3_vde', 'mu2_vde', 'mu1_vde'
    ]
    if all_df is not None:
        model_list += ['y_vzx0', 'y_vzx1', 'mu2_ne*', 'mu1_ne*']

    if os.path.exists(hparam_path):
        print('Loading hparams')
        with open(hparam_path, 'r') as f:
            hp_dict = json.load(f)
    else:
        hp_dict = {}
    if sample is not None:
        search_df = sample(n=N[-1], seed=1001, **sample_kwargs)
    else:
        search_df = all_df.copy()
    aux_cols = {c: None for c in ['mu2_ne', 'mu2_ne*', 'mu3_vde', 'mu2_vde']}
    aux_df = pd.DataFrame(aux_cols, index=search_df.index)
    search_df = pd.concat([search_df, aux_df], axis=1)
    search_df = search_df.reset_index(drop=True)  
    search_df_x0 = search_df[search_df[X] == 0]
    search_df_x1 = search_df[search_df[X] == 1]
    idxs = np.arange(len(search_df))
    print('Fitting propensity hparams')
    for m in tqdm(model_list[:4]):
        if m in hp_dict.keys():
            continue
        cols = [z for z in Z_cols]
        cond = m.split('_')[1]
        if 'w' in cond:
            cols += W_cols
        if 'v' in cond:
            cols += V_cols
        hp = grid_search(search_df[cols], search_df[X], True, HPARAMS)
        hp_dict[m] = hp

    print('Fitting outcome hparams')
    if 'y_zx0' not in hp_dict.keys():
        hp_dict['y_zx0'] = grid_search(
            search_df_x0[Z_cols], search_df_x0[Y], Y_binary, HPARAMS)
    if 'y_zx1' not in hp_dict.keys():
        hp_dict['y_zx1'] = grid_search(
            search_df_x1[Z_cols], search_df_x1[Y], Y_binary, HPARAMS)
    if 'y_wvzx0' not in hp_dict.keys():
        hp_dict['y_wvzx0'] = grid_search(
            search_df_x0[Z_cols + W_cols + V_cols], search_df_x0[Y], Y_binary, HPARAMS)
    if 'y_wvzx1' not in hp_dict.keys():
        hp_dict['y_wvzx1'] = grid_search(
            search_df_x1[Z_cols + W_cols + V_cols], search_df_x1[Y], Y_binary, HPARAMS)

    if all_df is not None:
        if 'y_vzx0' not in hp_dict.keys():
            hp_dict['y_vzx0'] = grid_search(
                search_df_x0[Z_cols + V_cols], search_df_x0[Y], Y_binary, HPARAMS)
        if 'y_vzx1' not in hp_dict.keys():
            hp_dict['y_vzx1'] = grid_search(
                search_df_x1[Z_cols + V_cols], search_df_x1[Y], Y_binary, HPARAMS)

    print('Fitting nested outcome hparams')
    if 'mu2_zx0' not in hp_dict.keys():
        np.random.seed(1001)
        np.random.shuffle(idxs)
        mu, ns = np.array_split(idxs, 2)
        df_mu, df_ns = search_df.iloc[mu], search_df.iloc[ns]
        (_, df_mu_x0), (_, df_mu_x1) = df_mu.groupby(df_mu[X])
        (_, df_ns_x0), (_, df_ns_x1) = df_ns.groupby(df_ns[X])
        y_wvzx1 = fit(df_mu_x1[W_cols + V_cols + Z_cols], df_mu_x1[Y], binary=Y_binary, **hp_dict['y_wvzx1'])
        df_ns_x0['mu2_ne'] = pred(df_ns_x0[W_cols + V_cols + Z_cols], y_wvzx1, binary=Y_binary)
        hp = grid_search(df_ns_x0[Z_cols], df_ns_x0['mu2_ne'], False, HPARAMS)
        hp_dict['mu2_zx0'] = hp

    if all_df is not None and 'mu2_zx0*' not in hp_dict.keys():
        np.random.seed(1001)
        np.random.shuffle(idxs)
        mu, ns = np.array_split(idxs, 2)
        df_mu, df_ns = search_df.iloc[mu], search_df.iloc[ns]
        (_, df_mu_x0), (_, df_mu_x1) = df_mu.groupby(df_mu[X])
        (_, df_ns_x0), (_, df_ns_x1) = df_ns.groupby(df_ns[X])
        y_vzx1 = fit(df_mu_x1[V_cols + Z_cols], df_mu_x1[Y], binary=Y_binary, **hp_dict['y_vzx1'])
        df_ns_x0['mu2_ne*'] = pred(df_ns_x0[V_cols + Z_cols], y_vzx1, binary=Y_binary)
        hp = grid_search(df_ns_x0[Z_cols], df_ns_x0['mu2_ne*'], False, HPARAMS)
        hp_dict['mu2_zx0*'] = hp
    
    if 'mu3_wzx0' not in hp_dict.keys():
        np.random.seed(1043)
        np.random.shuffle(idxs)
        mu, ns1, ns2 = np.array_split(idxs, 3)
        df_mu, df_ns1, df_ns2 = search_df.iloc[mu], search_df.iloc[ns1], search_df.iloc[ns2]
        (_, df_mu_x0), (_, df_mu_x1) = df_mu.groupby(df_mu[X])
        (_, df_ns1_x0), (_, df_ns1_x1) = df_ns1.groupby(df_ns1[X])
        (_, df_ns2_x0), (_, df_ns2_x1) = df_ns2.groupby(df_ns2[X])
        y_wvzx1 = fit(df_mu_x1[W_cols + V_cols + Z_cols], df_mu_x1[Y], binary=Y_binary, **hp_dict['y_wvzx1'])
        df_ns1_x0['mu3_vde'] = pred(df_ns1_x0[W_cols + V_cols + Z_cols], y_wvzx1, binary=Y_binary)
        hp = grid_search(df_ns1_x0[W_cols + Z_cols], df_ns1_x0['mu3_vde'], False, HPARAMS)
        hp_dict['mu3_wzx0'] = hp
    
    if 'mu2_zx1' not in hp_dict.keys():
        mu3_wzx0 = fit(df_ns1_x0[W_cols + Z_cols], df_ns1_x0['mu3_vde'], binary=False, **hp_dict['mu3_wzx0'])
        df_ns2_x1['mu2_vde'] = pred(df_ns2_x1[W_cols + Z_cols], mu3_wzx0, binary=False)
        hp = grid_search(df_ns2_x1[Z_cols], df_ns2_x1['mu2_vde'], False, HPARAMS)
        hp_dict['mu2_zx1'] = hp

    with open(hparam_path, 'w') as f:
        json.dump(hp_dict, f, indent=4)

    results_df = pd.DataFrame()
    aux_cols = {c: None for c in model_list}
    synthetic = (sample is not None)
    for n in N:
        all_idxs = np.arange(n)
        print('Sample size:', n)
        for i in tqdm(range(bootstraps)):
            folds = KFold(n_splits=K, shuffle=True, random_state=i)
            if sample is not None:
                df = sample(n=n, seed=i, **sample_kwargs)
                df = df[[X,Y] + W_cols + V_cols + Z_cols]
            else:
                df = all_df.copy()
            aux_df = pd.DataFrame(aux_cols, index=df.index)
            df = pd.concat([df, aux_df], axis=1)
            df = df.reset_index(drop=True)
            for tr, ts in folds.split(all_idxs):
                df_tr, df_ts = df.iloc[tr], df.iloc[ts]
                (_, df_tr_x0), (_, df_tr_x1) = df_tr.groupby(df_tr[X])

                # Propensity models
                px_z = fit(df_tr[Z_cols], df_tr[X], **hp_dict['px_z'])
                px_wz = fit(df_tr[W_cols + Z_cols], df_tr[X], **hp_dict['px_wz'])
                px_vz = fit(df_tr[V_cols + Z_cols], df_tr[X], **hp_dict['px_vz'])
                px_wvz = fit(df_tr[W_cols + V_cols + Z_cols], df_tr[X], **hp_dict['px_wvz'])
                df.loc[ts, 'px_z'] = pred(df_ts[Z_cols], px_z, clip=clip)
                df.loc[ts, 'px_wz'] = pred(df_ts[W_cols + Z_cols], px_wz, clip=clip)
                df.loc[ts, 'px_vz'] = pred(df_ts[V_cols + Z_cols], px_vz, clip=clip)
                df.loc[ts, 'px_wvz'] = pred(df_ts[W_cols + V_cols + Z_cols], px_wvz, clip=clip)

                # Outcome models
                y_zx0 = fit(df_tr_x0[Z_cols], df_tr_x0[Y], binary=Y_binary, **hp_dict['y_zx0'])
                y_zx1 = fit(df_tr_x1[Z_cols], df_tr_x1[Y], binary=Y_binary, **hp_dict['y_zx1'])
                y_wvzx0 = fit(df_tr_x0[W_cols + V_cols + Z_cols], df_tr_x0[Y], binary=Y_binary, **hp_dict['y_wvzx0'])
                y_wvzx1 = fit(df_tr_x1[W_cols + V_cols + Z_cols], df_tr_x1[Y], binary=Y_binary, **hp_dict['y_wvzx1'])
                df.loc[ts, 'y_zx0'] = pred(df_ts[Z_cols], y_zx0, binary=Y_binary)
                df.loc[ts, 'y_zx1'] = pred(df_ts[Z_cols], y_zx1, binary=Y_binary)
                df.loc[ts, 'y_wvzx0'] = pred(df_ts[W_cols + V_cols + Z_cols], y_wvzx0, binary=Y_binary)
                df.loc[ts, 'y_wvzx1'] = pred(df_ts[W_cols + V_cols + Z_cols], y_wvzx1, binary=Y_binary)

                # Split df_tr into two equal parts mu and ns
                np.random.seed(i)
                np.random.shuffle(tr)
                mu, ns = np.array_split(tr, 2)
                df_mu, df_ns = df.iloc[mu], df.iloc[ns]
                (_, df_mu_x0), (_, df_mu_x1) = df_mu.groupby(df_mu[X])
                (_, df_ns_x0), (_, df_ns_x1) = df_ns.groupby(df_ns[X])

                # Regress Y on W, V, Z at level x1
                y_wvzx1 = fit(df_mu_x1[W_cols + V_cols + Z_cols], df_mu_x1[Y], binary=Y_binary, **hp_dict['y_wvzx1'])
                df_ns_x0['mu2_ne'] = pred(df_ns_x0[W_cols + V_cols + Z_cols], y_wvzx1, binary=Y_binary)
                df.loc[ts, 'mu2_ne'] = pred(df_ts[W_cols + V_cols + Z_cols], y_wvzx1, binary=Y_binary)
                # Regress E[Y|W, V, Z, X=1] on Z at level x0
                mu1_zx0 = fit(df_ns_x0[Z_cols], df_ns_x0['mu2_ne'], binary=False, **hp_dict['mu2_zx0'])
                df.loc[ts, 'mu1_ne'] = pred(df_ts[Z_cols], mu1_zx0, binary=False)
                if Y_binary:
                    df.loc[ts, 'mu1_ne'] = np.clip(df.loc[ts, 'mu1_ne'], a_min=0, a_max=1)

                if all_df is not None:
                    y_vzx1 = fit(df_mu_x1[V_cols + Z_cols], df_mu_x1[Y], binary=Y_binary, **hp_dict['y_vzx1'])
                    df_ns_x0['mu2_ne*'] = pred(df_ns_x0[V_cols + Z_cols], y_vzx1, binary=Y_binary)
                    df.loc[ts, 'mu2_ne*'] = pred(df_ts[V_cols + Z_cols], y_vzx1, binary=Y_binary)
                    # Regress E[Y|W, V, Z, X=1] on Z at level x0
                    mu1_zx0_ = fit(df_ns_x0[Z_cols], df_ns_x0['mu2_ne*'], binary=False, **hp_dict['mu2_zx0*'])
                    df.loc[ts, 'mu1_ne*'] = pred(df_ts[Z_cols], mu1_zx0_, binary=False)
                    if Y_binary:
                        df.loc[ts, 'mu1_ne*'] = np.clip(df.loc[ts, 'mu1_ne*'], a_min=0, a_max=1)

                # Split df_tr into three equal parts mu, ns1, ns2
                np.random.seed(i+42)
                np.random.shuffle(tr)
                mu, ns1, ns2 = np.array_split(tr, 3)
                df_mu, df_ns1, df_ns2 = df.iloc[mu], df.iloc[ns1], df.iloc[ns2]
                (_, df_mu_x0), (_, df_mu_x1) = df_mu.groupby(df_mu[X])
                (_, df_ns1_x0), (_, df_ns1_x1) = df_ns1.groupby(df_ns1[X])
                (_, df_ns2_x0), (_, df_ns2_x1) = df_ns2.groupby(df_ns2[X])

                # Regress Y on W, V, Z at level x1
                y_wvzx1 = fit(df_mu_x1[W_cols + V_cols + Z_cols], df_mu_x1[Y], binary=Y_binary, **hp_dict['y_wvzx1'])
                df_ns1_x0['mu3_vde'] = pred(df_ns1_x0[W_cols + V_cols + Z_cols], y_wvzx1, binary=Y_binary)
                df.loc[ts, 'mu3_vde'] = pred(df_ts[W_cols + V_cols + Z_cols], y_wvzx1, binary=Y_binary)
                # Regress E[Y | X=x1, W, V, Z] on W, Z at level x0
                mu1_wzx0 = fit(df_ns1_x0[W_cols + Z_cols], df_ns1_x0['mu3_vde'], binary=False, **hp_dict['mu3_wzx0'])
                df_ns2_x1['mu2_vde'] = pred(df_ns2_x1[W_cols + Z_cols], mu1_wzx0, binary=False)
                df.loc[ts, 'mu2_vde'] = pred(df_ts[W_cols + Z_cols], mu1_wzx0, binary=False)
                if Y_binary:
                    df_ns2_x1['mu2_vde'] = np.clip(df_ns2_x1['mu2_vde'], a_min=0, a_max=1)
                    df.loc[ts, 'mu2_vde'] = np.clip(df.loc[ts, 'mu2_vde'], a_min=0, a_max=1)
                # Regress E[E[Y | X=x1, W, V, Z] | X=x0, W, Z] on Z at level x1
                mu2_zx1 = fit(df_ns2_x1[Z_cols], df_ns2_x1['mu2_vde'], binary=False, **hp_dict['mu2_zx1'])
                df.loc[ts, 'mu1_vde'] = pred(df_ts[Z_cols], mu2_zx1, binary=False)
                if Y_binary:
                    df.loc[ts, 'mu1_vde'] = np.clip(df.loc[ts, 'mu1_vde'], a_min=0, a_max=1)
            results = compute_effects_from_df(df, X, Y, Y_binary, synthetic, gt_effects)
            results_df = pd.concat(
                [results_df, results], 
                ignore_index=True,
            )
    return results_df

def compute_gt_effects(sample, n_mc, X, Y, seed, sample_kwargs):
    mc_df = sample(n=n_mc, seed=seed, **sample_kwargs)
    mc_df_x0 = mc_df[mc_df[X] == 0]
    mc_df_x1 = mc_df[mc_df[X] == 1]
    EYx0 = np.mean(mc_df[f'{Y}[x0]'])
    EYx1 = np.mean(mc_df[f'{Y}[x1]'])
    EYx1WVx0 = np.mean(mc_df[f'{Y}[x1WVx0]'])
    EYx1Vx0Wx1 = np.mean(mc_df[f'{Y}[x1Vx0Wx1]'])
    TV = np.mean(mc_df_x1[Y]) - np.mean(mc_df_x0[Y])
    TE = np.mean(mc_df[f'{Y}[x1]']) - np.mean(mc_df[f'{Y}[x0]'])
    NDE = np.mean(mc_df[f'{Y}[x1WVx0]']) - np.mean(mc_df[f'{Y}[x0]'])
    NIE = np.mean(mc_df[f'{Y}[x1]']) - np.mean(mc_df[f'{Y}[x1WVx0]'])
    VDE = np.mean(mc_df[f'{Y}[x1]']) - np.mean(mc_df[f'{Y}[x1Vx0Wx1]'])
    gt_effects = (EYx0, EYx1, EYx1WVx0, EYx1Vx0Wx1, TV, TE, NDE, NIE, VDE)
    return mc_df, gt_effects

def compute_effects_from_df(df, X, Y, Y_binary, synthetic, gt_effects):
    if synthetic:
        EYx0, EYx1, EYx1WVx0, EYx1Vx0Wx1, TV, TE, NDE, NIE, VDE = gt_effects

    # do(x) nuisance parameters
    pi_x0 = (df[X] == 0) / (1 - df['px_z'])
    pi_x1 = (df[X] == 1) / (df['px_z'])
    pi_x0_sn = pi_x0 / np.mean(pi_x0)
    pi_x1_sn = pi_x1 / np.mean(pi_x1)
    Epi_x0 = np.mean(pi_x0)
    Epi_x1 = np.mean(pi_x1)

    # do(x) standard
    hat_Yx0 = pi_x0 * (df[Y] - df['y_zx0']) + df['y_zx0']
    hat_Yx1 = pi_x1 * (df[Y] - df['y_zx1']) + df['y_zx1']
    Estimated_EYx0 = np.mean(hat_Yx0)
    Estimated_EYx1 = np.mean(hat_Yx1)

    # do(x) self-normalized
    hat_Yx0_sn = pi_x0_sn * (df[Y] - df['y_zx0']) + df['y_zx0']
    hat_Yx1_sn = pi_x1_sn * (df[Y] - df['y_zx1']) + df['y_zx1']
    Estimated_EYx0_sn = np.mean(hat_Yx0_sn)
    Estimated_EYx1_sn = np.mean(hat_Yx1_sn)

    # NDE/NIE nuisance parameters
    pi2 = (df[X] == 1) * (1 - df['px_wvz']) / (df['px_wvz'] * (1 - df['px_z']))
    pi1 = (df[X] == 0) / (1 - df['px_z'])
    pi2_sn = pi2 / np.mean(pi2)
    pi1_sn = pi1 / np.mean(pi1)
    Epi2_ne = np.mean(pi2)
    Epi1_ne = np.mean(pi1)

    # NDE/NIE standard
    Estimated_EYx1WVx0_pi2Y = np.mean((pi2 * df[Y]))
    Estimated_EYx1WVx0_pi2mu2 = np.mean((pi2 * df['mu2_ne']))
    Estimated_EYx1WVx0_pi1mu2 = np.mean(pi1 * df['mu2_ne'])
    Estimated_EYx1WVx0_pi1mu1 = np.mean(pi1 * df['mu1_ne'])
    Estimated_EYx1WVx0_mu1 = np.mean(df['mu1_ne'])
    Estimated_EYx1WVx0 = (
        (Estimated_EYx1WVx0_pi2Y - Estimated_EYx1WVx0_pi2mu2) + 
        (Estimated_EYx1WVx0_pi1mu2 - Estimated_EYx1WVx0_pi1mu1) + 
        Estimated_EYx1WVx0_mu1
    )
    Estimated_NDE = Estimated_EYx1WVx0 - Estimated_EYx0
    Estimated_NIE = Estimated_EYx1 - Estimated_EYx1WVx0

    # NDE standard sub-estimates
    Estimated_NDE_pi2Y = Estimated_EYx1WVx0_pi2Y - Estimated_EYx0
    Estimated_NDE_pi2mu2 = Estimated_EYx1WVx0_pi2mu2 - Estimated_EYx0
    Estimated_NDE_pi1mu2 = Estimated_EYx1WVx0_pi1mu2 - Estimated_EYx0
    Estimated_NDE_pi1mu1 = Estimated_EYx1WVx0_pi1mu1 - Estimated_EYx0
    Estimated_NDE_mu1 = Estimated_EYx1WVx0_mu1 - Estimated_EYx0

    # NIE standard sub-estimates
    Estimated_NIE_pi2Y = Estimated_EYx1 - Estimated_EYx1WVx0_pi2Y
    Estimated_NIE_pi2mu2 = Estimated_EYx1 - Estimated_EYx1WVx0_pi2mu2
    Estimated_NIE_pi1mu2 = Estimated_EYx1 - Estimated_EYx1WVx0_pi1mu2
    Estimated_NIE_pi1mu1 = Estimated_EYx1 - Estimated_EYx1WVx0_pi1mu1
    Estimated_NIE_mu1 = Estimated_EYx1 - Estimated_EYx1WVx0_mu1

    # NDE/NIE self-normalized
    Estimated_EYx1WVx0_pi2Y_sn = np.mean((pi2_sn * df[Y]))
    Estimated_EYx1WVx0_pi2mu2_sn = np.mean((pi2_sn * df['mu2_ne']))
    Estimated_EYx1WVx0_pi1mu2_sn = np.mean(pi1_sn * df['mu2_ne'])
    Estimated_EYx1WVx0_pi1mu1_sn = np.mean(pi1_sn * df['mu1_ne'])
    Estimated_EYx1WVx0_sn = (
        (Estimated_EYx1WVx0_pi2Y_sn - Estimated_EYx1WVx0_pi2mu2_sn) + 
        (Estimated_EYx1WVx0_pi1mu2_sn - Estimated_EYx1WVx0_pi1mu1_sn) + 
        Estimated_EYx1WVx0_mu1
    )
    Estimated_NDE_sn = Estimated_EYx1WVx0_sn - Estimated_EYx0_sn
    Estimated_NIE_sn = Estimated_EYx1_sn - Estimated_EYx1WVx0_sn

    # NDE self-normalized sub-estimates
    Estimated_NDE_pi2Y_sn = Estimated_EYx1WVx0_pi2Y_sn - Estimated_EYx0_sn
    Estimated_NDE_pi2mu2_sn = Estimated_EYx1WVx0_pi2mu2_sn - Estimated_EYx0_sn
    Estimated_NDE_pi1mu2_sn = Estimated_EYx1WVx0_pi1mu2_sn - Estimated_EYx0_sn
    Estimated_NDE_pi1mu1_sn = Estimated_EYx1WVx0_pi1mu1_sn - Estimated_EYx0_sn
    Estimated_NDE_mu1_sn = Estimated_EYx1WVx0_mu1 - Estimated_EYx0_sn

    # NIE self-normalized sub-estimates
    Estimated_NIE_pi2Y_sn = Estimated_EYx1_sn - Estimated_EYx1WVx0_pi2Y_sn
    Estimated_NIE_pi2mu2_sn = Estimated_EYx1_sn - Estimated_EYx1WVx0_pi2mu2_sn
    Estimated_NIE_pi1mu2_sn = Estimated_EYx1_sn - Estimated_EYx1WVx0_pi1mu2_sn
    Estimated_NIE_pi1mu1_sn = Estimated_EYx1_sn - Estimated_EYx1WVx0_pi1mu1_sn
    Estimated_NIE_mu1_sn = Estimated_EYx1_sn - Estimated_EYx1WVx0_mu1

    # VDE nuisance parameters
    pi3 = (df[X] == 1) * (1 - df['px_wvz']) * df['px_wz']  / (df['px_wvz'] * (1 - df['px_wz']) * df['px_z'])
    pi2 = (df[X] == 0) * df['px_wz']  / ((1 - df['px_wz']) * df['px_z'])
    pi1 = (df[X] == 1) / df['px_z']
    pi3_sn = pi3 / np.mean(pi3)
    pi2_sn = pi2 / np.mean(pi2)
    pi1_sn = pi1 / np.mean(pi1)
    Epi3_vde = np.mean(pi3)
    Epi2_vde = np.mean(pi2)
    Epi1_vde = np.mean(pi1)

    # VDE standard
    Estimated_EYx1Vx0Wx1_pi3Y = np.mean((pi3 * df[Y]))
    Estimated_EYx1Vx0Wx1_pi3mu3 = np.mean((pi3 * df['mu3_vde']))
    Estimated_EYx1Vx0Wx1_pi2mu3 = np.mean((pi2 * df['mu3_vde']))
    Estimated_EYx1Vx0Wx1_pi2mu2 = np.mean((pi2 * df['mu2_vde']))
    Estimated_EYx1Vx0Wx1_pi1mu2 = np.mean((pi1 * df['mu2_vde']))
    Estimated_EYx1Vx0Wx1_pi1mu1 = np.mean((pi1 * df['mu1_vde']))
    Estimated_EYx1Vx0Wx1_mu1 = np.mean((df['mu1_vde']))
    Estimated_EYx1Vx0Wx1 = (
        (Estimated_EYx1Vx0Wx1_pi3Y -Estimated_EYx1Vx0Wx1_pi3mu3) + 
        (Estimated_EYx1Vx0Wx1_pi2mu3 - Estimated_EYx1Vx0Wx1_pi2mu2) + 
        (Estimated_EYx1Vx0Wx1_pi1mu2 - Estimated_EYx1Vx0Wx1_pi1mu1) + 
        Estimated_EYx1Vx0Wx1_mu1
    )
    Estimated_VDE = Estimated_EYx1 - Estimated_EYx1Vx0Wx1

    # VDE standard sub-estimates
    Estimated_VDE_pi3Y = Estimated_EYx1 - Estimated_EYx1Vx0Wx1_pi3Y
    Estimated_VDE_pi3mu3 = Estimated_EYx1 - Estimated_EYx1Vx0Wx1_pi3mu3
    Estimated_VDE_pi2mu3 = Estimated_EYx1 - Estimated_EYx1Vx0Wx1_pi2mu3
    Estimated_VDE_pi2mu2 = Estimated_EYx1 - Estimated_EYx1Vx0Wx1_pi2mu2
    Estimated_VDE_pi1mu2 = Estimated_EYx1 - Estimated_EYx1Vx0Wx1_pi1mu2
    Estimated_VDE_pi1mu1 = Estimated_EYx1 - Estimated_EYx1Vx0Wx1_pi1mu1
    Estimated_VDE_mu1 = Estimated_EYx1 - Estimated_EYx1Vx0Wx1_mu1

    # VDE self-normalized
    Estimated_EYx1Vx0Wx1_pi3Y_sn = np.mean((pi3_sn * df[Y]))
    Estimated_EYx1Vx0Wx1_pi3mu3_sn = np.mean((pi3_sn * df['mu3_vde']))
    Estimated_EYx1Vx0Wx1_pi2mu3_sn = np.mean((pi2_sn * df['mu3_vde']))
    Estimated_EYx1Vx0Wx1_pi2mu2_sn = np.mean((pi2_sn * df['mu2_vde']))
    Estimated_EYx1Vx0Wx1_pi1mu2_sn = np.mean((pi1_sn * df['mu2_vde']))
    Estimated_EYx1Vx0Wx1_pi1mu1_sn = np.mean((pi1_sn * df['mu1_vde']))
    Estimated_EYx1Vx0Wx1_sn = (
        (Estimated_EYx1Vx0Wx1_pi3Y_sn - Estimated_EYx1Vx0Wx1_pi3mu3_sn) + 
        (Estimated_EYx1Vx0Wx1_pi2mu3_sn - Estimated_EYx1Vx0Wx1_pi2mu2_sn) + 
        (Estimated_EYx1Vx0Wx1_pi1mu2_sn - Estimated_EYx1Vx0Wx1_pi1mu1_sn) + 
        Estimated_EYx1Vx0Wx1_mu1
    )
    Estimated_VDE_sn = Estimated_EYx1_sn - Estimated_EYx1Vx0Wx1_sn

    # VDE self-normalized sub-estimates
    Estimated_VDE_pi3Y_sn = Estimated_EYx1_sn - Estimated_EYx1Vx0Wx1_pi3Y_sn
    Estimated_VDE_pi3mu3_sn = Estimated_EYx1_sn - Estimated_EYx1Vx0Wx1_pi3mu3_sn
    Estimated_VDE_pi2mu3_sn = Estimated_EYx1_sn - Estimated_EYx1Vx0Wx1_pi2mu3_sn
    Estimated_VDE_pi2mu2_sn = Estimated_EYx1_sn - Estimated_EYx1Vx0Wx1_pi2mu2_sn
    Estimated_VDE_pi1mu2_sn = Estimated_EYx1_sn - Estimated_EYx1Vx0Wx1_pi1mu2_sn
    Estimated_VDE_pi1mu1_sn = Estimated_EYx1_sn - Estimated_EYx1Vx0Wx1_pi1mu1_sn
    Estimated_VDE_mu1_sn = Estimated_EYx1_sn - Estimated_EYx1Vx0Wx1_mu1

    if synthetic:
        results = {
            'n': [len(df)],
            'EYx0': [EYx0],
            'EYx1': [EYx1],
            'EYx1WVx0': [EYx1WVx0],
            'EYx1Vx0Wx1': [EYx1Vx0Wx1],
            'NDE': [NDE],
            'NIE': [NIE],
            'VDE': [VDE],
            # Estimates
            'Estimated_EYx0': [Estimated_EYx0],
            'Estimated_EYx1': [Estimated_EYx1],
            'Estimated_EYx0_sn': [Estimated_EYx0_sn],
            'Estimated_EYx1_sn': [Estimated_EYx1_sn],
            'Estimated_EYx1WVx0': [Estimated_EYx1WVx0],
            'Estimated_EYx1Vx0Wx1': [Estimated_EYx1Vx0Wx1],
            'Estimated_EYx1WVx0_sn': [Estimated_EYx1WVx0_sn],
            'Estimated_EYx1Vx0Wx1_sn': [Estimated_EYx1Vx0Wx1_sn],
            'Estimated_NDE': [Estimated_NDE],
            'Estimated_NIE': [Estimated_NIE],
            'Estimated_VDE': [Estimated_VDE],
            'Estimated_NDE_sn': [Estimated_NDE_sn],
            'Estimated_NIE_sn': [Estimated_NIE_sn],
            'Estimated_VDE_sn': [Estimated_VDE_sn],
            # Absolute errors
            'EYx0_Error': [Estimated_EYx0 - EYx0],
            'EYx1_Error': [Estimated_EYx1 - EYx1],
            'EYx0_sn_Error': [Estimated_EYx0_sn - EYx0],
            'EYx1_sn_Error': [Estimated_EYx1_sn - EYx1],
            'EYx1WVx0_Error': [Estimated_EYx1WVx0 - EYx1WVx0],
            'EYx1Vx0Wx1_Error': [Estimated_EYx1Vx0Wx1 - EYx1Vx0Wx1],
            'EYx1WVx0_sn_Error': [Estimated_EYx1WVx0_sn - EYx1WVx0],
            'EYx1Vx0Wx1_sn_Error': [Estimated_EYx1Vx0Wx1_sn - EYx1Vx0Wx1],
            'NDE_Error': [Estimated_NDE - NDE],
            'NIE_Error': [Estimated_NIE - NIE],
            'VDE_Error': [Estimated_VDE - VDE],
            'NDE_sn_Error': [Estimated_NDE_sn - NDE],
            'NIE_sn_Error': [Estimated_NIE_sn - NIE],
            'VDE_sn_Error': [Estimated_VDE_sn - VDE],
            # Relative errors
            'EYx0_Error%': [rel_err(Estimated_EYx0, EYx0)],
            'EYx1_Error%': [rel_err(Estimated_EYx1, EYx1)],
            'EYx0_sn_Error%': [rel_err(Estimated_EYx0_sn, EYx0)],
            'EYx1_sn_Error%': [rel_err(Estimated_EYx1_sn, EYx1)],
            'EYx1WVx0_Error%': [rel_err(Estimated_EYx1WVx0, EYx1WVx0)],
            'EYx1Vx0Wx1_Error%': [rel_err(Estimated_EYx1Vx0Wx1, EYx1Vx0Wx1)],
            'EYx1WVx0_sn_Error%': [rel_err(Estimated_EYx1WVx0_sn, EYx1WVx0)],
            'EYx1Vx0Wx1_sn_Error%': [rel_err(Estimated_EYx1Vx0Wx1_sn, EYx1Vx0Wx1)],
            'NDE_Error%': [rel_err(Estimated_NDE, NDE)],
            'NIE_Error%': [rel_err(Estimated_NIE, NIE)],
            'VDE_Error%': [rel_err(Estimated_VDE, VDE)],
            'NDE_sn_Error%': [rel_err(Estimated_NDE_sn, NDE)],
            'NIE_sn_Error%': [rel_err(Estimated_NIE_sn, NIE)],
            'VDE_sn_Error%': [rel_err(Estimated_VDE_sn, VDE)],
            # Sub-estimates
            'Estimated_NDE_pi2Y': [Estimated_NDE_pi2Y],
            'Estimated_NDE_pi2mu2': [Estimated_NDE_pi2mu2],
            'Estimated_NDE_pi1mu2': [Estimated_NDE_pi1mu2],
            'Estimated_NDE_pi1mu1': [Estimated_NDE_pi1mu1],
            'Estimated_NDE_mu1': [Estimated_NDE_mu1],
            'Estimated_NIE_pi2Y': [Estimated_NIE_pi2Y],
            'Estimated_NIE_pi2mu2': [Estimated_NIE_pi2mu2],
            'Estimated_NIE_pi1mu2': [Estimated_NIE_pi1mu2],
            'Estimated_NIE_pi1mu1': [Estimated_NIE_pi1mu1],
            'Estimated_NIE_mu1': [Estimated_NIE_mu1],
            'Estimated_VDE_pi3Y': [Estimated_VDE_pi3Y],
            'Estimated_VDE_pi3mu3': [Estimated_VDE_pi3mu3],
            'Estimated_VDE_pi2mu3': [Estimated_VDE_pi2mu3],
            'Estimated_VDE_pi2mu2': [Estimated_VDE_pi2mu2],
            'Estimated_VDE_pi1mu2': [Estimated_VDE_pi1mu2],
            'Estimated_VDE_pi1mu1': [Estimated_VDE_pi1mu1],
            'Estimated_VDE_mu1': [Estimated_VDE_mu1],
            # Sub-estimates (SN)
            'Estimated_NDE_pi2Y_sn': [Estimated_NDE_pi2Y_sn],
            'Estimated_NDE_pi2mu2_sn': [Estimated_NDE_pi2mu2_sn],
            'Estimated_NDE_pi1mu2_sn': [Estimated_NDE_pi1mu2_sn],
            'Estimated_NDE_pi1mu1_sn': [Estimated_NDE_pi1mu1_sn],
            'Estimated_NDE_mu1_sn': [Estimated_NDE_mu1_sn],
            'Estimated_NIE_pi2Y_sn': [Estimated_NIE_pi2Y_sn],
            'Estimated_NIE_pi2mu2_sn': [Estimated_NIE_pi2mu2_sn],
            'Estimated_NIE_pi1mu2_sn': [Estimated_NIE_pi1mu2_sn],
            'Estimated_NIE_pi1mu1_sn': [Estimated_NIE_pi1mu1_sn],
            'Estimated_NIE_mu1_sn': [Estimated_NIE_mu1_sn],
            'Estimated_VDE_pi3Y_sn': [Estimated_VDE_pi3Y_sn],
            'Estimated_VDE_pi3mu3_sn': [Estimated_VDE_pi3mu3_sn],
            'Estimated_VDE_pi2mu3_sn': [Estimated_VDE_pi2mu3_sn],
            'Estimated_VDE_pi2mu2_sn': [Estimated_VDE_pi2mu2_sn],
            'Estimated_VDE_pi1mu2_sn': [Estimated_VDE_pi1mu2_sn],
            'Estimated_VDE_pi1mu1_sn': [Estimated_VDE_pi1mu1_sn],
            'Estimated_VDE_mu1_sn': [Estimated_VDE_mu1_sn],
            # Sub-estimate errors
            'NDE_pi2Y_Error': [Estimated_NDE_pi2Y - NDE],
            'NDE_pi2mu2_Error': [Estimated_NDE_pi2mu2 - NDE],
            'NDE_pi1mu2_Error': [Estimated_NDE_pi1mu2 - NDE],
            'NDE_pi1mu1_Error': [Estimated_NDE_pi1mu1 - NDE],
            'NDE_mu1_Error': [Estimated_NDE_mu1 - NDE],
            'NIE_pi2Y_Error': [Estimated_NIE_pi2Y - NIE],
            'NIE_pi2mu2_Error': [Estimated_NIE_pi2mu2 - NIE],
            'NIE_pi1mu2_Error': [Estimated_NIE_pi1mu2 - NIE],
            'NIE_pi1mu1_Error': [Estimated_NIE_pi1mu1 - NIE],
            'NIE_mu1_Error': [Estimated_NIE_mu1 - NIE],
            'VDE_pi3Y_Error': [Estimated_VDE_pi3Y - VDE],
            'VDE_pi3mu3_Error': [Estimated_VDE_pi3mu3 - VDE],
            'VDE_pi2mu3_Error': [Estimated_VDE_pi2mu3 - VDE],
            'VDE_pi2mu2_Error': [Estimated_VDE_pi2mu2 - VDE],
            'VDE_pi1mu2_Error': [Estimated_VDE_pi1mu2 - VDE],
            'VDE_pi1mu1_Error': [Estimated_VDE_pi1mu1 - VDE],
            'VDE_mu1_Error': [Estimated_VDE_mu1 - VDE],
            # Sub-estimate errors (SN)
            'NDE_pi2Y_sn_Error': [Estimated_NDE_pi2Y_sn - NDE],
            'NDE_pi2mu2_sn_Error': [Estimated_NDE_pi2mu2_sn - NDE],
            'NDE_pi1mu2_sn_Error': [Estimated_NDE_pi1mu2_sn - NDE],
            'NDE_pi1mu1_sn_Error': [Estimated_NDE_pi1mu1_sn - NDE],
            'NDE_mu1_sn_Error': [Estimated_NDE_mu1_sn - NDE],
            'NIE_pi2Y_sn_Error': [Estimated_NIE_pi2Y_sn - NIE],
            'NIE_pi2mu2_sn_Error': [Estimated_NIE_pi2mu2_sn - NIE],
            'NIE_pi1mu2_sn_Error': [Estimated_NIE_pi1mu2_sn - NIE],
            'NIE_pi1mu1_sn_Error': [Estimated_NIE_pi1mu1_sn - NIE],
            'NIE_mu1_sn_Error': [Estimated_NIE_mu1_sn - NIE],
            'VDE_pi3Y_sn_Error': [Estimated_VDE_pi3Y_sn - VDE],
            'VDE_pi3mu3_sn_Error': [Estimated_VDE_pi3mu3_sn - VDE],
            'VDE_pi2mu3_sn_Error': [Estimated_VDE_pi2mu3_sn - VDE],
            'VDE_pi2mu2_sn_Error': [Estimated_VDE_pi2mu2_sn - VDE],
            'VDE_pi1mu2_sn_Error': [Estimated_VDE_pi1mu2_sn - VDE],
            'VDE_pi1mu1_sn_Error': [Estimated_VDE_pi1mu1_sn - VDE],
            'VDE_mu1_sn_Error': [Estimated_VDE_mu1_sn - VDE],
            #Sub-estimate relative errors
            'NDE_pi2Y_Error%': [rel_err(Estimated_NDE_pi2Y, NDE)],
            'NDE_pi2mu2_Error%': [rel_err(Estimated_NDE_pi2mu2, NDE)],
            'NDE_pi1mu2_Error%': [rel_err(Estimated_NDE_pi1mu2, NDE)],
            'NDE_pi1mu1_Error%': [rel_err(Estimated_NDE_pi1mu1, NDE)],
            'NDE_mu1_Error%': [rel_err(Estimated_NDE_mu1, NDE)],
            'NIE_pi2Y_Error%': [rel_err(Estimated_NIE_pi2Y, NIE)],
            'NIE_pi2mu2_Error%': [rel_err(Estimated_NIE_pi2mu2, NIE)],
            'NIE_pi1mu2_Error%': [rel_err(Estimated_NIE_pi1mu2, NIE)],
            'NIE_pi1mu1_Error%': [rel_err(Estimated_NIE_pi1mu1, NIE)],
            'NIE_mu1_Error%': [rel_err(Estimated_NIE_mu1, NIE)],
            'VDE_pi3Y_Error%': [rel_err(Estimated_VDE_pi3Y, VDE)],
            'VDE_pi3mu3_Error%': [rel_err(Estimated_VDE_pi3mu3, VDE)],
            'VDE_pi2mu3_Error%': [rel_err(Estimated_VDE_pi2mu3, VDE)],
            'VDE_pi2mu2_Error%': [rel_err(Estimated_VDE_pi2mu2, VDE)],
            'VDE_pi1mu2_Error%': [rel_err(Estimated_VDE_pi1mu2, VDE)],
            'VDE_pi1mu1_Error%': [rel_err(Estimated_VDE_pi1mu1, VDE)],
            'VDE_mu1_Error%': [rel_err(Estimated_VDE_mu1, VDE)],
            #Sub-estimate relative errors (SN)
            'NDE_pi2Y_sn_Error%': [rel_err(Estimated_NDE_pi2Y_sn, NDE)],
            'NDE_pi2mu2_sn_Error%': [rel_err(Estimated_NDE_pi2mu2_sn, NDE)],
            'NDE_pi1mu2_sn_Error%': [rel_err(Estimated_NDE_pi1mu2_sn, NDE)],
            'NDE_pi1mu1_sn_Error%': [rel_err(Estimated_NDE_pi1mu1_sn, NDE)],
            'NDE_mu1_sn_Error%': [rel_err(Estimated_NDE_mu1_sn, NDE)],
            'NIE_pi2Y_sn_Error%': [rel_err(Estimated_NIE_pi2Y_sn, NIE)],
            'NIE_pi2mu2_sn_Error%': [rel_err(Estimated_NIE_pi2mu2_sn, NIE)],
            'NIE_pi1mu2_sn_Error%': [rel_err(Estimated_NIE_pi1mu2_sn, NIE)],
            'NIE_pi1mu1_sn_Error%': [rel_err(Estimated_NIE_pi1mu1_sn, NIE)],
            'NIE_mu1_sn_Error%': [rel_err(Estimated_NIE_mu1_sn, NIE)],
            'VDE_pi3Y_sn_Error%': [rel_err(Estimated_VDE_pi3Y_sn, VDE)],
            'VDE_pi3mu3_sn_Error%': [rel_err(Estimated_VDE_pi3mu3_sn, VDE)],
            'VDE_pi2mu3_sn_Error%': [rel_err(Estimated_VDE_pi2mu3_sn, VDE)],
            'VDE_pi2mu2_sn_Error%': [rel_err(Estimated_VDE_pi2mu2_sn, VDE)],
            'VDE_pi1mu2_sn_Error%': [rel_err(Estimated_VDE_pi1mu2_sn, VDE)],
            'VDE_pi1mu1_sn_Error%': [rel_err(Estimated_VDE_pi1mu1_sn, VDE)],
            'VDE_mu1_sn_Error%': [rel_err(Estimated_VDE_mu1_sn, VDE)],
            # Nuisance parameters
            'Epi_x0': [Epi_x0],
            'Epi_x1': [Epi_x1],
            'Epi2_ne': [Epi2_ne],
            'Epi1_ne': [Epi1_ne],
            'Epi3_vde': [Epi3_vde],
            'Epi2_vde': [Epi2_vde],
            'Epi1_vde': [Epi1_vde],
        }
    else:
        # do(x) biased
        pi_x0_biased = (df[X] == 0) / (1 - df['px_wvz'])
        pi_x1_biased = (df[X] == 1) / (df['px_wvz'])
        hat_Yx0_biased = pi_x0_biased * (df[Y] - df['y_wvzx0']) + df['y_wvzx0']
        hat_Yx1_biased = pi_x1_biased * (df[Y] - df['y_wvzx1']) + df['y_wvzx1']
        Estimated_EYx0_biased = np.mean(hat_Yx0_biased)
        Estimated_EYx1_biased = np.mean(hat_Yx1_biased)

        # Total Effect
        Estimated_TE = Estimated_EYx1 - Estimated_EYx0
        Estimated_TE_sn = Estimated_EYx1_sn - Estimated_EYx0_sn
        Estimated_TE_biased = Estimated_EYx1_biased - Estimated_EYx0_biased

        # NIE* nuisance parameters
        pi2_star = (df[X] == 1) * (1 - df['px_vz']) / (df['px_vz'] * (1 - df['px_z']))
        pi1_star = (df[X] == 0) / (1 - df['px_z'])
        pi2_star_sn = pi2_star / np.mean(pi2_star)
        pi1_star_sn = pi1_star / np.mean(pi1_star)

        # NDE/NIE* standard
        Estimated_EYx1WVx0_pi2Y_star = np.mean((pi2_star * df[Y]))
        Estimated_EYx1WVx0_pi2mu2_star = np.mean((pi2_star * df['mu2_ne*']))
        Estimated_EYx1WVx0_pi1mu2_star = np.mean(pi1_star * df['mu2_ne*'])
        Estimated_EYx1WVx0_pi1mu1_star = np.mean(pi1_star * df['mu1_ne*'])
        Estimated_EYx1WVx0_mu1_star = np.mean(df['mu1_ne*'])
        Estimated_EYx1WVx0_star = (
            (Estimated_EYx1WVx0_pi2Y_star - Estimated_EYx1WVx0_pi2mu2_star) + 
            (Estimated_EYx1WVx0_pi1mu2_star - Estimated_EYx1WVx0_pi1mu1_star) + 
            Estimated_EYx1WVx0_mu1_star
        )
        Estimated_NDE_star = Estimated_EYx1WVx0_star - Estimated_EYx0
        Estimated_NIE_star = Estimated_EYx1 - Estimated_EYx1WVx0_star

        # NDE/NIE* self-normalized
        Estimated_EYx1WVx0_pi2Y_star_sn = np.mean((pi2_star_sn * df[Y]))
        Estimated_EYx1WVx0_pi2mu2_star_sn = np.mean((pi2_star_sn * df['mu2_ne*']))
        Estimated_EYx1WVx0_pi1mu2_star_sn = np.mean(pi1_star_sn * df['mu2_ne*'])
        Estimated_EYx1WVx0_pi1mu1_star_sn = np.mean(pi1_star_sn * df['mu1_ne*'])
        Estimated_EYx1WVx0_star_sn = (
            (Estimated_EYx1WVx0_pi2Y_star_sn - Estimated_EYx1WVx0_pi2mu2_star_sn) + 
            (Estimated_EYx1WVx0_pi1mu2_star_sn - Estimated_EYx1WVx0_pi1mu1_star_sn) + 
            Estimated_EYx1WVx0_mu1_star
        )
        Estimated_NDE_star_sn = Estimated_EYx1WVx0_star_sn - Estimated_EYx0_sn
        Estimated_NIE_star_sn = Estimated_EYx1_sn - Estimated_EYx1WVx0_star_sn

        # do(x) conditional
        Estimated_cond_EYx0 = np.mean(pi_x0 * (df[Y] - df['y_zx0'])) + df['y_zx0']
        Estimated_cond_EYx1 = np.mean(pi_x1 * (df[Y] - df['y_zx1'])) + df['y_zx1']
        Estimated_cond_EYx0_sn = np.mean(pi_x0_sn * (df[Y] - df['y_zx0'])) + df['y_zx0']
        Estimated_cond_EYx1_sn = np.mean(pi_x1_sn * (df[Y] - df['y_zx1'])) + df['y_zx1']

        # VDE conditional
        Estimated_cond_EYx1Vx0Wx1 = (
            (Estimated_EYx1Vx0Wx1_pi3Y - Estimated_EYx1Vx0Wx1_pi3mu3) + 
            (Estimated_EYx1Vx0Wx1_pi2mu3 - Estimated_EYx1Vx0Wx1_pi2mu2) + 
            (Estimated_EYx1Vx0Wx1_pi1mu2 - Estimated_EYx1Vx0Wx1_pi1mu1) +
            df['mu1_vde']
        )
        Estimated_cond_EYx1Vx0Wx1_sn = (
            (Estimated_EYx1Vx0Wx1_pi3Y_sn - Estimated_EYx1Vx0Wx1_pi3mu3_sn) + 
            (Estimated_EYx1Vx0Wx1_pi2mu3_sn - Estimated_EYx1Vx0Wx1_pi2mu2_sn) + 
            (Estimated_EYx1Vx0Wx1_pi1mu2_sn - Estimated_EYx1Vx0Wx1_pi1mu1_sn) +
            df['mu1_vde']
        )
        Estimated_cond_VDE = Estimated_cond_EYx1 - Estimated_cond_EYx1Vx0Wx1
        Estimated_cond_VDE_sn = Estimated_cond_EYx1_sn - Estimated_cond_EYx1Vx0Wx1_sn

        vcond_df = pd.DataFrame({
            'discrepancy': df['discrepancy'].squeeze(),
            'spo2': df['spo2'].squeeze(),
            'sao2': df['sao2'].squeeze(),
            'VDE': Estimated_cond_VDE,
            'VDE_sn': Estimated_cond_VDE_sn,
        })
        vcond_df['spo2_bin'] = np.ceil(vcond_df['spo2'] / 5) * 5
        vcond_df['sao2_bin'] = np.ceil(vcond_df['sao2'] / 5) * 5
        vcond_df.loc[vcond_df['spo2_bin'] == 70, 'spo2_bin'] = 75
        vcond_df.loc[vcond_df['sao2_bin'] == 70, 'sao2_bin'] = 75
        vcond_df['discrepancy_bin'] = np.ceil(vcond_df['discrepancy'] / 10) * 10
        vcond_df.loc[vcond_df['discrepancy_bin'] == -30, 'discrepancy_bin'] = -20
        discrepancy_spo2_cond_VDE = vcond_df.groupby(
            ['spo2_bin', 'discrepancy_bin'])['VDE'].mean().to_dict()
        discrepancy_sao2_cond_VDE = vcond_df.groupby(
            ['sao2_bin', 'discrepancy_bin'])['VDE'].mean().to_dict()
        discrepancy_spo2_cond_VDE_sn = vcond_df.groupby(
            ['spo2_bin', 'discrepancy_bin'])['VDE_sn'].mean().to_dict()
        discrepancy_sao2_cond_VDE_sn = vcond_df.groupby(
            ['sao2_bin', 'discrepancy_bin'])['VDE_sn'].mean().to_dict()

        results = {
            # Estimates
            'Estimated_EYx0': [Estimated_EYx0],
            'Estimated_EYx1': [Estimated_EYx1],
            'Estimated_EYx0_sn': [Estimated_EYx0_sn],
            'Estimated_EYx1_sn': [Estimated_EYx1_sn],
            'Estimated_EYx1WVx0': [Estimated_EYx1WVx0],
            'Estimated_EYx1Vx0Wx1': [Estimated_EYx1Vx0Wx1],
            'Estimated_EYx1WVx0*': [Estimated_EYx1WVx0_star],
            'Estimated_EYx1WVx0_sn': [Estimated_EYx1WVx0_sn],
            'Estimated_EYx1Vx0Wx1_sn': [Estimated_EYx1Vx0Wx1_sn],
            'Estimated_EYx1WVx0*_sn': [Estimated_EYx1WVx0_star_sn],
            'Estimated_TE': [Estimated_TE],
            'Estimated_NDE': [Estimated_NDE],
            'Estimated_NIE': [Estimated_NIE],
            'Estimated_VDE': [Estimated_VDE],
            'Estimated_TE_sn': [Estimated_TE_sn],
            'Estimated_NDE_sn': [Estimated_NDE_sn],
            'Estimated_NIE_sn': [Estimated_NIE_sn],
            'Estimated_VDE_sn': [Estimated_VDE_sn],
            'Estimated_NDE*': [Estimated_NDE_star],
            'Estimated_NIE*': [Estimated_NIE_star],
            'Estimated_NDE*_sn': [Estimated_NDE_star_sn],
            'Estimated_NIE*_sn': [Estimated_NIE_star_sn],
            'Estimated_TE_biased': [Estimated_TE_biased],
            # Sub-estimates
            'Estimated_NDE_pi2Y': [Estimated_NDE_pi2Y],
            'Estimated_NDE_pi2mu2': [Estimated_NDE_pi2mu2],
            'Estimated_NDE_pi1mu2': [Estimated_NDE_pi1mu2],
            'Estimated_NDE_pi1mu1': [Estimated_NDE_pi1mu1],
            'Estimated_NDE_mu1': [Estimated_NDE_mu1],
            'Estimated_NIE_pi2Y': [Estimated_NIE_pi2Y],
            'Estimated_NIE_pi2mu2': [Estimated_NIE_pi2mu2],
            'Estimated_NIE_pi1mu2': [Estimated_NIE_pi1mu2],
            'Estimated_NIE_pi1mu1': [Estimated_NIE_pi1mu1],
            'Estimated_NIE_mu1': [Estimated_NIE_mu1],
            'Estimated_VDE_pi3Y': [Estimated_VDE_pi3Y],
            'Estimated_VDE_pi3mu3': [Estimated_VDE_pi3mu3],
            'Estimated_VDE_pi2mu3': [Estimated_VDE_pi2mu3],
            'Estimated_VDE_pi2mu2': [Estimated_VDE_pi2mu2],
            'Estimated_VDE_pi1mu2': [Estimated_VDE_pi1mu2],
            'Estimated_VDE_pi1mu1': [Estimated_VDE_pi1mu1],
            'Estimated_VDE_mu1': [Estimated_VDE_mu1],
            # Sub-estimates (SN)
            'Estimated_NDE_pi2Y_sn': [Estimated_NDE_pi2Y_sn],
            'Estimated_NDE_pi2mu2_sn': [Estimated_NDE_pi2mu2_sn],
            'Estimated_NDE_pi1mu2_sn': [Estimated_NDE_pi1mu2_sn],
            'Estimated_NDE_pi1mu1_sn': [Estimated_NDE_pi1mu1_sn],
            'Estimated_NDE_mu1_sn': [Estimated_NDE_mu1_sn],
            'Estimated_NIE_pi2Y_sn': [Estimated_NIE_pi2Y_sn],
            'Estimated_NIE_pi2mu2_sn': [Estimated_NIE_pi2mu2_sn],
            'Estimated_NIE_pi1mu2_sn': [Estimated_NIE_pi1mu2_sn],
            'Estimated_NIE_pi1mu1_sn': [Estimated_NIE_pi1mu1_sn],
            'Estimated_NIE_mu1_sn': [Estimated_NIE_mu1_sn],
            'Estimated_VDE_pi3Y_sn': [Estimated_VDE_pi3Y_sn],
            'Estimated_VDE_pi3mu3_sn': [Estimated_VDE_pi3mu3_sn],
            'Estimated_VDE_pi2mu3_sn': [Estimated_VDE_pi2mu3_sn],
            'Estimated_VDE_pi2mu2_sn': [Estimated_VDE_pi2mu2_sn],
            'Estimated_VDE_pi1mu2_sn': [Estimated_VDE_pi1mu2_sn],
            'Estimated_VDE_pi1mu1_sn': [Estimated_VDE_pi1mu1_sn],
            'Estimated_VDE_mu1_sn': [Estimated_VDE_mu1_sn],
            # V-conditioned VDE
            'Estimated_discrepancy/spo2_conditional_VDE': [discrepancy_spo2_cond_VDE],
            'Estimated_discrepancy/sao2_conditional_VDE': [discrepancy_sao2_cond_VDE],
            'Estimated_discrepancy/spo2_conditional_VDE_sn': [discrepancy_spo2_cond_VDE_sn],
            'Estimated_discrepancy/sao2_conditional_VDE_sn': [discrepancy_sao2_cond_VDE_sn],
            # Nuisance parameters
            'Epi_x0': [Epi_x0],
            'Epi_x1': [Epi_x1],
            'Epi2_ne': [Epi2_ne],
            'Epi1_ne': [Epi1_ne],
            'Epi3_vde': [Epi3_vde],
            'Epi2_vde': [Epi2_vde],
            'Epi1_vde': [Epi1_vde],
        }
    return pd.DataFrame(results)

def fit(X, Y, binary=True, **kwargs):
    X = cp.asarray(X)
    Y = cp.asarray(Y)
    model_class = xgb.XGBClassifier if binary else xgb.XGBRegressor
    m = model_class(
        tree_method='hist',
        device='cuda:0',
        **kwargs
    ).fit(X, Y)
    return m

def pred(X, m, binary=True, clip=0.0, **kwargs):
    X = cp.asarray(X)
    if binary:
        y = m.predict_proba(X)[:, 1]
    if not binary:
        y = m.predict(X)
    if clip:
        y = np.clip(y, a_min=clip, a_max=1-clip)
    return y

def grid_search(X, Y, binary, hparams):
    model_class = xgb.XGBClassifier if binary else xgb.XGBRegressor
    scoring = 'neg_brier_score' if binary else 'neg_mean_squared_error'
    m = model_class(
        tree_method='hist',
        device='cuda:0',
        verbosity=0,
    )
    grid_search = GridSearchCV(
        estimator=m, param_grid=hparams,
        scoring=scoring, cv=5, verbose=0,
    )
    grid_search.fit(X, Y)
    return grid_search.best_params_

def rel_err(y_pred, y_true):
    if np.abs(y_true) < 1e-8:
        return None
    return 100 * (y_pred - y_true) / np.abs(y_true)