import os
import numpy as np
import pandas as pd
import pickle as pk
import xgboost as xgb
from estimation import fit, pred

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def inv_sigmoid(x):
    return np.log(x / (1 - x))

def shift_logit(x, shift):
    if x.ndim == 1:
        x = x[:, None]
    if shift.ndim == 1:
        shift = shift[:, None]
    return sigmoid(inv_sigmoid(x) + shift)

def sample_synth_binary(n, dims, seed=42):
    np.random.seed(seed)
    z_dim = dims['Z']
    x_dim = dims['X']
    w_dim = dims['W']
    v_dim = dims['V']
    y_dim = dims['Y']
    for v in ['Z', 'X', 'W', 'V', 'Y']:
        assert dims[v] == 1

    U_XZ = np.random.randn(n, z_dim)
    U_X = np.random.randn(n, x_dim)
    U_W = np.random.randn(n, w_dim)
    U_V = np.random.randn(n, v_dim)
    U_Y = np.random.randn(n, y_dim)
    Z = (U_XZ > 0.2).astype(float)
    X = ((Z + U_XZ + U_X) > 0.2).astype(float)
    W = ((X - Z + U_W) > 0.8).astype(float)
    V = ((X - Z + W + U_V) > 0.8).astype(float)
    Y = ((X - Z + 2 * (W - V) + U_Y) > 0.5).astype(float)

    Wx0 = ((0 - Z + U_W) > 0.8).astype(float)
    Wx1 = ((1 - Z + U_W) > 0.8).astype(float)
    Vx0 = ((0 - Z + Wx0 + U_V) > 0.8).astype(float)
    Vx1 = ((1 - Z + Wx1 + U_V) > 0.8).astype(float)
    Vx0Wx1 = ((0 - Z + Wx1 + U_V) > 0.8).astype(float)
    Yx1Vx0Wx1 = ((1 - Z + 2 * (Wx1 - Vx0Wx1) + U_Y) > 0.5).astype(float)
    Yx1WVx0 = ((1 - Z + 2 * (Wx0 - Vx0) + U_Y) > 0.5).astype(float)
    Yx0 = ((0 - Z + 2 * (Wx0 - Vx0) + U_Y) > 0.5).astype(float)
    Yx1 = ((1 - Z + 2 * (Wx1 - Vx1) + U_Y) > 0.5).astype(float)

    df = pd.DataFrame({
        'X': X.flatten(),
        'W': W.flatten(),
        'V': V.flatten(),
        'Z': Z.flatten(),
        'Y': Y.flatten(),
        'Y[x0]': Yx0.flatten(),
        'Y[x1]': Yx1.flatten(),
        'Y[x1WVx0]': Yx1WVx0.flatten(),
        'Y[x1Vx0Wx1]': Yx1Vx0Wx1.flatten(),
    })
    return df

def sample_synth_continuous(n, dims, seed=42):
    np.random.seed(42)
    z_dim = dims['Z']
    x_dim = dims['X']
    w_dim = dims['W']
    v_dim = dims['V']
    y_dim = dims['Y']
    alpha_xz = np.random.randn(2 * z_dim, 1)
    alpha_wxz = np.random.randn(z_dim + x_dim, w_dim)
    alpha_vxwz = np.random.randn(z_dim + w_dim + x_dim, v_dim)
    alpha_yxwvz = np.random.randn(z_dim + w_dim + v_dim + x_dim, y_dim)

    alpha_yxwvz[:x_dim] *= 5
    alpha_wxz[x_dim:x_dim + z_dim] /= np.sqrt(z_dim)
    alpha_vxwz[x_dim:x_dim + w_dim] /= np.sqrt(w_dim)
    alpha_yxwvz[x_dim:x_dim + w_dim] /= np.sqrt(w_dim)
    alpha_vxwz[x_dim + w_dim:x_dim + w_dim + z_dim] /= np.sqrt(z_dim)
    alpha_yxwvz[x_dim + w_dim:x_dim + w_dim + v_dim] /= np.sqrt(1 / 5)
    alpha_yxwvz[x_dim + w_dim + v_dim:x_dim + w_dim + v_dim + z_dim] /= np.sqrt(z_dim)

    np.random.seed(seed)
    W_cols = [f'W{i+1}' for i in range(w_dim)] if w_dim > 1 else ['W']
    V_cols = [f'V{i+1}' for i in range(v_dim)] if v_dim > 1 else ['V']
    Z_cols = [f'Z{i+1}' for i in range(z_dim)] if z_dim > 1 else ['Z']

    U_XZ = np.random.randn(n, z_dim)
    U_X = np.random.randn(n, x_dim)
    U_W = np.random.randn(n, w_dim)
    U_V = np.random.randn(n, v_dim)
    U_Y = np.random.randn(n, y_dim)

    Z = U_XZ - 1
    X = sigmoid(np.hstack([Z, U_XZ]) @ alpha_xz + U_X) > 0.5
    W = np.hstack([X, Z]) @ alpha_wxz + U_W
    V = np.hstack([X, W, Z]) @ alpha_vxwz + U_V
    Y = sigmoid(np.hstack([X, W, V, Z]) @ alpha_yxwvz + U_Y) > 0.2
    X = X.astype(int)
    Y = Y.astype(int)

    X0 = np.zeros_like(X)
    X1 = np.ones_like(X)
    Wx0 = np.hstack([X0, Z]) @ alpha_wxz + U_W
    Wx1 = np.hstack([X1, Z]) @ alpha_wxz + U_W
    Vx0 = np.hstack([X0, Wx0, Z]) @ alpha_vxwz + U_V
    Vx1 = np.hstack([X1, Wx1, Z]) @ alpha_vxwz + U_V
    Vx0Wx1 = np.hstack([X0, Wx1, Z]) @ alpha_vxwz + U_V
    Yx1Vx0Wx1 = sigmoid(np.hstack([X1, Wx1, Vx0Wx1, Z]) @ alpha_yxwvz + U_Y) > 0.2
    Yx1WVx0 = sigmoid(np.hstack([X1, Wx0, Vx0, Z]) @ alpha_yxwvz + U_Y) > 0.2
    Yx0 = sigmoid(np.hstack([X0, Wx0, Vx0, Z]) @ alpha_yxwvz + U_Y) > 0.2
    Yx1 = sigmoid(np.hstack([X1, Wx1, Vx1, Z]) @ alpha_yxwvz + U_Y) > 0.2
    Yx1Vx0Wx1 = Yx1Vx0Wx1.astype(int)
    Yx1WVx0 = Yx1WVx0.astype(int)
    Yx0 = Yx0.astype(int)
    Yx1 = Yx1.astype(int)

    data = {'X': X.flatten()}
    for i,c in enumerate(W_cols):
        data[c] = W[:, i]
    for i,c in enumerate(V_cols):
        data[c] = V[:, i]
    for i,c in enumerate(Z_cols):
        data[c] = Z[:, i]
    data.update({
        'Y': Y.flatten(),
        'Y[x0]': Yx0.flatten(),
        'Y[x1]': Yx1.flatten(),
        'Y[x1WVx0]': Yx1WVx0.flatten(),
        'Y[x1Vx0Wx1]': Yx1Vx0Wx1.flatten(),
    })
    df = pd.DataFrame(data)
    return df

def fit_semisynth(dataset, force_refit=True):
    assert dataset in ['eicu', 'mimic']
    df = load_eicu() if dataset == 'eicu' else load_mimic()
    vars = load_vars()
    X = vars['X']
    Y = 'vented'
    W_cols = vars['W']
    V_cols = vars['V']
    Z_cols = vars['Z']
    
    dtypes = {X: int, Y: int}
    df['const'] = 0
    for Z in Z_cols:
        values = df[Z]
        values = values[values.notna()]
        isint = (values.astype(float) == values.astype(int)).all()
        dtypes[Z] = int if isint.all() else float
    for W in W_cols:
        values = df[W]
        values = values[values.notna()]
        isint = (values.astype(float) == values.astype(int)).all()
        dtypes[W] = int if isint.all() else float
    for V in V_cols:
        isint = (df[V].astype(float) == df[V].astype(int)).all()
        dtypes[V] = int if isint.all() else float
        path = f'semisynth/{dataset}_models.pkl'
    if os.path.exists(path) and not force_refit:
        print(f'{path} exists and force_refit is False')
        return vars, dtypes

    models = {}
    np.random.seed(42)
    fit_kwargs = {'n_estimators': 20, 'max_depth': 4}
    px_z = fit(df[Z_cols], df[X], **fit_kwargs)
    train_df = df[[X, Y] + W_cols + V_cols + Z_cols]
    train_df = train_df[train_df[Y].notna()]
    py_xwvz = fit(
        train_df[[X] + W_cols + V_cols + Z_cols], 
        train_df[Y], **fit_kwargs
    )
    models[Y] = py_xwvz
    models[X] = px_z
    for Z in Z_cols:
        binary = (Z == 'sex')
        train_df = df[[Z, 'const']]
        train_df = train_df[train_df[Z].notna()]
        models[Z] = fit(
            train_df[['const']], train_df[Z],
            binary=binary, **fit_kwargs
        )
    for W in W_cols:
        train_df = df[[X, W] + Z_cols]
        train_df = train_df[train_df[W].notna()]
        models[W] = fit(
            train_df[[X] + Z_cols], train_df[W], 
            binary=False, **fit_kwargs
        )
    for V in V_cols:
        train_df = df[[X, V] + W_cols + Z_cols]
        train_df = train_df[train_df[V].notna()]
        models[V] = fit(
            train_df[[X] + W_cols + Z_cols], train_df[V], 
            binary=False, **fit_kwargs
        )
    pk.dump(models, open(path, 'wb'))
    return vars, dtypes

def sample_semisynth(dataset, n, vars, dtypes, seed=42, **kwargs):
    assert dataset in ['eicu', 'mimic']
    models = pk.load(open(f'semisynth/{dataset}_models.pkl', 'rb'))
    df = load_eicu() if dataset == 'eicu' else load_mimic()
    df = df.reindex(range(n), fill_value=0)
    df['const'] = 0
    df[:] = 0
    
    np.random.seed(seed)
    X = vars['X']
    Y = 'vented'
    W_cols = vars['W']
    V_cols = vars['V']
    Z_cols = vars['Z']
    z_dim = len(Z_cols)
    v_dim = len(V_cols)
    w_dim = len(W_cols)
    y_dim = 1
    x_dim = 1

    U_Z = 5 * np.random.randn(n, z_dim)
    U_X = 5 * np.random.randn(n, x_dim)
    U_W = 1.2 * np.random.randn(n, w_dim)
    U_V = 0.2 * np.random.randn(n, v_dim)
    U_Y = 1 * np.random.randn(n, y_dim)

    X0 = X + '[x0]'
    X1 = X + '[x1]'
    Yx0 = Y + '[x0]'
    Yx1 = Y + '[x1]'
    Yx1WVx0 = Y + '[x1WVx0]'
    Yx1Vx0Wx1 = Y + '[x1Vx0Wx1]'
    Wx0_cols = [W + '[x0]' for W in W_cols]
    Wx1_cols = [W + '[x1]' for W in W_cols]
    Vx0_cols = [V + '[x0]' for V in V_cols]
    Vx1_cols = [V + '[x1]' for V in V_cols]
    Vx0Wx1_cols = [V + '{x0Wx1}' for V in V_cols]

    new_cols = {}
    new_cols[X0] = None
    new_cols[X1] = None
    for Wx0 in Wx0_cols:
        new_cols[Wx0] = None
    for Wx1 in Wx1_cols:
        new_cols[Wx1] = None
    for Vx0 in Vx0_cols:
        new_cols[Vx0] = None
    for Vx1 in Vx1_cols:
        new_cols[Vx1] = None
    for Vx0Wx1 in Vx0Wx1_cols:
        new_cols[Vx0Wx1] = None
    new_cols[Yx0] = None
    new_cols[Yx1] = None
    new_cols[Yx1WVx0] = None
    new_cols[Yx1Vx0Wx1] = None
    new_cols_df = pd.DataFrame(new_cols, index=df.index)
    df = pd.concat([df, new_cols_df], axis=1)

    for i, Z in enumerate(Z_cols):
        binary = (Z == 'sex')
        predZ = pred(
            df[[X]], models[Z],
            binary=binary,
        )
        if binary:
            predZ = shift_logit(predZ, U_Z[:, i])
            df[Z] = (predZ > 0.5).astype(dtypes[Z])
        else:
            predZ += U_Z[:, i]
            df[Z] = predZ.astype(dtypes[Z])
    
    predX = pred(df[Z_cols], models[X])
    predX = shift_logit(predX, U_X)
    df[X] = (predX > 0.5).astype(dtypes[X])
    df[X + '[x0]'] = 0
    df[X + '[x1]'] = 1

    for i, W in enumerate(W_cols):
        Wx0 = Wx0_cols[i]
        Wx1 = Wx1_cols[i]
        predW = pred(df[[X] + Z_cols], models[W], binary=False) + U_W[:, i]
        predWx0 = pred(df[[X0] + Z_cols], models[W], binary=False) + U_W[:, i]
        predWx1 = pred(df[[X1] + Z_cols], models[W], binary=False) + U_W[:, i]
        df[W] = predW.astype(dtypes[W])
        df[Wx0] = predWx0.astype(dtypes[W])
        df[Wx1] = predWx1.astype(dtypes[W])

    for i, V in enumerate(V_cols):
        Vx0 = Vx0_cols[i]
        Vx1 = Vx1_cols[i]
        Vx0Wx1 = Vx0Wx1_cols[i]
        predV = pred(df[[X] + W_cols + Z_cols], models[V], binary=False) + U_V[:, i]
        predVx0 = pred(df[[X0] + Wx0_cols + Z_cols], models[V], binary=False) + U_V[:, i]
        predVx1 = pred(df[[X1] + Wx1_cols + Z_cols], models[V], binary=False) + U_V[:, i]
        predVx0Wx1 = pred(df[[X0] + Wx1_cols + Z_cols], models[V], binary=False) + U_V[:, i]
        df[V] = predV.astype(dtypes[V])
        df[Vx0] = predVx0.astype(dtypes[V])
        df[Vx1] = predVx1.astype(dtypes[V])
        df[Vx0Wx1] = predVx0Wx1.astype(dtypes[V])

    predY = pred(df[[X] + W_cols + V_cols + Z_cols], models[Y])
    predYx0 = pred(df[[X0] + Wx0_cols + Vx0_cols + Z_cols], models[Y])
    predYx1 = pred(df[[X1] + Wx1_cols + Vx1_cols + Z_cols], models[Y])
    predYx1WVx0 = pred(df[[X1] + Wx0_cols + Vx0_cols + Z_cols], models[Y])
    predYx1Vx0Wx1 = pred(df[[X1] + Wx1_cols + Vx0Wx1_cols + Z_cols], models[Y])
    predY = shift_logit(predY, U_Y)
    predYx0 = shift_logit(predYx0, U_Y)
    predYx1 = shift_logit(predYx1, U_Y)
    predYx1WVx0 = shift_logit(predYx1WVx0, U_Y)
    predYx1Vx0Wx1 = shift_logit(predYx1Vx0Wx1, U_Y)
    df[Y] = (predY > 0.5).astype(dtypes[Y])
    df[Yx0] = (predYx0 > 0.5).astype(dtypes[Y])
    df[Yx1] = (predYx1 > 0.5).astype(dtypes[Y])
    df[Yx1WVx0] = (predYx1WVx0 > 0.5).astype(dtypes[Y])
    df[Yx1Vx0Wx1] = (predYx1Vx0Wx1 > 0.5).astype(dtypes[Y])
    df = df.drop(columns=['const'])
    return df

def load(path):
    df = pd.read_csv(path)
    if 'ethnicity' in df.columns:
        df = df.rename(columns={'ethnicity': 'race'})
    df = df.rename(columns={'gender': 'sex'})
    df = df[(df['race'] == 0) | (df['race'] == 2)]
    df.loc[:, 'race'] = df['race'].map({0: 1, 2:0})
    df = df[(df['spo2_ewa'] >= 70) & (df['spo2_ewa'] <= 100)]
    df = df[(df['sao2_ewa'] >= 70) & (df['sao2_ewa'] <= 100)]
    for c in df.columns:
        if c.endswith('_ewa'):
            df.rename(columns={c: c[:-4]}, inplace=True)
    df['discrepancy'] = df['discrepancy'].clip(lower=-30, upper=30)

    vars = load_vars()
    X = vars['X']
    W_cols = vars['W']
    V_cols = vars['V']
    Z_cols = vars['Z']
    df = df[
        [X, 'vented', 'vent_duration']
        + Z_cols + W_cols + V_cols
    ]
    return df

def load_eicu():
    return load('../data/eicu/cohort.csv')

def load_mimic():
    df = load('../data/mimic-iv/cohort.csv')
    df['vent_duration'] /= 60
    return df

def load_vars():
    Z_cols = ['sex', 'age', 'charlson', 'pre_icu_los_oasis']
    V_cols = ['sao2', 'spo2', 'discrepancy']
    W_cols = [
        'ph', 'paco2', 'pao2', 'bicarbonate', 'co2',
        'carboxyhemoglobin', 'chloride', 'creatinine', 'glucose', 'hct',
        'hgb', 'lactate', 'methemoglobin', 'potassium', 'sodium', 'respiration',
        'temperature', 'heartrate', 'age_oasis', 'gcs_oasis', 'heartrate_oasis',
        'map_oasis', 'respiratoryrate_oasis', 'temperature_oasis',
        'urineoutput_oasis', 'vent_oasis', 'electivesurgery_oasis', 'oasis',
    ]
    vars = {
        'X': 'race', 
        'W': W_cols,
        'V': V_cols,
        'Z': Z_cols,
    }
    return vars

if __name__ == '__main__':
    make_dir('./semisynth/')
    print('Validating synthetic binary sampler')
    synth_binary_dims = {'Z': 1, 'X': 1, 'W': 1, 'V': 1, 'Y': 1}
    synth_binary_df = sample_synth_binary(10000, synth_binary_dims)
    print('Validating synthetic continuous sampler')
    synth_continuous_dims = {'Z': 3, 'X': 1, 'W': 10, 'V': 3, 'Y': 1}
    synth_continuous_df = sample_synth_continuous(10000, synth_continuous_dims)
    print('Fitting + validating semi-synthetic eICU sampler')
    vars, dtypes = fit_semisynth('eicu', force_refit=False)
    semisynth_eicu_df = sample_semisynth('eicu', 10000, vars, dtypes)
    print('Fitting + validating semi-synthetic MIMIC-IV sampler')
    vars, dtypes = fit_semisynth('mimic', force_refit=False)
    semisynth_mimic_df = sample_semisynth('mimic', 10000, vars, dtypes)
    print('Binary     synthetic     :', synth_binary_df.shape)
    print('Continuous synthetic     :', synth_continuous_df.shape)
    print('eICU       semi-synthetic:', semisynth_eicu_df.shape)
    print('MIMIC      semi-synthetic:', semisynth_mimic_df.shape)