import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from collections import Counter

import lightgbm as lgb
import xgboost as xgb

from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    fbeta_score, precision_recall_curve, auc, brier_score_loss,
    accuracy_score, matthews_corrcoef, confusion_matrix
)
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

# ------------------------------------------------
# Utility
# ------------------------------------------------
def align_to_schema(df: pd.DataFrame, schema_cols, fill_value=0):
    df2 = df.copy()
    for c in schema_cols:
        if c not in df2.columns: df2[c] = fill_value
    return df2[[c for c in schema_cols]].drop(
        columns=[c for c in df2.columns if c not in schema_cols], errors='ignore')

def apply_sampling_method(X, y, method='original'):
    if method == 'original': return X, y
    if method == 'rus':      return RandomUnderSampler(random_state=42).fit_resample(X, y)
    if method == 'smoteenn': return SMOTEENN(random_state=42).fit_resample(X, y)
    raise ValueError(f"Unknown sampling method: {method}")

def find_optimal_threshold_for_f1(y_true, proba):
    p, r, t = precision_recall_curve(y_true, proba)
    f1 = 2*p*r/(p+r+1e-12); f1 = f1[:-1]
    if len(f1) == 0: return 0.5, 0.0
    i = int(np.argmax(f1))
    return float(t[i]), float(f1[i])

def find_thr_for_fbeta(y_true, proba, beta=2):
    p, r, t = precision_recall_curve(y_true, proba)
    f = (1+beta**2)*p*r/(beta**2*p + r + 1e-12); f = f[:-1]
    if len(f) == 0: return 0.5, 0.0
    i = int(np.nanargmax(f))
    return float(t[i]), float(f[i])

def calculate_auprc(y_true, y_pred):
    p, r, _ = precision_recall_curve(y_true, y_pred)
    return auc(r, p)

def _bootstrap_two_sided_pvalue(deltas):
    deltas = np.asarray(deltas, float)
    return float(2 * min(np.mean(deltas <= 0.0), np.mean(deltas >= 0.0)))

def split_train_val_test(X, y, test_size=0.2, val_size=0.2, random_state=42, stratify=True):
    kw = dict(test_size=test_size, random_state=random_state)
    if stratify: kw['stratify'] = y
    X_trv, X_te, y_trv, y_te = train_test_split(X, y, **kw)
    val_ratio = val_size / (1.0 - test_size)
    kw2 = dict(test_size=val_ratio, random_state=random_state)
    if stratify: kw2['stratify'] = y_trv
    X_tr, X_va, y_tr, y_va = train_test_split(X_trv, y_trv, **kw2)
    return X_tr, X_va, X_te, y_tr, y_va, y_te

# ------------------------------------------------
# LightGBM
# ------------------------------------------------
def lgb_f1_metric(preds, train_data):
    y = train_data.get_label()
    _, f1 = find_optimal_threshold_for_f1(y, preds)
    return ('f1_opt', f1, True)

def lgb_auprc_metric(preds, train_data):
    return ('auprc', calculate_auprc(train_data.get_label(), preds), True)

def train_lightgbm_model(X_train, y_train, X_val, y_val, optimize_for='auprc'):
    if optimize_for == 'f1':
        params = {
            'objective':'binary','metric':'None','boosting_type':'gbdt',
            'num_leaves':63,'learning_rate':0.08,'feature_fraction':0.85,
            'bagging_fraction':0.85,'bagging_freq':5,'min_child_samples':10,
            'reg_alpha':0.1,'reg_lambda':0.1,'random_state':42,'is_unbalance':True,'verbose':-1
        }
        feval = lgb_f1_metric
    else:
        params = {
            'objective':'binary','metric':'None','boosting_type':'gbdt',
            'num_leaves':31,'learning_rate':0.06,'feature_fraction':0.9,
            'bagging_fraction':0.8,'bagging_freq':5,'min_child_samples':20,
            'reg_alpha':0.05,'reg_lambda':0.05,'random_state':42,'is_unbalance':True,'verbose':-1
        }
        feval = lgb_auprc_metric

    tr = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    va = lgb.Dataset(X_val,   label=y_val,   reference=tr, free_raw_data=False)
    model = lgb.train(
        params, tr, valid_sets=[tr, va], num_boost_round=1500, feval=feval,
        callbacks=[lgb.early_stopping(80, first_metric_only=True), lgb.log_evaluation(0)]
    )
    p_val = model.predict(X_val, num_iteration=model.best_iteration)
    model.optimal_threshold, _ = find_optimal_threshold_for_f1(y_val, p_val)
    return model

def soft_transfer_learning_lightgbm(pretrained, X_tr, y_tr, X_va, y_va,
                                    alpha=0.4, optimize_for='auprc'):
    feval = lgb_f1_metric if optimize_for == 'f1' else lgb_auprc_metric
    ft_params = pretrained.params.copy()
    ft_params.update({
        'learning_rate': 0.02 if optimize_for=='f1' else 0.015,
        'num_leaves':    63   if optimize_for=='f1' else 31,
        'min_child_samples': 8 if optimize_for=='f1' else 15,
        'reg_alpha': 0.08 if optimize_for=='f1' else 0.05,
        'reg_lambda': 0.08 if optimize_for=='f1' else 0.05,
        'metric': 'None'
    })
    tr = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
    va = lgb.Dataset(X_va, label=y_va, reference=tr, free_raw_data=False)
    fine = lgb.train(ft_params, tr, valid_sets=[tr, va], num_boost_round=100,
                     init_model=pretrained, feval=feval,
                     callbacks=[lgb.early_stopping(25, first_metric_only=True), lgb.log_evaluation(0)])
    hosp_params = ft_params.copy()
    hosp_params.update({
        'num_leaves':   40 if optimize_for=='f1' else 25,
        'learning_rate': 0.1 if optimize_for=='f1' else 0.08,
        'is_unbalance': True
    })
    hosp = lgb.train(hosp_params, tr, valid_sets=[tr, va], num_boost_round=80,
                     feval=feval,
                     callbacks=[lgb.early_stopping(20, first_metric_only=True), lgb.log_evaluation(0)])

    class LGBEnsemble:
        def __init__(self, m1, m2, m3, a, b):
            self.m1, self.m2, self.m3 = m1, m2, m3
            s = a + b + max(0.0, 1-a-b)
            self.a, self.b, self.c = a/s, b/s, max(0.0,1-a-b)/s
            self.optimal_threshold = 0.5
        def predict_proba(self, X):
            return (self.a * self.m1.predict(X, num_iteration=self.m1.best_iteration)
                  + self.b * self.m2.predict(X, num_iteration=self.m2.best_iteration)
                  + self.c * self.m3.predict(X, num_iteration=self.m3.best_iteration))
        def fit_threshold(self, Xv, yv):
            self.optimal_threshold, _ = find_optimal_threshold_for_f1(yv, self.predict_proba(Xv))

    ens = LGBEnsemble(pretrained, fine, hosp, alpha, 0.35)
    ens.fit_threshold(X_va, y_va)
    return ens

def regularized_transfer_learning_lightgbm(pretrained, X_tr, y_tr, X_va, y_va,
                                           reg_lambda=1.0, optimize_for='auprc'):
    feval = lgb_f1_metric if optimize_for == 'f1' else lgb_auprc_metric
    params = pretrained.params.copy()
    if optimize_for == 'f1':
        params.update({
            'learning_rate':0.03,'reg_alpha':0.3,'reg_lambda':reg_lambda*0.8,
            'min_data_in_leaf':max(8,len(X_tr)//80),'feature_fraction':0.85,
            'bagging_fraction':0.85,'bagging_freq':5,'num_leaves':50,
            'is_unbalance':True,'metric':'None'
        })
    else:
        params.update({
            'learning_rate':0.025,'reg_alpha':0.2,'reg_lambda':reg_lambda*0.6,
            'min_data_in_leaf':max(10,len(X_tr)//100),'feature_fraction':0.8,
            'bagging_fraction':0.8,'bagging_freq':5,'num_leaves':35,
            'is_unbalance':True,'metric':'None'
        })
    tr = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
    va = lgb.Dataset(X_va, label=y_va, reference=tr, free_raw_data=False)
    model = lgb.train(params, tr, valid_sets=[tr, va], num_boost_round=120,
                      init_model=pretrained, feval=feval,
                      callbacks=[lgb.early_stopping(30, first_metric_only=True), lgb.log_evaluation(0)])
    p_va = model.predict(X_va, num_iteration=model.best_iteration)
    model.optimal_threshold, _ = find_optimal_threshold_for_f1(y_va, p_va)
    return model

# ------------------------------------------------
# XGBoost
# ------------------------------------------------
def _xgb_best_end(booster):
    try:
        bi = booster.attr("best_iteration")
        if bi is not None: return int(bi) + 1
    except Exception: pass
    if hasattr(booster, "best_iteration"):
        try: return int(booster.best_iteration) + 1
        except Exception: pass
    return None

def xgb_predict_proba(booster, X):
    dm  = xgb.DMatrix(X)
    end = _xgb_best_end(booster)
    return booster.predict(dm) if end is None else booster.predict(dm, iteration_range=(0, end))

def _ap(y, p):
    prc, rec, _ = precision_recall_curve(y, p)
    return auc(rec, prc)

def _xgb_train_one(params, dtr, dva, num_boost_round, early_stopping_rounds, xgb_model=None):
    return xgb.train(params, dtr, num_boost_round=num_boost_round,
                     evals=[(dtr,'train'),(dva,'valid')],
                     early_stopping_rounds=early_stopping_rounds,
                     verbose_eval=False, xgb_model=xgb_model)

def _score_xgb(booster, X_val, y_val, optimize_for):
    p  = xgb_predict_proba(booster, X_val)
    ap = _ap(y_val, p); th, f1 = find_optimal_threshold_for_f1(y_val, p); br = brier_score_loss(y_val, p)
    key = (f1, ap, -br) if optimize_for == 'f1' else (ap, f1, -br)
    return dict(key=key, th=th, f1=f1, auprc=ap, brier=br)

def _xgb_small_grid(param_template, dtr, dva, X_val, y_val,
                    optimize_for='auprc', max_trials=24, random_state=42,
                    xgb_model=None, num_boost_round=1500, early_stopping_rounds=80):
    from itertools import product
    rng   = np.random.default_rng(random_state)
    space = list(product([4,5,6],[2,3,4,5],[0.6,0.75,0.9],[0.6,0.75,0.9]))
    best  = {'key': (-1,-1,+1e9)}
    for md, mcw, sub, col in [space[i] for i in rng.permutation(len(space))[:max_trials]]:
        p = param_template.copy()
        p.update({'max_depth':md,'min_child_weight':mcw,'subsample':sub,'colsample_bytree':col})
        b = _xgb_train_one(p, dtr, dva, num_boost_round, early_stopping_rounds, xgb_model)
        s = _score_xgb(b, X_val, y_val, optimize_for)
        if s['key'] > best['key']: best = {**s, 'booster':b}
    best['booster'].optimal_threshold = float(best['th'])
    return best['booster']

def _xgb_base_params(y_train=None, optimize_for='auprc'):
    params = {
        'objective':'binary:logistic','eval_metric':'aucpr','tree_method':'hist',
        'learning_rate': 0.08 if optimize_for=='f1' else 0.06,
        'max_depth':     7    if optimize_for=='f1' else 6,
        'subsample':     0.85 if optimize_for=='f1' else 0.8,
        'colsample_bytree': 0.85 if optimize_for=='f1' else 0.8,
        'min_child_weight': 3.0 if optimize_for=='f1' else 2.0,
        'lambda': 1.5 if optimize_for=='f1' else 1.0,
        'alpha':  0.1 if optimize_for=='f1' else 0.05,
        'random_state':42,'seed':42,'verbosity':0
    }
    if y_train is not None:
        pos = int(np.sum(np.asarray(y_train)==1))
        neg = int(np.sum(np.asarray(y_train)==0))
        if pos > 0: params['scale_pos_weight'] = neg / max(pos,1)
    return params

def train_xgb_model(X_train, y_train, X_val, y_val, optimize_for='auprc', grid_trials=24):
    dtr = xgb.DMatrix(X_train, label=y_train)
    dva = xgb.DMatrix(X_val,   label=y_val)
    return _xgb_small_grid(_xgb_base_params(y_train, optimize_for),
                           dtr, dva, X_val, y_val,
                           optimize_for=optimize_for, max_trials=grid_trials,
                           num_boost_round=1500, early_stopping_rounds=80)

def soft_transfer_learning_xgb(pretrained, X_tr, y_tr, X_va, y_va,
                                alpha=0.4, optimize_for='auprc'):
    dtr = xgb.DMatrix(X_tr, label=y_tr)
    dva = xgb.DMatrix(X_va, label=y_va)
    ft_p = _xgb_base_params(y_tr, optimize_for)
    ft_p.update({'learning_rate':0.05 if optimize_for=='f1' else 0.04,
                 'max_depth':7 if optimize_for=='f1' else 6,
                 'lambda':2.0 if optimize_for=='f1' else 1.5,
                 'alpha':0.15 if optimize_for=='f1' else 0.1})
    ft_model   = _xgb_small_grid(ft_p, dtr, dva, X_va, y_va, optimize_for, 12, xgb_model=pretrained,
                                  num_boost_round=200, early_stopping_rounds=30)
    hosp_p     = _xgb_base_params(y_tr, optimize_for)
    if optimize_for == 'f1': hosp_p.update({'learning_rate':0.1,'max_depth':6,'lambda':1.0})
    hosp_model = _xgb_small_grid(hosp_p, dtr, dva, X_va, y_va, optimize_for, 12,
                                  num_boost_round=150, early_stopping_rounds=25)

    class XGBEnsemble:
        def __init__(self, pre, ft, hosp, a, b):
            s = a + b + max(0.0,1-a-b)
            self.pre, self.ft, self.hosp = pre, ft, hosp
            self.a, self.b, self.c = a/s, b/s, max(0.0,1-a-b)/s
            self.optimal_threshold = 0.5
        def predict_proba(self, X):
            return (self.a * xgb_predict_proba(self.pre,  X)
                  + self.b * xgb_predict_proba(self.ft,   X)
                  + self.c * xgb_predict_proba(self.hosp, X))
        def fit_threshold(self, Xv, yv):
            self.optimal_threshold, _ = find_optimal_threshold_for_f1(yv, self.predict_proba(Xv))

    ens = XGBEnsemble(pretrained, ft_model, hosp_model, alpha, 0.35)
    ens.fit_threshold(X_va, y_va)
    return ens

def regularized_transfer_learning_xgb(pretrained, X_tr, y_tr, X_va, y_va,
                                      reg_lambda=2.0, optimize_for='auprc'):
    params = _xgb_base_params(y_tr, optimize_for)
    if optimize_for == 'f1':
        params.update({'lambda':reg_lambda*1.2,'alpha':0.3,'learning_rate':0.04,'max_depth':7,'min_child_weight':4})
    else:
        params.update({'lambda':reg_lambda,'alpha':0.2,'learning_rate':0.03,'max_depth':6,'min_child_weight':3})
    dtr = xgb.DMatrix(X_tr, label=y_tr)
    dva = xgb.DMatrix(X_va, label=y_va)
    model = _xgb_small_grid(params, dtr, dva, X_va, y_va, optimize_for, 16,
                             xgb_model=pretrained, num_boost_round=200, early_stopping_rounds=35)
    model.optimal_threshold, _ = find_optimal_threshold_for_f1(y_va, xgb_predict_proba(model, X_va))
    return model

# ------------------------------------------------
# Helper: get probability from any model type
# ------------------------------------------------
def _proba(model, X):
    try:
        if isinstance(model, xgb.Booster): return xgb_predict_proba(model, X)
    except Exception: pass
    try:
        if isinstance(model, lgb.Booster): return model.predict(X, num_iteration=model.best_iteration)
    except Exception: pass
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X); return p if p.ndim==1 else p[:,1]
    p = model.predict(X); return p if p.ndim==1 else p[:,1]

# ------------------------------------------------
# Main experiment: synthetic -> hospital
# ------------------------------------------------
def run_experiments(X_train, y_train, X_val, y_val,
                    models_to_run=('lgbm','xgb'),
                    sampling_methods=('original','rus','smoteenn'),
                    optimize_for='auprc',
                    hospital_splits=None,
                    threshold_source='hospital'):
    schema_cols = list(X_train.columns)
    X_h_tr, X_h_va, X_h_te, y_h_tr, y_h_va, y_h_te = hospital_splits
    for df in [X_h_tr, X_h_va, X_h_te]:
        df = align_to_schema(df, schema_cols, 0)

    models = {}
    splits = {
        'hospital_train': (X_h_tr, y_h_tr),
        'hospital_val':   (X_h_va, y_h_va),
        'hospital_test':  (X_h_te, y_h_te),
        'common_features': schema_cols
    }

    for method in sampling_methods:
        print(f"\n=== Sampling: {method.upper()} | Optimize: {optimize_for.upper()} ===")
        Xs, ys = apply_sampling_method(X_train, y_train, method)
        print(f"Synthetic train dist: {Counter(ys)}")

        if 'lgbm' in models_to_run:
            lgb_syn  = train_lightgbm_model(Xs, ys, X_val, y_val, optimize_for)
            p_pre_va = lgb_syn.predict(X_h_va, num_iteration=lgb_syn.best_iteration)
            th_pre   = find_optimal_threshold_for_f1(y_h_va, p_pre_va)[0] if threshold_source=='hospital' else lgb_syn.optimal_threshold
            lgb_soft = soft_transfer_learning_lightgbm(lgb_syn, X_h_tr, y_h_tr, X_h_va, y_h_va, 0.4, optimize_for)
            lgb_reg  = regularized_transfer_learning_lightgbm(lgb_syn, X_h_tr, y_h_tr, X_h_va, y_h_va, 1.0, optimize_for)
            models[('lgbm', method)] = {
                'synthetic': lgb_syn, 'hospital_pre': {'threshold': th_pre},
                'soft_ensemble': lgb_soft, 'regularized': lgb_reg
            }

        if 'xgb' in models_to_run:
            xgb_syn  = train_xgb_model(Xs, ys, X_val, y_val, optimize_for)
            p_pre_va = xgb_predict_proba(xgb_syn, X_h_va)
            th_pre   = find_optimal_threshold_for_f1(y_h_va, p_pre_va)[0] if threshold_source=='hospital' else xgb_syn.optimal_threshold
            xgb_soft = soft_transfer_learning_xgb(xgb_syn, X_h_tr, y_h_tr, X_h_va, y_h_va, 0.4, optimize_for)
            xgb_reg  = regularized_transfer_learning_xgb(xgb_syn, X_h_tr, y_h_tr, X_h_va, y_h_va, 2.0, optimize_for)
            models[('xgb', method)] = {
                'synthetic': xgb_syn, 'hospital_pre': {'threshold': th_pre},
                'soft_ensemble': xgb_soft, 'regularized': xgb_reg
            }

    return models, splits