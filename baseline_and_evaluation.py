import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import (
    f1_score, fbeta_score, precision_score, recall_score,
    precision_recall_curve, auc, brier_score_loss,
    roc_auc_score, accuracy_score, matthews_corrcoef, confusion_matrix
)
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

# imports from 02_transfer_learning.py
from transfer_learning import (
    train_lightgbm_model, train_xgb_model,
    xgb_predict_proba, _proba,
    apply_sampling_method,
    find_optimal_threshold_for_f1, find_thr_for_fbeta,
    _bootstrap_two_sided_pvalue, _ap
)

# ------------------------------------------------
# Baseline: hospital -> hospital
# ------------------------------------------------
def run_h2h_baseline(models, splits,
                     models_to_run=('lgbm','xgb'),
                     sampling_methods=('original','rus','smoteenn'),
                     optimize_for='auprc'):
    X_tr, y_tr = splits['hospital_train']
    X_va, y_va = splits['hospital_val']
    for method in sampling_methods:
        Xs, ys = apply_sampling_method(X_tr, y_tr, method)
        print(f"\n[BASELINE] sampling={method} | dist: {Counter(ys)}")
        if 'lgbm' in models_to_run:
            bl = train_lightgbm_model(Xs, ys, X_va, y_va, optimize_for)
            models.setdefault(('lgbm', method), {})['baseline'] = bl
        if 'xgb' in models_to_run:
            bl = train_xgb_model(Xs, ys, X_va, y_va, optimize_for)
            models.setdefault(('xgb', method), {})['baseline'] = bl
    return models

# ------------------------------------------------
# Metrics helpers
# ------------------------------------------------
def _more_cls_metrics(y_true, proba, th):
    y_pred = (np.asarray(proba) > th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    return dict(
        AUROC=roc_auc_score(y_true, proba),
        Precision=precision_score(y_true, y_pred, zero_division=0),
        Recall=recall_score(y_true, y_pred, zero_division=0),
        Accuracy=accuracy_score(y_true, y_pred),
        MCC=matthews_corrcoef(y_true, y_pred),
        Specificity=tn/(tn+fp) if (tn+fp)>0 else 0.0
    )

def _resolve_preds(models, splits, alg, method, kind):
    X_va, y_va = splits['hospital_val']
    X_te, y_te = splits['hospital_test']
    entry = models[(alg, method)]
    if kind == 'hospital_pre':
        model = entry['synthetic']
        p_va  = _proba(model, X_va); p_te = _proba(model, X_te)
        th_f1 = entry['hospital_pre']['threshold']
    else:
        model = entry[kind]
        p_va  = _proba(model, X_va); p_te = _proba(model, X_te)
        th_f1 = getattr(model, 'optimal_threshold', None) or find_optimal_threshold_for_f1(y_va, p_va)[0]
    th_f2 = find_thr_for_fbeta(y_va, p_va, beta=2)[0]
    return y_te, p_te, th_f1, th_f2

# ------------------------------------------------
# Bootstrap comparison: all kinds vs baseline
# ------------------------------------------------
def compare_all_kinds_vs_baseline(models, splits,
                                  algs=('lgbm','xgb'),
                                  methods=('original','rus','smoteenn'),
                                  kinds=('hospital_pre','soft_ensemble','regularized'),
                                  optimize_for='auprc',
                                  n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    X_va, y_va = splits['hospital_val']
    X_te, y_te = splits['hospital_test']

    print(f"\n🎯 {optimize_for.upper()} — Transfer vs Baseline")
    print("=" * 80)

    for alg in algs:
        for method in methods:
            key = (alg, method)
            if key not in models or 'baseline' not in models[key]: continue
            entry = models[key]

            pB_va  = _proba(entry['baseline'], X_va)
            thB_f1 = find_optimal_threshold_for_f1(y_va, pB_va)[0]
            thB_f2 = find_thr_for_fbeta(y_va, pB_va, beta=2)[0]
            pB_te  = _proba(entry['baseline'], X_te)

            B_F1   = f1_score(y_te, (pB_te>thB_f1).astype(int), zero_division=0)
            B_F2   = fbeta_score(y_te, (pB_te>thB_f2).astype(int), beta=2, zero_division=0)
            B_AP   = _ap(y_te, pB_te)
            B_BR   = brier_score_loss(y_te, pB_te)
            B_more = _more_cls_metrics(y_te, pB_te, thB_f1)

            print(f"\n{'='*60}")
            print(f"🔹 {alg.upper()} | {method.upper()} ({optimize_for.upper()})")
            print(f"[Baseline] F1={B_F1:.4f} F2={B_F2:.4f} AUPRC={B_AP:.4f} Brier={B_BR:.4f}")
            print(f"[Baseline+] AUROC={B_more['AUROC']:.4f} Precision={B_more['Precision']:.4f} "
                  f"Recall={B_more['Recall']:.4f} MCC={B_more['MCC']:.4f} Spec={B_more['Specificity']:.4f}")

            for kind in kinds:
                if kind not in entry and kind != 'hospital_pre': continue
                try:
                    yF_te, pF_te, thF_f1, thF_f2 = _resolve_preds(models, splits, alg, method, kind)
                except KeyError:
                    continue

                F1   = f1_score(yF_te, (pF_te>thF_f1).astype(int), zero_division=0)
                F2   = fbeta_score(yF_te, (pF_te>thF_f2).astype(int), beta=2, zero_division=0)
                AP   = _ap(yF_te, pF_te)
                BR   = brier_score_loss(yF_te, pF_te)
                more = _more_cls_metrics(yF_te, pF_te, thF_f1)

                print(f"\n▶ {kind}  {'🔥' if (F1>B_F1 if optimize_for=='f1' else AP>B_AP) else '❄️'}")
                print(f"  F1={F1:.4f} F2={F2:.4f} AUPRC={AP:.4f} Brier={BR:.4f}")
                print(f"  AUROC={more['AUROC']:.4f} Precision={more['Precision']:.4f} "
                      f"Recall={more['Recall']:.4f} MCC={more['MCC']:.4f} Spec={more['Specificity']:.4f}")

                n = len(y_te)
                dF1, dF2, dAP, dBr = [], [], [], []
                for _ in range(n_boot):
                    idx = rng.integers(0, n, n)
                    yt  = np.asarray(y_te)[idx]
                    if len(np.unique(yt)) < 2: continue
                    pFt = np.asarray(pF_te)[idx]; pBt = np.asarray(pB_te)[idx]
                    dF1.append(f1_score(yt,(pFt>thF_f1).astype(int),zero_division=0)
                              -f1_score(yt,(pBt>thB_f1).astype(int),zero_division=0))
                    dF2.append(fbeta_score(yt,(pFt>thF_f2).astype(int),beta=2,zero_division=0)
                              -fbeta_score(yt,(pBt>thB_f2).astype(int),beta=2,zero_division=0))
                    prF,rcF,_ = precision_recall_curve(yt,pFt); prB,rcB,_ = precision_recall_curve(yt,pBt)
                    dAP.append(auc(rcF,prF)-auc(rcB,prB))
                    dBr.append(brier_score_loss(yt,pFt)-brier_score_loss(yt,pBt))

                def _ci(x):
                    x = np.asarray(x,float)
                    lo,hi = np.percentile(x,[2.5,97.5])
                    return float(np.mean(x)),float(lo),float(hi),_bootstrap_two_sided_pvalue(x)

                mF1,lF1,uF1,pF1 = _ci(dF1); mF2,lF2,uF2,pF2 = _ci(dF2)
                mAP,lAP,uAP,pAP = _ci(dAP); mBr,lBr,uBr,pBr = _ci(dBr)

                def sig(p): return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""
                print(f"  ΔF1   ={mF1:+.4f} (95%CI {lF1:+.4f}~{uF1:+.4f}, p={pF1:.4f}) {sig(pF1)}")
                print(f"  ΔF2   ={mF2:+.4f} (95%CI {lF2:+.4f}~{uF2:+.4f}, p={pF2:.4f})")
                print(f"  ΔAUPRC={mAP:+.4f} (95%CI {lAP:+.4f}~{uAP:+.4f}, p={pAP:.4f}) {sig(pAP)}")
                print(f"  ΔBrier={mBr:+.4f} (95%CI {lBr:+.4f}~{uBr:+.4f}, p={pBr:.4f}) {sig(pBr)}")

# ------------------------------------------------
# Calibration helpers
# ------------------------------------------------
def fit_platt_on_val(y_val, p_val, max_iter=2000):
    lr = LogisticRegression(solver='lbfgs', max_iter=max_iter)
    lr.fit(np.asarray(p_val).reshape(-1,1), y_val)
    def calibrate(p): return lr.predict_proba(np.asarray(p).reshape(-1,1))[:,1]
    return calibrate

def bootstrap_ci(y, p, threshold, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed); y = np.asarray(y); p = np.asarray(p,float); rows = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y)); yt, pt = y[idx], p[idx]
        if np.min(yt)==np.max(yt): continue
        prc,rec,_ = precision_recall_curve(yt,pt)
        rows.append({"AUPRC":auc(rec,prc),
                     "F1":f1_score(yt,(pt>threshold).astype(int),zero_division=0),
                     "Brier":brier_score_loss(yt,pt)})
    df = pd.DataFrame(rows)
    def ci1(x): return dict(mean=float(x.mean()),p2_5=float(np.percentile(x,2.5)),
                            p50=float(np.percentile(x,50)),p97_5=float(np.percentile(x,97.5)))
    return {k: ci1(df[k].values) for k in df.columns}

# ------------------------------------------------
# Final test report (best model)
# ------------------------------------------------
def run_test_report(models, splits,
                    alg_pref=('lgbm','xgb'),
                    kind_pref=('regularized','soft_ensemble'),
                    results=None):
    X_te, y_te = splits['hospital_test']
    X_va, y_va = splits['hospital_val']

    # pick best AUPRC combination
    candidates = []
    for (alg, method), entry in models.items():
        for kind in kind_pref:
            if kind not in entry: continue
            model = entry[kind]
            p_te  = _proba(model, X_te)
            prc, rec, _ = precision_recall_curve(y_te, p_te)
            candidates.append({'alg':alg,'method':method,'kind':kind,'auprc':auc(rec,prc),'model':model})
    best = sorted(candidates, key=lambda x: (kind_pref.index(x['kind']) if x['kind'] in kind_pref else 9,
                                              alg_pref.index(x['alg'])  if x['alg']  in alg_pref  else 9,
                                              -x['auprc']))[0]

    ALG, METHOD, KIND, model = best['alg'], best['method'], best['kind'], best['model']
    print(f"\n=== Best model: {ALG.upper()} + {METHOD.upper()} [{KIND}] ===")

    p_va = _proba(model, X_va)
    th   = getattr(model, 'optimal_threshold', None) or find_optimal_threshold_for_f1(y_va, p_va)[0]
    p_te = _proba(model, X_te)

    prc, rec, _ = precision_recall_curve(y_te, p_te)
    print("\n--- TEST performance ---")
    print(pd.Series({"AUROC":roc_auc_score(y_te,p_te),"AUPRC":auc(rec,prc),
                     "Brier":brier_score_loss(y_te,p_te),
                     "F1":f1_score(y_te,(p_te>th).astype(int),zero_division=0),"threshold":th}))

    print("\nBootstrap 95% CI @TEST")
    print(pd.DataFrame(bootstrap_ci(y_te, p_te, th)).T)

    # Calibration
    iso = IsotonicRegression(out_of_bounds='clip').fit(p_va, y_va)
    p_te_iso = iso.transform(p_te)
    th_iso   = find_optimal_threshold_for_f1(y_va, iso.transform(p_va))[0]

    calibrate = fit_platt_on_val(y_va, p_va)
    p_te_pl   = calibrate(p_te)
    th_pl     = find_optimal_threshold_for_f1(y_va, calibrate(p_va))[0]

    def cal_metrics(y, p, th):
        prc,rec,_ = precision_recall_curve(y,p)
        return {"AUROC":roc_auc_score(y,p),"AUPRC":auc(rec,prc),
                "Brier":brier_score_loss(y,p),
                "F1":f1_score(y,(p>th).astype(int),zero_division=0),"threshold":th}

    orig_m = cal_metrics(y_te, p_te, th)
    iso_m  = cal_metrics(y_te, p_te_iso, th_iso)
    pl_m   = cal_metrics(y_te, p_te_pl,  th_pl)
    print("\n--- Calibration comparison ---")
    print(pd.DataFrame({"Original":orig_m,"Isotonic":iso_m,"Platt":pl_m}).T)

# ------------------------------------------------
# Main
# ------------------------------------------------
# if __name__ == "__main__":
#     from transfer_learning import run_experiments

#     models, splits = run_experiments(
#         models_to_run=('lgbm','xgb'),
#         sampling_methods=('original','rus','smoteenn'),
#         optimize_for='auprc',
#         hospital_splits=(
#             X_hospital_train, X_hospital_val, X_hospital_test,
#             y_hospital_train, y_hospital_val, y_hospital_test
#         ),
#         threshold_source='hospital'
#     )

#     models = run_h2h_baseline(
#         models, splits,
#         models_to_run=('lgbm','xgb'),
#         sampling_methods=('original','rus','smoteenn'),
#         optimize_for='auprc'
#     )

#     compare_all_kinds_vs_baseline(
#         models, splits,
#         algs=('lgbm','xgb'),
#         methods=('original','rus','smoteenn'),
#         kinds=('hospital_pre','soft_ensemble','regularized'),
#         optimize_for='auprc',
#         n_boot=2000, seed=42
#     )

#     run_test_report(models, splits, alg_pref=('lgbm','xgb'), kind_pref=('regularized','soft_ensemble'))