import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chi2_contingency, fisher_exact

# syndata, testdata are loaded in 01_data_preparation.py

var_list = [
    "AGE","PRIOR_CEA","PRIOR_CEA_LOG","POST_CEA_LOG","MTST",
    "MALE_C","FEMALE_C",
    "AGE20_C","AGE30_C","AGE40_C","AGE50_C","AGE60_C","AGE70_C","AGE80_C","AGE90_C",
    "VAS_INV_C","LYMP_INV_C","NR_INV_C",
    "COLON_C","RECTUM_C","JUNCTION_C",
    "STAGE_1_C","STAGE_2_C","STAGE_3_C","STAGE_4_C",
    "FOLFOX_C","FOLFIRI_C","GELOX_C","OTHCH_C","RAD_C",
    "CARCINOMA_C","diag_year","DFS_5Y_C","OS_7Y_C"
]

num_vars = ["AGE","PRIOR_CEA","PRIOR_CEA_LOG","POST_CEA_LOG","MTST"]
cat_vars = [v for v in var_list if v not in num_vars]

present  = [v for v in var_list if v in syndata.columns or v in testdata.columns]
num_vars = [v for v in num_vars if v in present]
cat_vars = [v for v in cat_vars if v in present]

train = syndata.loc[syndata['diag_year'] < 2013]
val   = syndata.loc[syndata['diag_year'] >= 2013]
all_  = syndata
test  = testdata

# ------------------------------------------------
# Helper functions
# ------------------------------------------------
def _to_numeric_safe(s):
    return s if pd.api.types.is_numeric_dtype(s) else pd.to_numeric(s, errors='coerce')

def _to_int_if_binary(s):
    if pd.api.types.is_integer_dtype(s) or pd.api.types.is_bool_dtype(s): return s
    if str(s.dtype) == 'category' or pd.api.types.is_object_dtype(s):
        try: return s.astype(int)
        except Exception: return pd.to_numeric(s, errors='coerce').fillna(0).astype(int)
    return s

def _chi2_or_fisher(table):
    if table.shape == (2,2):
        _, p_chi, _, expected = chi2_contingency(table, correction=False)
        return fisher_exact(table, alternative='two-sided')[1] if (expected < 5).any() else p_chi
    return chi2_contingency(table, correction=False)[1]

def _median_iqr(s):
    s = s.dropna()
    if s.empty: return "NA"
    return f"{s.median():.2f} ({s.quantile(0.25):.2f}-{s.quantile(0.75):.2f})"

# ------------------------------------------------
# Build result table
# ------------------------------------------------
rows = []

for col in num_vars:
    s_all  = _to_numeric_safe(all_[col]).dropna()
    s_tr   = _to_numeric_safe(train[col]).dropna()
    s_val  = _to_numeric_safe(val[col]).dropna()
    s_test = _to_numeric_safe(test[col]).dropna()
    try: p_tv = ks_2samp(s_tr, s_val).pvalue if len(s_tr)>0 and len(s_val)>0 else np.nan
    except: p_tv = np.nan
    try: p_ah = ks_2samp(s_all, s_test).pvalue if len(s_all)>0 and len(s_test)>0 else np.nan
    except: p_ah = np.nan
    rows.append({
        "Variable":col,
        "All N":int(all_[col].notna().sum()), "All Median (IQR)":_median_iqr(s_all),
        "All N 1":"", "All %":"",
        "Train N":int(train[col].notna().sum()), "Train Median (IQR)":_median_iqr(s_tr),
        "Val N":int(val[col].notna().sum()),     "Val Median (IQR)":_median_iqr(s_val),
        "p (Train vs Val)":None if pd.isna(p_tv) else round(float(p_tv),4),
        "Test N":int(test[col].notna().sum()),   "Test Median (IQR)":_median_iqr(s_test),
        "p (Synth vs Hosp)":None if pd.isna(p_ah) else round(float(p_ah),4),
    })

for col in cat_vars:
    a_i  = _to_int_if_binary(all_[col])
    tr_i = _to_int_if_binary(train[col])
    va_i = _to_int_if_binary(val[col])
    te_i = _to_int_if_binary(test[col])

    def bc(s):
        n = int(s.notna().sum()); n1 = int((s==1).sum())
        return n, n1, (n1/n*100) if n>0 else 0.0

    all_n,all_1,all_pct = bc(a_i); tr_n,tr_1,tr_pct = bc(tr_i)
    va_n,va_1,va_pct    = bc(va_i); te_n,te_1,te_pct = bc(te_i)

    def _p2k(s1, s2):
        cats = np.array(sorted(np.union1d(s1.dropna().unique(), s2.dropna().unique())))
        tbl  = np.vstack([[(s1==c).sum() for c in cats],[(s2==c).sum() for c in cats]])
        return _chi2_or_fisher(tbl) if tbl.sum()>0 else np.nan

    try:    p_tv = _p2k(tr_i, va_i)
    except: p_tv = np.nan
    try:    p_ah = _p2k(a_i, te_i)
    except: p_ah = np.nan

    rows.append({
        "Variable":col,
        "All N":all_n, "All Median (IQR)":"", "All N 1":all_1, "All %":round(all_pct,2),
        "Train N":tr_n, "Train Median (IQR)":"",
        "Val N":va_n,   "Val Median (IQR)":"",
        "p (Train vs Val)":None if pd.isna(p_tv) else round(float(p_tv),4),
        "Test N":te_n,  "Test Median (IQR)":"",
        "p (Synth vs Hosp)":None if pd.isna(p_ah) else round(float(p_ah),4),
    })

result_df = pd.DataFrame(rows)[[
    "Variable","All N","All Median (IQR)","All N 1","All %",
    "Train N","Train Median (IQR)",
    "Val N","Val Median (IQR)","p (Train vs Val)",
    "Test N","Test Median (IQR)","p (Synth vs Hosp)"
]]

print(result_df)
