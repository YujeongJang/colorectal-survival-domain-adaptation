import os
os.chdir(r'C:')

# Cell 1 - Data preparation
# Preprocessed datasets should be loaded here.

# Cell 2 - Distribution comparison
%run distribution_comparison.py

# Cell 3 - Load transfer learning functions
%run transfer_learning.py

# Cell 4 - Load baseline and evaluation functions
%run baseline_and_evaluation.py

# Cell 5 - Run transfer learning experiments
models, splits = run_experiments(
    X_train, y_train, X_val, y_val,
    models_to_run=('lgbm','xgb'),
    sampling_methods=('original','rus','smoteenn'),
    optimize_for='auprc',
    hospital_splits=(
        X_hospital_train, X_hospital_val, X_hospital_test,
        y_hospital_train, y_hospital_val, y_hospital_test
    ),
    threshold_source='hospital'
)

# Cell 6 - Run baseline
models = run_h2h_baseline(
    models, splits,
    models_to_run=('lgbm','xgb'),
    sampling_methods=('original','rus','smoteenn'),
    optimize_for='auprc'
)

# Cell 7 - Compare transfer vs baseline
compare_all_kinds_vs_baseline(
    models, splits,
    algs=('lgbm','xgb'),
    methods=('original','rus','smoteenn'),
    kinds=('hospital_pre','soft_ensemble','regularized'),
    optimize_for='auprc',
    n_boot=2000, seed=42
)

# Cell 8 - Final test report
run_test_report(models, splits)