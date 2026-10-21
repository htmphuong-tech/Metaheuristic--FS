# ============================================================
# Software Fault Prediction - NASA / AEEM
# VERSION 4 OPTIMIZED: mealpy selectors with leakage-safe inner CV,
# imbalance-aware composite fitness (AUPRC/MCC/AUC), linear and
# non-linear proxy models, stable top-k decoding, and penalties for
# feature count, redundancy, and fold-to-fold instability.
#
# Why this change:
#   - Removes the risk of self-implementation bugs (in an earlier
#     run, hand-written BBOA/ZOA/STO produced byte-identical AUC
#     across all 10 datasets - almost certainly a wiring bug).
#   - mealpy's own ZOA docstring notes that several of these
#     "animal-named" algorithms (ZOA/OOA/POA/STO/SOA/WOA/FFO...)
#     are structurally near-duplicates of each other by design -
#     worth citing in the paper's Threats to Validity section.
#   - Gives citable, versioned, third-party implementations
#     (mealpy 3.0.x), which reviewers can independently verify.
#
# The held-out test set remains untouched until final evaluation.
# ============================================================
import numpy as np
import pandas as pd
import os, warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, matthews_corrcoef, make_scorer)
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from scipy.stats import friedmanchisquare, wilcoxon
from itertools import combinations

# ---- mealpy ----
from mealpy import FloatVar
from mealpy.evolutionary_based import GA as mp_GA
from mealpy.bio_based import BBOA as mp_BBOA
from mealpy.swarm_based import ZOA as mp_ZOA
from mealpy.swarm_based import SSA as mp_SSA
from mealpy.swarm_based import SSO as mp_SSO

K_MIN = 2
K_MAX = None
K_MAX_RATIO = 0.50
SIZE_PENALTY = 0.03
STABILITY_PENALTY = 0.15
REDUNDANCY_PENALTY = 0.02
N_RUNS, BASE_SEED, USE_SMOTE = 5, 42, True
N_POP, N_ITER = 20, 50


def get_proxy_models(seed):
    """Cheap linear + non-linear proxies reduce selector/model mismatch."""
    return [
        LogisticRegression(max_iter=500, class_weight="balanced", random_state=seed),
        HistGradientBoostingClassifier(max_iter=60, learning_rate=0.08,
                                       random_state=seed),
    ]


class ELM:
    def __init__(self, input_size, hidden_size=50, seed=None):
        self.input_size, self.hidden_size = input_size, hidden_size
        self.rng = np.random.default_rng(seed)
    def _act(self, x):
        return 1 / (1 + np.exp(-x))
    def fit(self, X, y):
        self.W = self.rng.standard_normal((self.input_size, self.hidden_size))
        self.b = self.rng.standard_normal(self.hidden_size)
        H = self._act(X @ self.W + self.b)
        self.beta = np.linalg.pinv(H) @ y
    def predict(self, X):
        return (self._act(X @ self.W + self.b) @ self.beta > 0.5).astype(int)
    def predict_proba(self, X):
        p = np.clip(self._act(X @ self.W + self.b) @ self.beta, 0, 1)
        return np.column_stack([1 - p, p])


def load_data(path):
    df = pd.read_csv(path).dropna()
    y = df.iloc[:, -1]
    if pd.api.types.is_numeric_dtype(y):
        y = (y > 0).astype(int)
    else:
        y = y.astype(str).str.lower().str.strip()
        y = y.map({'yes': 1, 'no': 0}) if set(y.unique()).issubset({'yes', 'no'}) else pd.factorize(y)[0]
    X = df.iloc[:, :-1].select_dtypes(include=[np.number]).values
    return X, np.asarray(y)


def model_cv_score(X, y, idx, seed=42, use_smote=True):
    """Leakage-safe, imbalance-aware CV score for one selected feature set."""
    X_sel = X[:, idx]
    if X_sel.shape[1] < 2:
        return 0.0
    n_splits = max(2, min(3, np.unique(y).size,
                          np.bincount(y).min() if len(np.unique(y)) > 1 else 2))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scoring = {
        "ap": "average_precision",
        "auc": "roc_auc",
        "mcc": make_scorer(matthews_corrcoef),
    }
    proxy_scores = []
    for model in get_proxy_models(seed):
        steps = [("scaler", MinMaxScaler())]
        if use_smote:
            steps.append(("smote", SMOTE(random_state=seed, k_neighbors=1)))
        steps.append(("model", model))
        pipeline = Pipeline(steps)
        try:
            scores = cross_validate(pipeline, X_sel, y, cv=cv, scoring=scoring,
                                    error_score=np.nan)
            ap = np.asarray(scores["test_ap"], dtype=float)
            auc = np.asarray(scores["test_auc"], dtype=float)
            mcc = np.asarray(scores["test_mcc"], dtype=float)
            if np.all(np.isnan(ap)) or np.all(np.isnan(auc)) or np.all(np.isnan(mcc)):
                continue
            composite = (0.50 * np.nanmean(ap) +
                         0.30 * np.nanmean(mcc) +
                         0.20 * np.nanmean(auc) -
                         STABILITY_PENALTY * np.nanstd(ap))
            proxy_scores.append(float(composite))
        except Exception:
            continue
    return float(np.mean(proxy_scores)) if proxy_scores else 0.0


def decode_mask(sol, d):
    """Convert continuous solution to a stable top-k feature mask."""
    scores = np.asarray(sol[:d], dtype=float)
    max_k = K_MAX if K_MAX is not None else int(np.ceil(K_MAX_RATIO * d))
    max_k = max(1, min(d, max_k))
    min_k = min(K_MIN, max_k)
    k_code = float(np.clip(sol[d], 0.0, 1.0))
    k = int(round(min_k + k_code * (max_k - min_k)))
    mask = np.zeros(d, dtype=bool)
    # Stable sort makes identical solutions reproducible when scores tie.
    ranked = np.argsort(-scores, kind="stable")
    mask[ranked[:k]] = True
    return mask


def feature_redundancy(X_selected):
    """Mean absolute pairwise correlation, ignoring the diagonal."""
    if X_selected.shape[1] < 2:
        return 0.0
    corr = np.corrcoef(X_selected, rowvar=False)
    corr = np.nan_to_num(np.abs(corr), nan=0.0, posinf=0.0, neginf=0.0)
    upper = corr[np.triu_indices_from(corr, k=1)]
    return float(np.mean(upper)) if upper.size else 0.0


def fitness(X, y, sol, seed=42, use_smote=True):
    d = X.shape[1]
    mask = decode_mask(sol, d)
    idx = np.flatnonzero(mask)
    base = model_cv_score(X, y, idx, seed=seed, use_smote=use_smote)
    redundancy = feature_redundancy(X[:, idx])
    return (base
            - SIZE_PENALTY * (idx.size / d)
            - REDUNDANCY_PENALTY * redundancy)


# ==========================
# mealpy wrapper
# ==========================
# Each entry: name -> zero-arg factory returning a FRESH mealpy
# optimizer instance (must be fresh per call; reusing one instance
# across many (dataset, run) cells would leak internal state).
MEALPY_ALGOS = {
    # k_way explicit: with small pop_size, mealpy's default k_way=0.2
    # can round down to a 1-individual tournament and crash. Safe with
    # N_POP=20 here, but pin it explicitly so this never breaks if
    # N_POP is lowered later (e.g. for a quick test run).
    "GA":   lambda: mp_GA.BaseGA(epoch=N_ITER, pop_size=N_POP, pc=0.9, pm=0.05, k_way=0.4),
    "BBOA": lambda: mp_BBOA.OriginalBBOA(epoch=N_ITER, pop_size=N_POP),
    "ZOA":  lambda: mp_ZOA.OriginalZOA(epoch=N_ITER, pop_size=N_POP),
    "SSA":  lambda: mp_SSA.OriginalSSA(epoch=N_ITER, pop_size=N_POP),
    "SSO":  lambda: mp_SSO.OriginalSSO(epoch=N_ITER, pop_size=N_POP),
}


def run_mealpy(optimizer_factory, obj_func, dim, seed):
    """Run one mealpy optimizer on obj_func (maximize) over [0,1]^dim, return best solution."""
    problem = {
        "obj_func": obj_func,
        "bounds": FloatVar(lb=[0.0] * dim, ub=[1.0] * dim),
        "minmax": "max",
        "log_to": None,   # suppress mealpy's per-iteration console output
    }
    optimizer = optimizer_factory()
    g_best = optimizer.solve(problem, seed=seed)
    return np.asarray(g_best.solution, dtype=float)


N_POP_UNUSED = None  # kept for readability; actual pop/iter set in MEALPY_ALGOS above


def get_models(input_dim, seed=42):
    return {
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=seed),
        "LightGBM": LGBMClassifier(n_estimators=200, learning_rate=0.05, verbose=-1, random_state=seed),
        "CatBoost": CatBoostClassifier(iterations=200, verbose=0, random_state=seed),
        "HGBoost": HistGradientBoostingClassifier(max_iter=200, random_state=seed),
        "MLP": MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, early_stopping=True, random_state=seed),
        "ELM": ELM(input_dim, seed=seed),
    }


def evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    try:
        y_prob = model.predict_proba(X_test)
        y_prob = y_prob[:, 1] if len(y_prob.shape) > 1 else y_prob
    except Exception:
        try:
            y_prob = model.decision_function(X_test)
        except Exception:
            y_prob = model.predict(X_test)
    y_pred = model.predict(X_test)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        auc = 0.5
    return {"Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
            "F1": f1_score(y_test, y_pred, average='weighted', zero_division=0),
            "AUC": auc}


def run_all(folder, n_runs=N_RUNS, base_seed=BASE_SEED, use_smote=USE_SMOTE):
    rows, feat_rows = [], []
    for file in sorted(os.listdir(folder)):
        if not file.endswith(".csv"):
            continue
        print("\n[V4-mealpy] Processing:", file)
        try:
            X_raw, y = load_data(os.path.join(folder, file))
            d = X_raw.shape[1]
            for run_id in range(n_runs):
                seed = base_seed + run_id
                stratify = y if np.min(np.unique(y, return_counts=True)[1]) >= 2 else None
                Xtr_raw, Xte_raw, ytr_raw, yte = train_test_split(
                    X_raw, y, test_size=0.3, random_state=seed, stratify=stratify)

                for algo_name, make_optimizer in MEALPY_ALGOS.items():
                    # Feature selection sees only the raw outer-training split.
                    # Scaling and SMOTE happen independently inside each CV fold.
                    obj_func = lambda s: fitness(Xtr_raw, ytr_raw, s, seed, use_smote)
                    best_sol = run_mealpy(make_optimizer, obj_func, d + 1, seed)
                    mask = decode_mask(best_sol, d)
                    idx = np.flatnonzero(mask)
                    k_sel = int(mask.sum())
                    print(f"    {algo_name}: k* = {k_sel}/{d}")

                    ranked_idx = idx[np.argsort(-best_sol[idx], kind="stable")]
                    for r, fi in enumerate(ranked_idx, 1):
                        feat_rows.append({"Dataset": file, "Algo": algo_name, "Run": run_id,
                                          "Feature": f"F{fi}", "Rank": r})

                    sc = MinMaxScaler()
                    Xs_tr = sc.fit_transform(Xtr_raw[:, idx])
                    Xs_te = sc.transform(Xte_raw[:, idx])
                    ytr = ytr_raw.copy()
                    if use_smote and np.min(np.bincount(ytr)) >= 2:
                        sm = SMOTE(
                            random_state=seed,
                            k_neighbors=max(1, min(5, np.min(np.bincount(ytr)) - 1)))
                        Xs_tr, ytr = sm.fit_resample(Xs_tr, ytr)
                    for mn, mdl in get_models(len(idx), seed=seed).items():
                        m = evaluate(mdl, Xs_tr, Xs_te, ytr, yte)
                        rows.append({"Dataset": file, "Algo": algo_name, "Model": mn,
                                     "Run": run_id, "K_selected": k_sel, **m})
        except Exception as e:
            print("Error:", file, e)
    return pd.DataFrame(rows), pd.DataFrame(feat_rows)


def summarize_runs(df):
    agg = df.groupby(["Dataset", "Algo", "Model"])[["Accuracy", "Precision", "Recall", "F1", "AUC"]] \
            .agg(["mean", "std"])
    agg.columns = ["_".join(c) for c in agg.columns]
    return agg.reset_index()


def k_report(df):
    return df.groupby(["Dataset", "Algo"])["K_selected"] \
             .agg(["mean", "std", "min", "max"]).reset_index()


def run_baseline(folder, n_runs=N_RUNS, base_seed=BASE_SEED, use_smote=USE_SMOTE):
    baseline_results = []
    for file in sorted(os.listdir(folder)):
        if not file.endswith(".csv"):
            continue
        print("\n[BASELINE] Processing:", file)
        try:
            X_raw, y = load_data(os.path.join(folder, file))
            d = X_raw.shape[1]
            for run_id in range(n_runs):
                seed = base_seed + run_id
                stratify = y if np.min(np.unique(y, return_counts=True)[1]) >= 2 else None
                X_train, X_test, y_train, y_test = train_test_split(
                    X_raw, y, test_size=0.3, random_state=seed, stratify=stratify)
                scaler = MinMaxScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                if use_smote and np.min(np.bincount(y_train)) >= 2:
                    sm = SMOTE(random_state=seed,
                               k_neighbors=max(1, min(5, np.min(np.bincount(y_train)) - 1)))
                    X_train, y_train = sm.fit_resample(X_train, y_train)
                for model_name, model in get_models(d, seed=seed).items():
                    metrics = evaluate(model, X_train, X_test, y_train, y_test)
                    baseline_results.append({
                        "Dataset": file, "FS": "Baseline_All_Features",
                        "Model": model_name, "Run": run_id, "N_Features": d, **metrics})
        except Exception as e:
            print("[BASELINE] Error:", file, e)
    return pd.DataFrame(baseline_results)


def summarize_baseline(df_baseline):
    agg = (df_baseline.groupby(["Dataset", "FS", "Model", "N_Features"])
                      [["Accuracy", "Precision", "Recall", "F1", "AUC"]]
                      .agg(["mean", "std"]))
    agg.columns = ["_".join(c) for c in agg.columns]
    return agg.reset_index()


def statistical_tests(df_runs, metric="AUC"):
    mean_df = df_runs.groupby(["Dataset", "Algo", "Model"])[metric].mean().reset_index()
    results = {}
    for model_name in mean_df["Model"].unique():
        sub = mean_df[mean_df["Model"] == model_name]
        pivot = sub.pivot(index="Dataset", columns="Algo", values=metric).dropna()
        if pivot.shape[1] < 3 or pivot.shape[0] < 3:
            continue
        algos = pivot.columns.tolist()
        try:
            stat, p = friedmanchisquare(*[pivot[a].values for a in algos])
        except Exception:
            stat, p = np.nan, np.nan
        pairwise = []
        if p is not None and not np.isnan(p) and p < 0.05:
            for a, b in combinations(algos, 2):
                try:
                    w_stat, w_p = wilcoxon(pivot[a], pivot[b])
                except Exception:
                    w_stat, w_p = np.nan, np.nan
                pairwise.append({"Algo_A": a, "Algo_B": b, "wilcoxon_p": w_p})
        results[model_name] = {"friedman_stat": stat, "friedman_p": p,
                               "pairwise_wilcoxon": pd.DataFrame(pairwise)}
    return results


if __name__ == "__main__":
    folder = r"E:\me\NCKH\DUE\DUE_Scoups\dataset\AEEM"  # adjust to your dataset folder

    print("=" * 70); print("V4 (mealpy) - 1. BASELINE (ALL FEATURES)"); print("=" * 70)
    df_baseline = run_baseline(folder)
    df_baseline.to_excel("V4_SFP_baseline_all_runs.xlsx", index=False)
    summarize_baseline(df_baseline).to_excel("V4_SFP_baseline_summary_meanstd.xlsx", index=False)

    print("=" * 70); print("V4 (mealpy) - 2. FEATURE SELECTION (adaptive k, mealpy optimizers)"); print("=" * 70)
    df_runs, feat_df = run_all(folder)
    df_runs.to_excel("V4_SFP_all_runs.xlsx", index=False)
    summarize_runs(df_runs).to_excel("V4_SFP_summary_meanstd.xlsx", index=False)

    print("=" * 70); print("V4 (mealpy) - 3. STATISTICAL TESTS"); print("=" * 70)
    tests = statistical_tests(df_runs, metric="AUC")
    with pd.ExcelWriter("V4_SFP_statistical_tests.xlsx") as writer:
        friedman_rows = []
        for model_name, res in tests.items():
            friedman_rows.append({"Model": model_name,
                                  "Friedman_stat": res["friedman_stat"],
                                  "Friedman_p": res["friedman_p"]})
            if not res["pairwise_wilcoxon"].empty:
                res["pairwise_wilcoxon"].to_excel(writer, sheet_name=f"{model_name}_wilcoxon", index=False)
        pd.DataFrame(friedman_rows).to_excel(writer, sheet_name="Friedman_summary", index=False)

    feat_df.to_csv("V4_SFP_top_features.csv", index=False)
    k_report(df_runs).to_excel("V4_SFP_k_report.xlsx", index=False)

    print("\nDONE - V4 (mealpy)")
