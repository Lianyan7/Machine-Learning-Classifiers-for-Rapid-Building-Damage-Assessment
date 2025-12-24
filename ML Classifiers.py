#!/usr/bin/env python3
"""
Integrated Classifier Evaluation Script
=========================================

This script reads data from an Excel workbook ("RBA.xlsx") containing four sheets,
performs leakage-safe preprocessing, and evaluates four classifiers (KNN, MLP,
Random Forest, and SVM) over a range of hyperparameters.

Evaluation design:
  1) Stratified 80/20 train-test split to reserve an independent final test set.
  2) Stratified 5-fold cross-validation performed ONLY within the 80% training subset
     for model selection and robust performance estimation.
  3) Each hyperparameter setting is then refit on the full 80% training subset and
     evaluated once on the held-out 20% test subset.

For each classifier, results are saved to Excel workbooks in the "Detailed results" directory:
  - "CV Results" sheet: mean/std metrics across inner CV folds + aggregated confusion matrix (sum over folds)
  - "Test Results" sheet: single-shot hold-out test metrics + confusion matrix

A confusion matrix visualisation (with additional indicators) is generated using the MLP
classifier's hold-out test confusion matrix from the best MLP setting selected by CV F1_mean.

Author: [Lianyan Li]
Date: [22/02/2025]

"""

# =============================================================================
# CONFIG
# =============================================================================

FILE_PATH = "RBA.xlsx"
OUTPUT_DIR = "Detailed results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS_INNER = 5  # Stratified 5-fold CV inside the training 80%

# Preferred engineering label order (used if it matches your label set)
PREFERRED_STATES = ["G", "Y", "R", "G1", "G2", "Y1", "Y2", "R1", "R2", "R3"]


# =============================================================================
# UTILITIES
# =============================================================================

def _dense_if_sparse(X):
    """Convert sparse matrices to dense arrays; pass-through for dense inputs."""
    return X.toarray() if hasattr(X, "toarray") else X


def build_preprocess(numeric_cols, categorical_cols):
    """Create a leakage-safe preprocessing transformer."""
    # Compatibility with different scikit-learn versions
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", ohe, categorical_cols),
        ],
        remainder="drop"
    )
    return preprocess


def get_label_order(y_all):
    """Use preferred label order if possible; otherwise fall back to sorted unique labels."""
    labels_unique = sorted(pd.unique(pd.Series(y_all).astype(str)))
    if set(PREFERRED_STATES).issubset(set(labels_unique)):
        # preserve preferred order, and append any extra labels (if present)
        extras = [lab for lab in labels_unique if lab not in PREFERRED_STATES]
        return PREFERRED_STATES + extras
    return labels_unique


def compute_metrics(y_true, y_pred):
    """Compute standard classification metrics (weighted)."""
    return {
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def inner_cv_evaluate(pipe, X_train_df, y_train, cv, labels_order):
    """
    Run stratified K-fold CV on training set only.
    Returns mean/std metrics and aggregated confusion matrix (sum across validation folds),
    plus mean fit/predict times per fold.
    """
    f1s, recalls, precisions, accs = [], [], [], []
    fit_times, pred_times = [], []
    cm_sum = np.zeros((len(labels_order), len(labels_order)), dtype=int)

    for tr_idx, va_idx in cv.split(X_train_df, y_train):
        X_tr = X_train_df.iloc[tr_idx]
        y_tr = y_train[tr_idx]
        X_va = X_train_df.iloc[va_idx]
        y_va = y_train[va_idx]

        t0 = time.time()
        pipe.fit(X_tr, y_tr)
        fit_times.append(time.time() - t0)

        t0 = time.time()
        y_pred = pipe.predict(X_va)
        pred_times.append(time.time() - t0)

        cm_sum += confusion_matrix(y_va, y_pred, labels=labels_order)

        m = compute_metrics(y_va, y_pred)
        f1s.append(m["f1"])
        recalls.append(m["recall"])
        precisions.append(m["precision"])
        accs.append(m["accuracy"])

    return {
        "cm_sum": cm_sum,
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s, ddof=1)),
        "recall_mean": float(np.mean(recalls)),
        "recall_std": float(np.std(recalls, ddof=1)),
        "precision_mean": float(np.mean(precisions)),
        "precision_std": float(np.std(precisions, ddof=1)),
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs, ddof=1)),
        "fit_time_mean": float(np.mean(fit_times)),
        "pred_time_mean": float(np.mean(pred_times)),
    }


def holdout_test_evaluate(pipe, X_train_df, y_train, X_test_df, y_test, labels_order):
    """
    Fit on full training subset (80%) and evaluate once on held-out test subset (20%).
    """
    t0 = time.time()
    pipe.fit(X_train_df, y_train)
    fit_time = time.time() - t0

    t0 = time.time()
    y_pred = pipe.predict(X_test_df)
    pred_time = time.time() - t0

    cm = confusion_matrix(y_test, y_pred, labels=labels_order)
    m = compute_metrics(y_test, y_pred)

    return {
        "cm": cm,
        "f1": float(m["f1"]),
        "recall": float(m["recall"]),
        "precision": float(m["precision"]),
        "accuracy": float(m["accuracy"]),
        "fit_time": float(fit_time),
        "pred_time": float(pred_time),
    }


# =============================================================================
# DATA LOADING
# =============================================================================

df_value = pd.read_excel(FILE_PATH, sheet_name="Value")
df_multiclass = pd.read_excel(FILE_PATH, sheet_name="Multi-class")
df_binary = pd.read_excel(FILE_PATH, sheet_name="Binary")
df_label = pd.read_excel(FILE_PATH, sheet_name="Label")

# Features as a single DataFrame (raw). Preprocessing will occur inside pipelines.
X = pd.concat([df_value, df_multiclass, df_binary], axis=1)

# Labels as 1D vector (strings). Assumes one label column.
if df_label.shape[1] != 1:
    raise ValueError("Expected 'Label' sheet to contain exactly one column of class labels.")
y = df_label.iloc[:, 0].astype(str).values

numeric_cols = df_value.columns.tolist()
categorical_cols = df_multiclass.columns.tolist() + df_binary.columns.tolist()

labels_order = get_label_order(y)

# Preprocess + densify (to support estimators that do not accept sparse input robustly)
preprocess = build_preprocess(numeric_cols, categorical_cols)
to_dense = FunctionTransformer(_dense_if_sparse, accept_sparse=True)

# Outer split: stratified 80/20
X_train_df, X_test_df, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Inner CV: stratified 5-fold on training subset only
inner_cv = StratifiedKFold(n_splits=N_SPLITS_INNER, shuffle=True, random_state=RANDOM_STATE)


# =============================================================================
# 1) KNN
# =============================================================================

neighbor_values = range(1, 11)
knn_path = os.path.join(OUTPUT_DIR, "KNN_results_holdout80_CV5.xlsx")
wb_knn = Workbook()

ws_cv_knn = wb_knn.active
ws_cv_knn.title = "CV Results"
ws_cv_knn.append([
    "Neighbors",
    "CV Confusion Matrix (sum over folds)",
    "F1_mean", "F1_std",
    "Recall_mean", "Recall_std",
    "Precision_mean", "Precision_std",
    "Accuracy_mean", "Accuracy_std",
    "FitTime_mean(s)", "PredTime_mean(s)"
])

ws_test_knn = wb_knn.create_sheet(title="Test Results")
ws_test_knn.append([
    "Neighbors",
    "Test Confusion Matrix",
    "F1", "Recall", "Precision", "Accuracy",
    "FitTime(s)", "PredTime(s)"
])

for k in neighbor_values:
    clf = KNeighborsClassifier(n_neighbors=k)
    pipe = Pipeline([("prep", preprocess), ("dense", to_dense), ("clf", clf)])

    cv_res = inner_cv_evaluate(pipe, X_train_df, y_train, inner_cv, labels_order)
    test_res = holdout_test_evaluate(pipe, X_train_df, y_train, X_test_df, y_test, labels_order)

    ws_cv_knn.append([
        k,
        str(cv_res["cm_sum"]),
        cv_res["f1_mean"], cv_res["f1_std"],
        cv_res["recall_mean"], cv_res["recall_std"],
        cv_res["precision_mean"], cv_res["precision_std"],
        cv_res["acc_mean"], cv_res["acc_std"],
        cv_res["fit_time_mean"], cv_res["pred_time_mean"]
    ])

    ws_test_knn.append([
        k,
        str(test_res["cm"]),
        test_res["f1"], test_res["recall"], test_res["precision"], test_res["accuracy"],
        test_res["fit_time"], test_res["pred_time"]
    ])

wb_knn.save(knn_path)
print(f"KNN results saved to {knn_path}")


# =============================================================================
# 2) MLP
# =============================================================================

hidden_states = [100, 200, 300, 400, 500]
max_iterations = [100, 300, 500, 700]
learning_rates = [0.01, 0.001]

mlp_path = os.path.join(OUTPUT_DIR, "MLP_results_holdout80_CV5.xlsx")
wb_mlp = Workbook()

ws_cv_mlp = wb_mlp.active
ws_cv_mlp.title = "CV Results"
ws_cv_mlp.append([
    "Hidden Layer Size", "Max Iterations", "Learning Rate",
    "CV Confusion Matrix (sum over folds)",
    "F1_mean", "F1_std",
    "Recall_mean", "Recall_std",
    "Precision_mean", "Precision_std",
    "Accuracy_mean", "Accuracy_std",
    "FitTime_mean(s)", "PredTime_mean(s)"
])

ws_test_mlp = wb_mlp.create_sheet(title="Test Results")
ws_test_mlp.append([
    "Hidden Layer Size", "Max Iterations", "Learning Rate",
    "Test Confusion Matrix",
    "F1", "Recall", "Precision", "Accuracy",
    "FitTime(s)", "PredTime(s)"
])

# Track best MLP by CV F1_mean
best_mlp = {"f1_mean": -1.0, "params": None, "test_cm": None}

for h in hidden_states:
    for mi in max_iterations:
        for lr in learning_rates:
            clf = MLPClassifier(
                hidden_layer_sizes=(h,),
                max_iter=mi,
                learning_rate_init=lr,
                random_state=RANDOM_STATE
            )
            pipe = Pipeline([("prep", preprocess), ("dense", to_dense), ("clf", clf)])

            cv_res = inner_cv_evaluate(pipe, X_train_df, y_train, inner_cv, labels_order)
            test_res = holdout_test_evaluate(pipe, X_train_df, y_train, X_test_df, y_test, labels_order)

            ws_cv_mlp.append([
                h, mi, lr,
                str(cv_res["cm_sum"]),
                cv_res["f1_mean"], cv_res["f1_std"],
                cv_res["recall_mean"], cv_res["recall_std"],
                cv_res["precision_mean"], cv_res["precision_std"],
                cv_res["acc_mean"], cv_res["acc_std"],
                cv_res["fit_time_mean"], cv_res["pred_time_mean"]
            ])

            ws_test_mlp.append([
                h, mi, lr,
                str(test_res["cm"]),
                test_res["f1"], test_res["recall"], test_res["precision"], test_res["accuracy"],
                test_res["fit_time"], test_res["pred_time"]
            ])

            if cv_res["f1_mean"] > best_mlp["f1_mean"]:
                best_mlp["f1_mean"] = cv_res["f1_mean"]
                best_mlp["params"] = (h, mi, lr)
                best_mlp["test_cm"] = test_res["cm"]

wb_mlp.save(mlp_path)
print(f"MLP results saved to {mlp_path}")


# =============================================================================
# 3) Random Forest
# =============================================================================

n_estimators_values = [50, 60, 70, 80, 90, 100, 110, 120]

rf_path = os.path.join(OUTPUT_DIR, "RandomForest_results_holdout80_CV5.xlsx")
wb_rf = Workbook()

ws_cv_rf = wb_rf.active
ws_cv_rf.title = "CV Results"
ws_cv_rf.append([
    "n_estimators",
    "CV Confusion Matrix (sum over folds)",
    "F1_mean", "F1_std",
    "Recall_mean", "Recall_std",
    "Precision_mean", "Precision_std",
    "Accuracy_mean", "Accuracy_std",
    "FitTime_mean(s)", "PredTime_mean(s)"
])

ws_test_rf = wb_rf.create_sheet(title="Test Results")
ws_test_rf.append([
    "n_estimators",
    "Test Confusion Matrix",
    "F1", "Recall", "Precision", "Accuracy",
    "FitTime(s)", "PredTime(s)"
])

for n_estimators in n_estimators_values:
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=RANDOM_STATE)
    pipe = Pipeline([("prep", preprocess), ("dense", to_dense), ("clf", clf)])

    cv_res = inner_cv_evaluate(pipe, X_train_df, y_train, inner_cv, labels_order)
    test_res = holdout_test_evaluate(pipe, X_train_df, y_train, X_test_df, y_test, labels_order)

    ws_cv_rf.append([
        n_estimators,
        str(cv_res["cm_sum"]),
        cv_res["f1_mean"], cv_res["f1_std"],
        cv_res["recall_mean"], cv_res["recall_std"],
        cv_res["precision_mean"], cv_res["precision_std"],
        cv_res["acc_mean"], cv_res["acc_std"],
        cv_res["fit_time_mean"], cv_res["pred_time_mean"]
    ])

    ws_test_rf.append([
        n_estimators,
        str(test_res["cm"]),
        test_res["f1"], test_res["recall"], test_res["precision"], test_res["accuracy"],
        test_res["fit_time"], test_res["pred_time"]
    ])

wb_rf.save(rf_path)
print(f"Random Forest results saved to {rf_path}")


# =============================================================================
# 4) SVM
# =============================================================================

kernel_values = ["rbf", "linear", "poly"]
degree_values = [2, 3, 4, 5]
gamma_values = ["scale", "auto"]

svm_path = os.path.join(OUTPUT_DIR, "SVM_results_holdout80_CV5.xlsx")
wb_svm = Workbook()

ws_cv_svm = wb_svm.active
ws_cv_svm.title = "CV Results"
ws_cv_svm.append([
    "Kernel", "Degree", "Gamma",
    "CV Confusion Matrix (sum over folds)",
    "F1_mean", "F1_std",
    "Recall_mean", "Recall_std",
    "Precision_mean", "Precision_std",
    "Accuracy_mean", "Accuracy_std",
    "FitTime_mean(s)", "PredTime_mean(s)"
])

ws_test_svm = wb_svm.create_sheet(title="Test Results")
ws_test_svm.append([
    "Kernel", "Degree", "Gamma",
    "Test Confusion Matrix",
    "F1", "Recall", "Precision", "Accuracy",
    "FitTime(s)", "PredTime(s)"
])

for kernel in kernel_values:
    for degree in degree_values:
        for gamma in gamma_values:
            clf = SVC(kernel=kernel, degree=degree, gamma=gamma, random_state=RANDOM_STATE)
            pipe = Pipeline([("prep", preprocess), ("dense", to_dense), ("clf", clf)])

            cv_res = inner_cv_evaluate(pipe, X_train_df, y_train, inner_cv, labels_order)
            test_res = holdout_test_evaluate(pipe, X_train_df, y_train, X_test_df, y_test, labels_order)

            ws_cv_svm.append([
                kernel, degree, gamma,
                str(cv_res["cm_sum"]),
                cv_res["f1_mean"], cv_res["f1_std"],
                cv_res["recall_mean"], cv_res["recall_std"],
                cv_res["precision_mean"], cv_res["precision_std"],
                cv_res["acc_mean"], cv_res["acc_std"],
                cv_res["fit_time_mean"], cv_res["pred_time_mean"]
            ])

            ws_test_svm.append([
                kernel, degree, gamma,
                str(test_res["cm"]),
                test_res["f1"], test_res["recall"], test_res["precision"], test_res["accuracy"],
                test_res["fit_time"], test_res["pred_time"]
            ])

wb_svm.save(svm_path)
print(f"SVM results saved to {svm_path}")


# =============================================================================
# 5) CONFUSION MATRIX VISUALISATION (Best MLP by CV F1_mean, plotted on hold-out test)
# =============================================================================

if best_mlp["test_cm"] is None:
    raise ValueError("No MLP confusion matrix available for visualisation.")

# Use the hold-out test confusion matrix for the best MLP setting
c_mat = best_mlp["test_cm"]
total = np.sum(c_mat)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 24

# Annotate with counts and percentages
labels = [[f"{val:0.0f}\n{(val / total):.2%}" if total > 0 else f"{val:0.0f}\n0.00%"
           for val in row] for row in c_mat]

plt.figure(figsize=(15, 13))
plt.subplots_adjust(left=0.03, right=0.97, top=0.95, bottom=0.03)

ax = sns.heatmap(
    c_mat, annot=labels, cmap="OrRd", fmt="",
    xticklabels=labels_order, yticklabels=labels_order, cbar=True,
    annot_kws={"size": 24}, linewidths=0.5, linecolor="gray"
)

h, mi, lr = best_mlp["params"]
ax.set_title(
    f"Predicted Damage State (MLP hold-out test) | Hidden={h}, MaxIter={mi}, LR={lr}",
    fontweight="bold", fontsize=20
)
ax.tick_params(labeltop=True, labelbottom=False, length=0)
ax.set_ylabel("Actual Damage State", fontweight="bold", fontsize=18)
ax.set_xlabel("")
ax.tick_params(axis="y", pad=10)

# Extra matrix to display recall, precision, and overall accuracy
f_mat = np.zeros((c_mat.shape[0] + 1, c_mat.shape[1] + 1), dtype=float)
with np.errstate(divide="ignore", invalid="ignore"):
    row_sums = np.sum(c_mat, axis=1)
    col_sums = np.sum(c_mat, axis=0)
    diag = np.diag(c_mat)

    # Recall per class (diag / row sum)
    f_mat[:-1, -1] = np.divide(diag, row_sums, out=np.zeros_like(diag, dtype=float), where=row_sums != 0)
    # Precision per class (diag / col sum)
    f_mat[-1, :-1] = np.divide(diag, col_sums, out=np.zeros_like(diag, dtype=float), where=col_sums != 0)
    # Overall accuracy
    f_mat[-1, -1] = (np.trace(c_mat) / total) if total > 0 else 0.0

f_mask = np.ones_like(f_mat, dtype=bool)
f_mask[:, -1] = False
f_mask[-1, :] = False

f_color = np.ones_like(f_mat)
f_color[-1, -1] = 0

f_annot = [[f"{val:0.2%}" for val in row] for row in f_mat]
f_annot[-1][-1] = "Acc.:\n" + f_annot[-1][-1]

sns.heatmap(
    f_color, mask=f_mask, annot=f_annot, fmt="",
    xticklabels=labels_order + ["Recall"],
    yticklabels=labels_order + ["Precision"],
    cmap=ListedColormap(["#f7f7f7", "#d9d9d9"]),
    cbar=False, ax=ax,
    annot_kws={"size": 22}
)

yticklabels = ax.get_yticklabels()
xticklabels = ax.get_xticklabels()
yticklabels[-1].set_size(24)
xticklabels[-1].set_size(24)
ax.set_yticklabels(yticklabels, rotation=90, ha="center")
ax.set_xticklabels(xticklabels, rotation=0, ha="center")

plt.show()

