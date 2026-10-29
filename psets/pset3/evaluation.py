"""Student implementations for Parts 1.1, 1.2, and 1.4."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
import pandas as pd

def plot_readmission_rates(frame, output_dir):
    """TODO: Plot readmission rates and group sizes by age, gender, race."""
    output_dir = Path(output_dir)                  # texte -> objet chemin
    output_dir.mkdir(parents=True, exist_ok=True)  # crée le dossier si besoin
    overall_rate = frame["target"].mean()

    for pop in ["age", "gender", "race"]:
        df = frame.groupby(pop, dropna=False)["target"].agg(["size", "mean"])
        print(df)
        labels = df.index.astype(str)

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

        axes[0].bar(labels, df["mean"])
        axes[0].axhline(overall_rate, color="red", linestyle="--",
                        label=f"overall ({overall_rate:.3f})")
        axes[0].set_title(f"30-day readmission rate by {pop}")
        axes[0].set_ylabel("Readmission rate")
        axes[0].legend()

        axes[1].bar(labels, df["size"])
        axes[1].set_title(f"Number of encounters by {pop}")
        axes[1].set_ylabel("Encounters")

        for ax in axes:                            # réglages communs aux 2 cases
            ax.set_xlabel(pop)
            ax.tick_params(axis="x", rotation=45)

        fig.tight_layout()
        fig.savefig(output_dir / f"readmission_{pop}.png", dpi=150)
        plt.close(fig)

    
def compute_auc_manual(y_true, probabilities):
    """TODO: Compute ROC AUC from predicted probabilities (not class labels)."""
    probabilities = np.asarray(probabilities)
    y_true = np.asarray(y_true)
    pos = probabilities[y_true == 1]
    neg = probabilities[y_true == 0]
    score = 0
    for p in pos:
        for n in neg:
            if p>n:
                score+=1
            elif p==n:
                score+=0.5
    return score/(len(pos)*len(neg))

def compute_auc(y_true, probabilities):
    """TODO: Compute ROC AUC from predicted probabilities (not class labels)."""
    return roc_auc_score(y_true, probabilities)


def auc_confidence_interval(y_true, probabilities, seed=42, confidence=0.95,
                            n_bootstrap=1000):
    """Return (lower, upper) percentile bootstrap CI for ROC AUC.

    Resample observation pairs, skipping single-class samples. This assumes
    independently sampled evaluation rows; repeated patient encounters may
    require patient-level clustered resampling instead.
    """

    y = np.asarray(y_true)
    p = np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p):
        raise ValueError("y_true and probabilities must be equal-length vectors")
    if len(y) == 0 or not np.isfinite(p).all():
        raise ValueError("Nonempty finite probabilities are required")
    if len(np.unique(y)) != 2:
        raise ValueError("ROC AUC requires both outcome classes")
    if not 0 < confidence < 1 or n_bootstrap < 1:
        raise ValueError("Invalid confidence or n_bootstrap")
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[indices])) < 2:
            continue
        aucs.append(roc_auc_score(y[indices], p[indices]))
    if not aucs:
        raise ValueError("No bootstrap samples contained both classes")
    alpha = (1 - confidence) / 2
    return tuple(float(v) for v in np.quantile(aucs, [alpha, 1 - alpha]))


def compare_models(results, output_dir):
    """TODO: Plot model AUCs with confidence intervals and describe variation."""
    import pandas as pd
    results = pd.DataFrame(results)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    kinds = ["linear", "tree", "neural"]
    seeds = sorted(results["seed"].unique())

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.12  # horizontal shift between seeds inside one model family
    for j, seed in enumerate(seeds):
        rows = results[results["seed"] == seed].set_index("kind").loc[kinds]
        x = np.arange(len(kinds)) + (j - (len(seeds) - 1) / 2) * width
        # yerr = distance from the point to each CI bound
        yerr = [rows["test_auc"] - rows["ci_low"], rows["ci_high"] - rows["test_auc"]]
        ax.errorbar(x, rows["test_auc"], yerr=yerr, fmt="o", capsize=3,
                    alpha=0.7, label=f"seed {seed}")

    # mean test AUC over seeds, drawn as a horizontal bar per family
    means = results.groupby("kind")["test_auc"].mean().loc[kinds]
    for i, m in enumerate(means):
        ax.hlines(m, i - 0.35, i + 0.35, colors="black", linewidth=2)
        ax.text(i + 0.37, m, f"mean {m:.3f}", va="center", fontsize=9)

    ax.set_xticks(range(len(kinds)))
    ax.set_xticklabels(["Logistic regression", "Random forest", "MLP"])
    ax.set_ylabel("Test AUC (95% bootstrap CI)")
    ax.set_title("Test AUC per model family over patient-disjoint splits")
    ax.legend(fontsize=8, ncol=len(seeds))
    fig.tight_layout()
    fig.savefig(output_dir / "model_comparison.png", dpi=150)
    plt.close(fig)

    summary = results.groupby("kind")["test_auc"].agg(["mean", "std", "min", "max"]).loc[kinds]
    print(summary.round(4))
    return summary


def analyze_age_shift(frame, seed=42):
    """TODO: Train on under-50 patients, evaluate on held-out under-50 and
    on 50-plus patients. Investigate outcome/feature differences and avoid
    using the older group's outcomes in training or model selection.
    """
    from data_utils import add_age_group, split_data, model_features
    from models import tune_model

    frame = pd.DataFrame(frame)
    frame = add_age_group(frame)
    mask_young = frame["age_group"] == "under_50"
    mask_old = frame["age_group"] == "50_plus"
    young = frame[mask_young]
    old = frame[mask_old]

    train_young, val_young, test_young = split_data(young, seed, patient_disjoint= True)
    train_old, val_old, test_old = split_data(old, seed, patient_disjoint= True)

    model_young, params_young, history_young = tune_model("linear", train_young, val_young, seed=seed)
    X_test_young, y_test_young = model_features(test_young)
    proba_young = model_young.predict_proba(X_test_young)[:,1]
    auc_young = compute_auc(y_test_young, proba_young)
    ci_low_young, ci_high_young = auc_confidence_interval(y_test_young, proba_young, seed=seed)


    n = len(test_old)
    mask =test_old["patient_nbr"].isin(list(train_young["patient_nbr"]) + list(val_young["patient_nbr"]))
    test_old = test_old[~mask]
    num = n-len(test_old)
    print(num)

    X_test_old, y_test_old = model_features(test_old)
    proba_old = model_young.predict_proba(X_test_old)[:,1]
    auc_old = compute_auc(y_test_old, proba_old)
    ci_low_old, ci_high_old = auc_confidence_interval(y_test_old, proba_old, seed=seed)



    model_old, params_old, history_old = tune_model("linear", train_old, val_old, seed=seed)
    proba_old2 = model_old.predict_proba(X_test_old)[:,1]
    auc_old2 = compute_auc(y_test_old, proba_old2)
    ci_low_old2, ci_high_old2 = auc_confidence_interval(y_test_old, proba_old2, seed=seed)

    results = [
        {"train_group": "under_50", "eval_group": "under_50", "n": len(test_young), "n_positive": int(y_test_young.sum()), "auc": auc_young, "ci_low": ci_low_young, "ci_high": ci_high_young},
        {"train_group": "under_50", "eval_group": "50_plus", "n": len(test_old), "n_positive": int(y_test_old.sum()), "auc": auc_old, "ci_low": ci_low_old, "ci_high": ci_high_old},
        {"train_group": "50_plus", "eval_group": "50_plus", "n": len(test_old), "n_positive": int(y_test_old.sum()), "auc": auc_old2, "ci_low": ci_low_old2, "ci_high": ci_high_old2}
    ]

    return results, model_young, model_old

def describe_age_groups(frame, output_dir):
    """Compare numeric features and primary diagnosis groups between age groups."""
    from llm_eval import diag_category
    columns = ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications", 
        "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses"]
    
    df = frame.copy()
    table1 = df.groupby("age_group")[columns].mean().T

    df["diag_group"] = df["diag_1"].map(diag_category)
    table2 = df.groupby("age_group")["diag_group"].value_counts(normalize=True).unstack(0)

    output_dir = Path(output_dir)
    table1.to_csv(output_dir / "age_groups_numeric_means.csv" )
    table2.to_csv(output_dir / "age_groups_diagnoses.csv")

    print(table1)
    print(table2)
    return table1, table2


def compare_coefficients(model_young, model_old, output_dir, top=10):
    """Compare the logistic regression coefficients of the under-50 and 50+ models."""
    name_coef_young = model_young.named_steps["preprocess"].get_feature_names_out()
    name_coef_old = model_old.named_steps["preprocess"].get_feature_names_out()

    coef_young = model_young.named_steps["model"].coef_[0]
    coef_old = model_old.named_steps["model"].coef_[0]

    serie_young = pd.Series(coef_young, index=name_coef_young)
    serie_old = pd.Series(coef_old, index=name_coef_old)

    table = pd.concat({"under_50": serie_young, "50_plus": serie_old}, axis=1)
    numeric_table = table[table.index.str.startswith("numeric__")]

    top_young = table.loc[table["under_50"].abs().sort_values(ascending=False).index[:top]]
    top_old = table.loc[table["50_plus"].abs().sort_values(ascending=False).index[:top]]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    numeric_table.to_csv(output_dir / "age_shift_coef_numeric.csv")
    top_young.to_csv(output_dir / "age_shift_coef_top_under_50.csv")
    top_old.to_csv(output_dir / "age_shift_coef_top_50_plus.csv")

    return numeric_table, top_young, top_old



