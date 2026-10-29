"""Starter orchestration; no model results are assumed or supplied."""
import argparse
from pathlib import Path
from data_utils import load_data, add_age_group, split_data, make_llm_sample, model_features
from evaluation import auc_confidence_interval, compute_auc, plot_readmission_rates, compare_models 
from models import tune_model
import pandas as pd

def run_experiments(frame, seeds):
    results = []
    search_log = [] 
    kinds = ["linear", "tree", "neural"]
    for seed in seeds:
        train, val, test = split_data(frame, seed, patient_disjoint=True)
        X_test, y_test = model_features(test)
        for kind in kinds:
            model, params, hist = tune_model(kind, train, val, seed)
            search_log.extend({"seed": seed, **trial} for trial in hist)
            val_auc = max(dict["val_auc"] for dict in hist)
            prediction_proba = model.predict_proba(X_test)[:,1]
            test_auc = compute_auc(y_test, prediction_proba) 
            ci_low, ci_high = auc_confidence_interval(y_test, prediction_proba)

            results.append({"seed": seed, "kind":kind, "best_params":params, 
                "val_auc":val_auc, "test_auc":test_auc, "ci_low":ci_low, "ci_high":ci_high})
        
        pd.DataFrame(results).to_csv("outputs/model_results.csv", index=False)
        pd.DataFrame(search_log).to_csv("outputs/model_search_log.csv", index=False)
    return results



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/diabetic_data.csv")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--explore", action="store_true")
    ap.add_argument("--make-llm-sample", action="store_true")
    ap.add_argument("--sample-size", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patient-disjoint", action="store_true")
    ap.add_argument("--train",  action="store_true")
    args = ap.parse_args()
    frame = add_age_group(load_data(args.data))
    train, val, test = split_data(frame, seed=args.seed, patient_disjoint=args.patient_disjoint)
    if args.inspect:
        print(f"Full dataset: {len(frame)} rows; positives: {frame.target.mean():.3f}")
        for label, part in (("train", train), ("val", val), ("test", test)):
            print(f"{label}: {len(part)} rows; positives: {part.target.mean():.3f}")
        print("Readmission by age:")
        print(frame.groupby("age", dropna=False).target.agg(["size", "mean"]))
    if args.make_llm_sample:
        sample = make_llm_sample(test, n=args.sample_size, seed=args.seed)
        output = Path("outputs/llm_sample.csv")
        output.parent.mkdir(exist_ok=True)
        sample.to_csv(output, index=False)
        print(f"Saved {len(sample)} held-out encounters to {output}")
    if args.explore:
        plot_readmission_rates(frame, "outputs")
    if args.train:
        results = run_experiments(frame, seeds = [42,0,1,2,3])
        compare = compare_models(results, "outputs")
        print(compare)
    if not args.inspect and not args.make_llm_sample and not args.explore and not args.train:
        print("TODO: Fit and compare three model classes on train/val/test splits.")
        print("See models.py, evaluation.py and README.md.")


if __name__ == "__main__":
    main()
