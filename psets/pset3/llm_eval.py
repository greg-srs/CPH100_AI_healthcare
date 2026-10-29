"""Evaluate an LLM with GroqCloud; repeatable prompts and JSONL caching.

Set GROQ_API_KEY in your environment. Use --dry-run to preview without calls.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time
import pandas as pd
from data_utils import EXCLUDE_FROM_FEATURES
import csv

DEFAULT_MODEL = "openai/gpt-oss-20b"
PRIVATE_COLUMNS = EXCLUDE_FROM_FEATURES | {"target", "age_group"}
MEDICATIONS_COLUMNS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone",
]
DOSE_WORDS = {
    "Up": "dose increased",
    "Down": 'dose decreased',
    "Steady": "dose unchanged"
}
UNKNOWN_LABELS = {"NULL", "Not Mapped", "Not Available", "Unknown/Invalid"}
CHANGE_WORDS = {"Ch": "yes", "No": "no"}

def diag_category(code):
    """Map an ICD-9 code to one of the 9 disease groups of Strack et al. (2014)."""
    if pd.isna(code):
        return "not recorded"
    code = str(code)
    if code[0]=="V" or code[0]== "E":
        return "other"
    x = float(code)
    if (390<=x and x < 460) or int(x) == 785:
        return "circulatory"
    elif (460 <= x and  x < 520) or int(x) == 786:
        return "respiratory"
    elif (520 <= x and  x < 580) or int(x) == 787:
        return "digestive"
    elif int(x) == 250:
        return "diabetes"
    elif 800 <= x and  x < 1000:
        return "injury"
    elif 710 <= x and  x < 740:
        return "musculoskeletal"
    elif (580 <= x and  x < 630) or int(x) == 788:
        return "genitourinary"
    elif 140 <= x and  x < 240:
        return "neoplasms"
    else: 
        return "other"   

def load_id_mappings(path):
    """Read IDS_mapping.csv into {column: {code: description}}."""
    with open(path) as f:
        mappings = {}
        table = None
        for row in csv.reader(f):
            code, description = row[0], row[1]
            if code == "":
                continue
            elif description == "description":
                table = code
                mappings[table] = {}
            else: 
                mappings[table][int(code)] = description.strip()
        return mappings

ID_MAPPINGS = load_id_mappings("data/IDS_mapping.csv")


def describe_medications(row):
    """List the diabetes drugs actually taken, with their dose change (or 'none')."""
    taken = []
    for name in MEDICATIONS_COLUMNS:
        value = row[name]
        if value == "No" or pd.isna(value):
            continue
        taken.append(f"{name} ({DOSE_WORDS[value]})")

    if taken == []:
        return "none"
    return ", ".join(taken)

def readable(value, unknown="not recorded"):
    """Return the value as text, or `unknown` if it is missing or a placeholder like NULL."""
    if pd.isna(value) or value in UNKNOWN_LABELS:
        return unknown
    else : 
        return str(value)

def read_id(row, column):
    return readable(ID_MAPPINGS[column][row[column]])

def make_prompt(row: dict, format_name: str) -> str:
    """Build two example representations; students design the third."""


    # Never provide the label, derived outcomes, or patient/encounter identifiers.
    fields = []
    for name, value in row.items():
        if name in PRIVATE_COLUMNS or name.startswith('Unnamed:'):
            continue
        if pd.isna(value):
            value = 'missing'
        fields.append((name, str(value)))

    instruction = (
        'Estimate the probability that this patient will be readmitted to a '
        'hospital within 30 days after discharge. Return only a single number '
        'between 0 and 1 (inclusive), with no explanation.\n\n'
    )
    if format_name == 'structured':
        representation = 'Patient encounter features (key: value):\n' + '\n'.join(
            f'{name}: {value}' for name, value in fields
        )
    elif format_name == "natural":
        representation = (
            'Patient encounter description: The recorded attributes are '
            + '; '.join(f'{name.replace("_", " ")} is {value}' for name, value in fields)
            + '.'
        )
    elif format_name == "alternative":
        lines = [
            f"Patient: {readable(row['race'])}, {row['gender']}, and aged {row['age']}.",
            f"Admission: {read_id(row, 'admission_type_id')} admission, admitted from: {read_id(row, 'admission_source_id')}. Admitting physician speciality: {readable(row['medical_specialty'])}. Length of stay: {row['time_in_hospital']} days.",
            f"Discharge: {read_id(row, 'discharge_disposition_id')}.",
            f"Diagnoses: primary = {diag_category(row['diag_1'])} (ICD-9 {readable(row['diag_1'])}); secondary = {diag_category(row['diag_2'])} (ICD-9 {readable(row['diag_2'])}); additional = {diag_category(row['diag_3'])} (ICD-9 {readable(row['diag_3'])}). Total number of diagnoses: {row['number_diagnoses']}.",
            f"During the stay: {row['num_lab_procedures']} lab tests, {row['num_procedures']} procedures, {row['num_medications']} distinct medications.",
            f"Previous year: {row['number_inpatient']} inpatient admissions, {row['number_emergency']} emergency visits, {row['number_outpatient']} outpatient visits.",
            f"Glucose tests: max serum glucose {readable(row['max_glu_serum'], unknown='not measured')}, HbA1c {readable(row['A1Cresult'], unknown='not measured')}.",
            f"Diabetes medications: {describe_medications(row)}. Diabetes medication changed during stay: {CHANGE_WORDS[row['change']]}. On diabetes medication: {row['diabetesMed']}."
        ]
        representation = "\n".join(lines)
    else : 
        raise ValueError('Unknown format')

    return instruction + representation


def parse_probability(text: str):
    """Strictly parse one probability; invalid model responses become NaN."""
    if not re.fullmatch(r"\s*(?:0(?:\.\d+)?|1(?:\.0+)?)\s*", text):
        return float("nan")
    return float(text.strip())


def cached_call(client, model, prompt, cache_path, max_retries=3):
    key = hashlib.sha256((model + "\n" + prompt).encode()).hexdigest()
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        with cache_path.open() as f:
            for line in f:
                item = json.loads(line)
                if item["key"] == key:
                    return item["response"]
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model, temperature=0, max_tokens=1024,
                messages=[{"role": "user", "content": prompt}])
            response = completion.choices[0].message.content or ""
            with cache_path.open("a") as f:
                f.write(json.dumps({"key": key, "model": model,
                                    "response": response}) + "\n")
            
            time.sleep(8)  #not to reach the limit of prompt per minute
            return response
        except Exception as exc:
            # For real deployments, inspect status codes/retry-after rather than
            # retrying all errors. Never print or log your API key.
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("Unreachable")


FORMATS = ["structured", "natural", "alternative"]


def summarize_predictions(source, y_true, probabilities, age_group):
    """AUC with bootstrap CI for all encounters and for each age group.

    Invalid LLM answers (NaN probability) are counted and excluded.
    """
    import numpy as np
    from evaluation import compute_auc, auc_confidence_interval

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)
    age_group = np.asarray(age_group)
    rows = []
    for group in ["all", "under_50", "50_plus"]:
        in_group = np.ones(len(y_true), bool) if group == "all" else age_group == group
        valid = in_group & ~np.isnan(probabilities)
        y, p = y_true[valid], probabilities[valid]
        row = {"source": source, "group": group, "n": int(in_group.sum()),
               "n_invalid": int((in_group & np.isnan(probabilities)).sum()),
               "n_positive": int(y.sum()), "mean_prob": float(p.mean())}
        if len(np.unique(y)) == 2:
            row["auc"] = compute_auc(y, p)
            row["ci_low"], row["ci_high"] = auc_confidence_interval(y, p)
        rows.append(row)
    return rows


def analyze_llm(sample_path="outputs/llm_sample.csv", data_path="data/diabetic_data.csv",
                model_results_path="outputs/model_results_seed42.csv", seed=42):
    """Compare the 3 LLM prompt formats and the trained models on the same sample.

    The LLM sample comes from the test set of the patient-disjoint split with
    this seed, so the trained models are refitted on the train set of that same
    split with the hyperparameters selected on validation for that seed.
    """
    import ast
    from data_utils import load_data, add_age_group, split_data, model_features
    from models import build_model

    sample = pd.read_csv(sample_path)
    rows = []

    # 1) LLM answers, one file per prompt format
    for fmt in FORMATS:
        answers = pd.read_csv(Path("outputs") / f"llm_{fmt}.csv")
        assert (answers["target"].values == sample["target"].values).all(), "sample order mismatch"
        rows += summarize_predictions(f"LLM {fmt}", answers["target"], answers["probability"],
                                      answers["age_group"])

    # 2) Trained models, refitted on the train set of the same split
    frame = add_age_group(load_data(data_path))
    train, _, _ = split_data(frame, seed=seed, patient_disjoint=True)
    assert not set(train["patient_nbr"]) & set(sample["patient_nbr"]), "sample patient in train"
    X_train, y_train = model_features(train)
    X_sample, y_sample = model_features(sample)
    best = pd.read_csv(model_results_path)
    predictions = sample[["encounter_id", "target", "age_group"]].copy()
    for _, result in best.iterrows():
        params = ast.literal_eval(result["best_params"])  # "{'C': 0.01}" -> {'C': 0.01}
        model = build_model(result["kind"], X_train, params=params, seed=seed)
        model.fit(X_train, y_train)
        predictions[result["kind"]] = model.predict_proba(X_sample)[:, 1]
        rows += summarize_predictions(f"model {result['kind']}", y_sample,
                                      predictions[result["kind"]], sample["age_group"])

    summary = pd.DataFrame(rows)
    summary.to_csv("outputs/llm_vs_models.csv", index=False)
    predictions.to_csv("outputs/model_predictions_llm_sample.csv", index=False)
    print(summary.round(3).to_string(index=False))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", action="store_true",
                    help="Compare the 3 formats and the trained models (no API calls)")
    ap.add_argument("--sample", default="outputs/llm_sample.csv")
    ap.add_argument("--format", choices=["structured", "natural", "alternative"], default="structured")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="Process first N sampled encounters")
    args = ap.parse_args()
    if args.analyze:
        analyze_llm(sample_path=args.sample)
        return
    frame = pd.read_csv(args.sample)
    if args.limit is not None:
        frame = frame.head(args.limit)
    client = None
    if not args.dry_run:
        from groq import Groq
        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY is not set; use --dry-run to preview")
        client = Groq()
    output = Path("outputs") / f"llm_{args.format}.csv"
    rows = []
    for i, (_, record) in enumerate(frame.iterrows()):
        prompt = make_prompt(record.to_dict(), args.format)
        if args.dry_run:
            if i == 0:
                print(prompt)
            continue
        reply = cached_call(client, args.model, prompt, "outputs/llm_cache.jsonl")
        rows.append({"sample_row": i, "target": record["target"],
                     "age_group": record.get("age_group", ""),
                     "response": reply, "probability": parse_probability(reply)})
        # Save progress even if later requests fail.
        pd.DataFrame(rows).to_csv(output, index=False)
    if args.dry_run:
        print(f"Dry run completed for {len(frame)} encounters; no API calls made")
    else:
        print(f"Saved {len(rows)} predictions to {output}")


if __name__ == "__main__":
    main()
