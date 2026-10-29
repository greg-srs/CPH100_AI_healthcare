"""Part 1.2: preprocessing and three prediction models.

You may use scikit-learn. Do not fit transformations on validation/test data.
"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import  MLPClassifier
from data_utils import model_features
from evaluation import compute_auc

def build_preprocessor(X_train):
    """Return an unfitted preprocessing transformer for diabetes features.

    Numeric columns: median imputation. Categorical columns: most-frequent
    imputation followed by one-hot encoding with unknown categories ignored.
    The returned transformer must be fitted ONLY on the training partition.
    """


    # The caller supplies feature columns, not labels or patient identifiers.
    cat = {"admission_type_id", "discharge_disposition_id", "admission_source_id"}
    excluded = {"readmitted", "target", "encounter_id", "patient_nbr", "age_group"}
    feature_names = [c for c in X_train.columns if c not in excluded]
    numeric = [c for c in X_train[feature_names].select_dtypes(include="number").columns
               if c not in cat]
    categorical = [c for c in feature_names if c not in numeric]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(
        [("numeric", numeric_pipe, numeric),
         ("categorical", categorical_pipe, categorical)],
        remainder="drop",
    )


def build_model(kind, X_train, params=None, seed=42):
    """Return an UNFITTED model pipeline.

    kind is one of 'linear', 'tree', 'neural'.

    TODO:
      - Choose and instantiate one model from each family.
      - Apply the same train-only preprocessing appropriately.
      - Expose tunable hyperparameters via params.
      - Set random seeds where supported.
    """
    if kind not in {'linear', 'tree', 'neural'}:
        raise ValueError(f'Unknown model family: {kind}')

    if params == None:
        params = {}
    
    preprocessor = build_preprocessor(X_train)

    if kind =="linear":
        model = LogisticRegression(**params, random_state=seed, max_iter=1000)
    elif kind == "tree":
        model = RandomForestClassifier(n_estimators=300, 
            n_jobs=-1,random_state=seed, **params)
    elif kind == "neural":
        model = MLPClassifier(hidden_layer_sizes=(64,), early_stopping=True,
            max_iter=200, random_state=seed, **params)
    else: 
        raise NotImplementedError("")

    return Pipeline([
            ("preprocess", preprocessor),
            ("model", model)
        ])



def tune_model(kind, train, val, seed=42):
    """Select hyperparameters using train and validation partitions only.

    TODO: Train candidate models, compute validation AUC, select a candidate,
    and record the search and results. Never select using the test set.
    """
    if kind not in {'linear', 'tree', 'neural'}:
        raise ValueError(f'Unknown model family: {kind}')

    X_train, y_train = model_features(train)
    X_val, y_val = model_features(val)
    hyperparameters = {
        "linear":[{"C": 0.001}, {"C": 0.005}, {"C": 0.01}, {"C": 0.05}, {"C": 0.1}, {"C": 1}],
        "tree":[{"min_samples_leaf": 3}, {"min_samples_leaf": 5}, {"min_samples_leaf": 8}, {"min_samples_leaf": 10}],
        "neural":[{"alpha": 1e-3}, {"alpha": 1e-2}, {"alpha": 1e-1}]
    }
    history = []
    best_auc = 0
    best_model = None
    best_params = None
    for param in hyperparameters[kind]:
        model = build_model(kind, X_train, param, seed)
        model.fit(X_train, y_train)
        auc = compute_auc(y_val, model.predict_proba(X_val)[:,1])
        history.append({"kind":kind, "params": param, "val_auc": auc})
        if auc > best_auc:
            best_auc = auc
            best_model = model
            best_params = param
    return best_model, best_params, history

