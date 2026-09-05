import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

# Resolve paths relative to THIS file's own location, not the
# current working directory — so it doesn't matter where you run
# `python 04_model_training.py` from. This mirrors how app.py finds
# its data file, so training and the dashboard always agree on the
# same CSV.
# Resolve paths relative to the PROJECT ROOT, not this script's own
# folder — this script lives in a `notebooks/` subfolder, while
# `data/`, the trained model, and the metadata file all live one
# level up, next to app.py. Using .parent.parent walks up from
# notebooks/04_model_training.py to the project root.
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "featured_jobs.csv"

DESCRIPTION_COLUMN = "Job Description"

# Output files
MODEL_OUT = BASE_DIR / "salary_prediction_model.pkl"
META_OUT = BASE_DIR / "salary_prediction_meta.json"

TARGET = "Avg_Salary"


# ============================================================
# TOOL / PLATFORM PATTERNS
# ============================================================
# Same style as the existing "languages" dict used earlier in the
# pipeline: dict key becomes the column name, value is the regex
# used to search the job description. These get added as *_yn
# columns alongside the language columns already in the CSV.

tools = {
    "AWS": r"\bAWS\b|\bAmazon Web Services\b",
    "Azure": r"\bAzure\b",
    "Spark": r"\bSpark\b",
    "Hadoop": r"\bHadoop\b",
    "Tableau": r"\bTableau\b",
    "Power_BI": r"\bPower\s?BI\b",
    "Docker": r"\bDocker\b",
    "Kubernetes": r"\bKubernetes\b|\bK8s\b",
    "Machine_Learning": r"\bMachine Learning\b|\bML\b",
    "Big_Data": r"\bBig Data\b",
}


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 70)
print("SALARY PREDICTION MODEL TRAINING")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print(f"\nReading from: {DATA_PATH}")
print(f"Loaded dataset: {df.shape}")

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' was not found in {DATA_PATH}."
    )

# Remove rows where target salary is missing
df = df.dropna(subset=[TARGET]).copy()

print(f"Rows after removing missing salaries: {len(df)}")


# ============================================================
# 1B. ADD TOOL / PLATFORM SKILL COLUMNS
# ============================================================
# Detects mentions of common data-science tools and platforms in
# the Job Description and adds them as binary *_yn columns,
# alongside the language columns (Python_yn, SQL_yn, etc.) that
# already exist in the CSV. The updated dataframe is written back
# to DATA_PATH so the Streamlit dashboard (which reads the same
# file independently) also picks up these columns.

print("\n" + "=" * 70)
print("ADDING TOOL SKILL COLUMNS")
print("=" * 70)

if DESCRIPTION_COLUMN not in df.columns:
    raise ValueError(
        f"Expected a '{DESCRIPTION_COLUMN}' column to detect "
        f"tools from, but it was not found in {DATA_PATH}.\n"
        f"Available columns: {list(df.columns)}"
    )

added_tool_columns = []

for tool_name, pattern in tools.items():

    column = f"{tool_name}_yn"

    df[column] = (
        df[DESCRIPTION_COLUMN]
        .str.contains(
            pattern,
            case=False,
            regex=True,
            na=False,
        )
        .astype(int)
    )

    added_tool_columns.append(column)

    print(
        f"{column:<22} "
        f"-> {df[column].sum()} postings matched"
    )

df.to_csv(DATA_PATH, index=False)

print(f"\nSaved updated dataset back to: {DATA_PATH}")
print(f"Dataset shape after adding tool columns: {df.shape}")


# ============================================================
# 2. FEATURE SELECTION
# ============================================================

categorical_features = [
    "Job_Simplified",
    "Job_State",
    "Experience_Group",
]

numeric_base_features = [
    "Rating",
    "Company_Age",
    "Remote_Job",
]

# Automatically detect skill/tool columns.
# Example:
# Python_yn
# SQL_yn
# Machine_Learning_yn
# R_yn
# Tableau_yn
skill_features = [
    c for c in df.columns
    if c.strip().lower().endswith("_yn")
]

numeric_features = numeric_base_features + skill_features

all_features = categorical_features + numeric_features

# Check that every required column exists
missing = [
    c for c in all_features
    if c not in df.columns
]

if missing:
    raise ValueError(
        f"\nMissing required columns:\n{missing}\n\n"
        f"Available columns are:\n{list(df.columns)}"
    )

print("\nFeatures being used:")
print(f"Categorical features : {len(categorical_features)}")
print(f"Base numeric features: {len(numeric_base_features)}")
print(f"Skill features       : {len(skill_features)}")
print(f"Total input features : {len(all_features)}")

print("\nSkill columns detected:")
print(skill_features)


# ============================================================
# 3. CREATE X AND y
# ============================================================

X = df[all_features].copy()
y = df[TARGET].copy()

print(f"\nX shape: {X.shape}")
print(f"y shape: {y.shape}")


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
)

print("\nTrain / Test split:")
print(f"Training rows: {len(X_train)}")
print(f"Testing rows : {len(X_test)}")


# ============================================================
# 5. PREPROCESSING
# ============================================================

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ]
)

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            categorical_transformer,
            categorical_features,
        ),
        (
            "numeric",
            numeric_transformer,
            numeric_features,
        ),
    ]
)


# ============================================================
# 6. DEFINE MODELS
# ============================================================

models = {

    "Ridge": Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                Ridge(alpha=10),
            ),
        ]
    ),

    "Random Forest": Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=8,
                    min_samples_leaf=3,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    ),

    "Gradient Boosting": Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                GradientBoostingRegressor(
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.05,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    ),
}


# ============================================================
# 7. 5-FOLD CROSS-VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("5-FOLD CROSS-VALIDATION")
print("=" * 70)

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)

cv_results = []

for name, model in models.items():

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )

    cv_mae = -scores.mean()
    cv_std = scores.std()

    cv_results.append(
        {
            "Model": name,
            "CV_MAE": cv_mae,
            "CV_MAE_std": cv_std,
        }
    )

    print(
        f"{name:<20} "
        f"MAE: {cv_mae:.2f}K "
        f"(± {cv_std:.2f})"
    )

cv_df = (
    pd.DataFrame(cv_results)
    .sort_values("CV_MAE")
    .reset_index(drop=True)
)


# ============================================================
# 8. TRAIN BASELINE MODELS
# ============================================================

print("\n" + "=" * 70)
print("BASELINE TEST RESULTS")
print("=" * 70)

test_results = []
fitted_models = {}

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    fitted_models[name] = model

    # Predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # Metrics
    train_mae = mean_absolute_error(
        y_train,
        train_pred,
    )

    test_mae = mean_absolute_error(
        y_test,
        test_pred,
    )

    test_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            test_pred,
        )
    )

    test_r2 = r2_score(
        y_test,
        test_pred,
    )

    train_test_gap = test_mae - train_mae

    test_results.append(
        {
            "Model": name,
            "Train_MAE": train_mae,
            "Test_MAE": test_mae,
            "Train_Test_Gap": train_test_gap,
            "Test_RMSE": test_rmse,
            "Test_R2": test_r2,
        }
    )

    print(f"Train MAE      : {train_mae:.2f}K")
    print(f"Test MAE       : {test_mae:.2f}K")
    print(f"Test RMSE      : {test_rmse:.2f}K")
    print(f"Test R²        : {test_r2:.3f}")
    print(f"Train-Test Gap : {train_test_gap:.2f}K")


results_df = (
    pd.DataFrame(test_results)
    .sort_values("Test_MAE")
    .reset_index(drop=True)
)

print("\n")
print(results_df.to_string(index=False))


# ============================================================
# 9. SELECT BEST BASELINE MODEL
# ============================================================

best_name = results_df.iloc[0]["Model"]

best_model = fitted_models[best_name]

baseline_test_mae = results_df.iloc[0]["Test_MAE"]

print("\n" + "=" * 70)
print(f"BEST BASELINE MODEL: {best_name}")
print("=" * 70)


# ============================================================
# 10. HYPERPARAMETER TUNING
# ============================================================

tuned_model = None
tuned_test_mae = None
tuning_used = False
best_params = None

if best_name != "Ridge":

    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING")
    print("=" * 70)

    base_model = models[best_name].named_steps["model"]

    if best_name == "Random Forest":

        param_grid = {
            "model__n_estimators": [
                200,
                300,
                400,
            ],
            "model__max_depth": [
                6,
                8,
                12,
            ],
            "model__min_samples_leaf": [
                2,
                3,
                5,
            ],
        }

    else:

        # Gradient Boosting
        param_grid = {
            "model__n_estimators": [
                150,
                200,
                300,
            ],
            "model__max_depth": [
                2,
                3,
                4,
            ],
            "model__learning_rate": [
                0.03,
                0.05,
                0.10,
            ],
        }

    grid_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", base_model),
        ]
    )

    grid_search = GridSearchCV(
        estimator=grid_pipeline,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=5,
        n_jobs=-1,
    )

    grid_search.fit(
        X_train,
        y_train,
    )

    best_params = grid_search.best_params_

    tuned_cv_mae = -grid_search.best_score_

    tuned_predictions = grid_search.predict(X_test)

    tuned_test_mae = mean_absolute_error(
        y_test,
        tuned_predictions,
    )

    print("\nBest parameters:")
    print(best_params)

    print(
        f"\nTuned CV MAE : {tuned_cv_mae:.2f}K"
    )

    print(
        f"Tuned Test MAE: {tuned_test_mae:.2f}K"
    )

    print(
        f"Baseline Test MAE: {baseline_test_mae:.2f}K"
    )

    # IMPORTANT:
    # Only use tuned model if it actually improves
    # the untouched test set.
    if tuned_test_mae < baseline_test_mae:

        best_model = grid_search.best_estimator_

        tuning_used = True

        print(
            "\n✓ Tuning improved the test MAE."
        )
        print(
            "✓ Using tuned model."
        )

    else:

        print(
            "\nTuning did not improve the baseline."
        )
        print(
            "Using the baseline model."
        )


# ============================================================
# 11. FINAL MODEL EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL MODEL EVALUATION")
print("=" * 70)

final_pred = best_model.predict(X_test)

final_mae = mean_absolute_error(
    y_test,
    final_pred,
)

final_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        final_pred,
    )
)

final_r2 = r2_score(
    y_test,
    final_pred,
)


# ============================================================
# 12. ACTUAL VS PREDICTED
# ============================================================

comparison = pd.DataFrame(
    {
        "Actual_Salary": y_test.values,
        "Predicted_Salary": final_pred,
    }
)

comparison["Absolute_Error"] = (
    comparison["Actual_Salary"]
    - comparison["Predicted_Salary"]
).abs()

comparison["Error_Percent"] = (
    comparison["Absolute_Error"]
    / comparison["Actual_Salary"].replace(0, np.nan)
) * 100

comparison = comparison.sort_values(
    "Absolute_Error"
).reset_index(drop=True)


print("\nActual vs Predicted Salary:")

print(
    comparison.head(30).round(2).to_string(
        index=False
    )
)


# ============================================================
# 13. PREDICTION ACCURACY RANGES
# ============================================================

errors = comparison["Absolute_Error"]

within_10 = (
    errors <= 10
).mean() * 100

within_20 = (
    errors <= 20
).mean() * 100

within_30 = (
    errors <= 30
).mean() * 100


print("\n" + "=" * 70)
print("PREDICTION ACCURACY")
print("=" * 70)

print(
    f"Within $10K : {within_10:.1f}%"
)

print(
    f"Within $20K : {within_20:.1f}%"
)

print(
    f"Within $30K : {within_30:.1f}%"
)

print(
    f"\nMean Absolute Error: ${final_mae:.1f}K"
)

print(
    f"RMSE               : ${final_rmse:.1f}K"
)

print(
    f"R²                 : {final_r2:.3f}"
)


# ============================================================
# 14. FEATURE IMPORTANCE
# ============================================================

if hasattr(
    best_model.named_steps["model"],
    "feature_importances_",
):

    print("\n" + "=" * 70)
    print("TOP 15 FEATURE IMPORTANCES")
    print("=" * 70)

    feature_names = (
        best_model
        .named_steps["preprocessor"]
        .get_feature_names_out()
    )

    importances = (
        best_model
        .named_steps["model"]
        .feature_importances_
    )

    importance_df = (
        pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": importances,
            }
        )
        .sort_values(
            "Importance",
            ascending=False,
        )
        .head(15)
        .reset_index(drop=True)
    )

    print(
        importance_df.to_string(
            index=False
        )
    )


# ============================================================
# 15. SAVE MODEL
# ============================================================

joblib.dump(
    best_model,
    MODEL_OUT,
)


# ============================================================
# 16. SAVE METADATA FOR STREAMLIT
# ============================================================

meta = {
    "model_name": best_name,

    "target": TARGET,

    "data_path": str(DATA_PATH),

    "n_rows": int(len(df)),

    "n_features": int(len(all_features)),

    "categorical_features": categorical_features,

    "numeric_base_features": numeric_base_features,

    "skill_features": skill_features,

    "all_features": all_features,

    "categorical_options": {
        col: sorted(
            df[col]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        for col in categorical_features
    },

    "numeric_ranges": {
        col: [
            float(df[col].min()),
            float(df[col].max()),
        ]
        for col in numeric_base_features
        if pd.api.types.is_numeric_dtype(
            df[col]
        )
    },

    "baseline_test_mae": float(
        baseline_test_mae
    ),

    "test_mae": float(
        final_mae
    ),

    "test_rmse": float(
        final_rmse
    ),

    "test_r2": float(
        final_r2
    ),

    "within_10k_percent": float(
        within_10
    ),

    "within_20k_percent": float(
        within_20
    ),

    "within_30k_percent": float(
        within_30
    ),

    "tuning_used": tuning_used,

    "best_params": best_params,
}


with open(
    META_OUT,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        meta,
        f,
        indent=2,
    )


# ============================================================
# 17. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("MODEL TRAINING COMPLETE")
print("=" * 70)

print(
    f"\nSelected model : {best_name}"
)

print(
    f"Test MAE       : ${final_mae:.1f}K"
)

print(
    f"Test RMSE      : ${final_rmse:.1f}K"
)

print(
    f"Test R²        : {final_r2:.3f}"
)

print(
    f"Within $10K    : {within_10:.1f}%"
)

print(
    f"Within $20K    : {within_20:.1f}%"
)

print(
    f"Within $30K    : {within_30:.1f}%"
)

print(
    f"\nSaved model    : {MODEL_OUT}"
)

print(
    f"Saved metadata : {META_OUT}"
)

print("\nDone.")