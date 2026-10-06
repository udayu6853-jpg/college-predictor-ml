import pandas as pd
import os
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# ============================================================
# FILE PATHS
# ============================================================

KCET_FILE = os.path.join(
    "dataset",
    "kcet_colleges.csv"
)

EAMCET_FILE = os.path.join(
    "dataset",
    "eamcet_colleges.csv"
)

MODEL_FOLDER = "model"

MODEL_FILE = os.path.join(
    MODEL_FOLDER,
    "college_predictor.pkl"
)

PERFORMANCE_FILE = os.path.join(
    MODEL_FOLDER,
    "model_performance.pkl"
)


# ============================================================
# LOAD DATASETS
# ============================================================

print()
print("==============================================")
print("COLLEGE PREDICTOR - ML MODEL TRAINING")
print("==============================================")
print()

print("Loading KCET dataset...")

kcet_df = pd.read_csv(
    KCET_FILE
)

print(
    f"KCET records: {len(kcet_df)}"
)


print()
print("Loading EAMCET dataset...")

eamcet_df = pd.read_csv(
    EAMCET_FILE
)

print(
    f"EAMCET records: {len(eamcet_df)}"
)


# ============================================================
# REMOVE UNNAMED COLUMNS
# ============================================================

kcet_df = kcet_df.loc[
    :,
    ~kcet_df.columns.str.contains(
        "^Unnamed",
        case=False
    )
]

eamcet_df = eamcet_df.loc[
    :,
    ~eamcet_df.columns.str.contains(
        "^Unnamed",
        case=False
    )
]


# ============================================================
# ADD EXAM COLUMN
# ============================================================

kcet_df["Exam"] = "KCET"

eamcet_df["Exam"] = "EAMCET"


# ============================================================
# COMBINE DATASETS
# ============================================================

data = pd.concat(
    [
        kcet_df,
        eamcet_df
    ],
    ignore_index=True,
    sort=False
)


print()
print(
    f"Combined records: {len(data)}"
)


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

data.columns = (
    data.columns
    .astype(str)
    .str.strip()
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "College Name",
    "Branch",
    "Cutoff Rank",
    "Rating",
    "Management Fees",
    "Hostel Fees",
    "Exam"
]


missing_columns = [
    column
    for column in required_columns
    if column not in data.columns
]


if missing_columns:

    print()
    print("ERROR: Missing required columns:")

    for column in missing_columns:
        print(
            f" - {column}"
        )

    raise SystemExit(1)


# ============================================================
# CONVERT NUMERIC COLUMNS
# ============================================================

numeric_columns = [
    "Cutoff Rank",
    "Rating",
    "Management Fees",
    "Hostel Fees"
]


for column in numeric_columns:

    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    )


# ============================================================
# CLEAN BRANCH
# ============================================================

data["Branch"] = (
    data["Branch"]
    .astype(str)
    .str.strip()
    .str.upper()
)


# ============================================================
# CLEAN EXAM
# ============================================================

data["Exam"] = (
    data["Exam"]
    .astype(str)
    .str.strip()
    .str.upper()
)


# ============================================================
# REMOVE INVALID CUTOFF DATA
# ============================================================

data = data.dropna(
    subset=[
        "Cutoff Rank"
    ]
).copy()


# ============================================================
# HANDLE MISSING NUMERIC VALUES
# ============================================================

data["Rating"] = data[
    "Rating"
].fillna(
    data["Rating"].median()
)


data["Management Fees"] = data[
    "Management Fees"
].fillna(0)


data["Hostel Fees"] = data[
    "Hostel Fees"
].fillna(0)


# ============================================================
# CREATE TRAINING SAMPLES
#
# The original dataset contains cutoff ranks.
#
# We create student-rank samples around each historical
# cutoff so the classifier learns:
#
# Student Rank + College Cutoff
#              ↓
#        Eligible / Not Eligible
#
# Rank <= cutoff  -> Eligible
# Rank > cutoff   -> Not Eligible
# ============================================================


print()
print("Creating student-rank training samples...")


training_rows = []


for _, college in data.iterrows():

    cutoff = float(
        college["Cutoff Rank"]
    )


    # --------------------------------------------------------
    # IMPORTANT RANK POINTS
    # --------------------------------------------------------

    rank_points = [

        cutoff * 0.25,

        cutoff * 0.50,

        cutoff * 0.75,

        cutoff * 0.90,

        cutoff * 0.95,

        cutoff,

        cutoff * 1.05,

        cutoff * 1.10,

        cutoff * 1.25,

        cutoff * 1.50

    ]


    for student_rank in rank_points:

        row = {

            "Exam":
                college["Exam"],

            "Branch":
                college["Branch"],

            "Student Rank":
                round(
                    student_rank
                ),

            "Rating":
                college["Rating"],

            "Management Fees":
                college["Management Fees"],

            "Hostel Fees":
                college["Hostel Fees"],

            "Cutoff Rank":
                cutoff,

            "Eligible":
                1
                if student_rank <= cutoff
                else 0

        }


        training_rows.append(
            row
        )


training_data = pd.DataFrame(
    training_rows
)


# ============================================================
# REMOVE INVALID TRAINING DATA
# ============================================================

training_data = training_data.dropna(
    subset=[
        "Student Rank",
        "Cutoff Rank"
    ]
).copy()


# ============================================================
# DISPLAY TRAINING INFORMATION
# ============================================================

print()
print(
    f"Training samples created: "
    f"{len(training_data)}"
)


print()
print(
    "Eligibility distribution:"
)

print(
    training_data[
        "Eligible"
    ].value_counts()
)


# ============================================================
# FEATURES
# ============================================================

features = [

    "Exam",

    "Branch",

    "Student Rank",

    "Rating",

    "Management Fees",

    "Hostel Fees",

    "Cutoff Rank"

]


target = "Eligible"


# ============================================================
# INPUT AND TARGET
# ============================================================

X = training_data[
    features
]

y = training_data[
    target
]


# ============================================================
# CATEGORICAL FEATURES
# ============================================================

categorical_features = [

    "Exam",

    "Branch"

]


# ============================================================
# NUMERIC FEATURES
# ============================================================

numeric_features = [

    "Student Rank",

    "Rating",

    "Management Fees",

    "Hostel Fees",

    "Cutoff Rank"

]


# ============================================================
# PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "categorical",

            OneHotEncoder(
                handle_unknown="ignore"
            ),

            categorical_features
        ),

        (
            "numeric",

            "passthrough",

            numeric_features
        )

    ]

)


# ============================================================
# RANDOM FOREST CLASSIFIER
# ============================================================

model = RandomForestClassifier(

    n_estimators=300,

    max_depth=12,

    min_samples_leaf=2,

    random_state=42,

    class_weight="balanced"

)


# ============================================================
# CREATE PIPELINE
# ============================================================

pipeline = Pipeline(

    steps=[

        (
            "preprocessor",

            preprocessor
        ),

        (
            "model",

            model
        )

    ]

)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print()
print(
    "Preparing training and testing data..."
)


X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)


# ============================================================
# TRAIN MODEL
# ============================================================

print()
print(
    "Training Random Forest classifier..."
)


pipeline.fit(

    X_train,

    y_train

)


# ============================================================
# PREDICTION
# ============================================================

print()
print(
    "Testing model..."
)


y_pred = pipeline.predict(
    X_test
)


# ============================================================
# PREDICTION PROBABILITY
# ============================================================

y_probability = pipeline.predict_proba(
    X_test
)[:, 1]


# ============================================================
# MODEL PERFORMANCE
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)


recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)


f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(

    y_test,

    y_pred,

    zero_division=0

)


# ============================================================
# MODEL PERFORMANCE INFORMATION
# ============================================================

performance = {

    "model":
        "Random Forest Classifier",

    "purpose":
        "College eligibility prediction",

    "records":
        len(data),

    "training_samples":
        len(X_train),

    "testing_samples":
        len(X_test),

    "features":
        features,

    "target":
        target,

    "accuracy":
        round(
            accuracy,
            4
        ),

    "accuracy_percentage":
        round(
            accuracy * 100,
            2
        ),

    "precision":
        round(
            precision,
            4
        ),

    "recall":
        round(
            recall,
            4
        ),

    "f1_score":
        round(
            f1,
            4
        ),

    "classification_report":
        report

}


# ============================================================
# CREATE MODEL FOLDER
# ============================================================

os.makedirs(

    MODEL_FOLDER,

    exist_ok=True

)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(

    pipeline,

    MODEL_FILE

)


# ============================================================
# SAVE PERFORMANCE
# ============================================================

joblib.dump(

    performance,

    PERFORMANCE_FILE

)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print()
print(
    "=============================================="
)

print(
    "MODEL TRAINING COMPLETED"
)

print(
    "=============================================="
)

print()

print(
    f"Original records       : {len(data)}"
)

print(
    f"Training samples       : {len(training_data)}"
)

print(
    f"Training records       : {len(X_train)}"
)

print(
    f"Testing records        : {len(X_test)}"
)

print()

print(
    "FEATURES"
)

print(
    "----------------------------------------------"
)

for feature in features:

    print(
        f" - {feature}"
    )

print()

print(
    "MODEL PERFORMANCE"
)

print(
    "----------------------------------------------"
)

print(
    f"Accuracy               : "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Precision              : "
    f"{precision * 100:.2f}%"
)

print(
    f"Recall                 : "
    f"{recall * 100:.2f}%"
)

print(
    f"F1 Score               : "
    f"{f1:.4f}"
)

print()

print(
    "CLASSIFICATION REPORT"
)

print(
    "----------------------------------------------"
)

print(
    report
)

print()

print(
    f"Model saved to         : "
    f"{MODEL_FILE}"
)

print(
    f"Performance saved to   : "
    f"{PERFORMANCE_FILE}"
)

print()

print(
    "=============================================="
)

print(
    "NEXT STEP:"
)

print(
    "Update app.py to use the new classifier."
)

print(
    "=============================================="
)

print()