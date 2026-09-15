import os
import zipfile
import urllib.request

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# BANK MARKETING DATASET - KNN CLASSIFICATION
# ============================================================

print("=" * 60)
print("BANK MARKETING DATASET - KNN CLASSIFICATION")
print("=" * 60)


# ============================================================
# 1. Download Dataset
# ============================================================

URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"

DATA_FOLDER = "data"

ZIP_FILE = os.path.join(
    DATA_FOLDER,
    "bank_marketing.zip"
)

EXTRACT_FOLDER = os.path.join(
    DATA_FOLDER,
    "extracted"
)

os.makedirs(
    DATA_FOLDER,
    exist_ok=True
)

print("\n[1] Downloading UCI Bank Marketing dataset...")

if not os.path.exists(ZIP_FILE):

    urllib.request.urlretrieve(
        URL,
        ZIP_FILE
    )

    print("Download completed.")

else:

    print("Dataset already downloaded.")


# ============================================================
# 2. Extract Main ZIP
# ============================================================

print("\n[2] Extracting main ZIP file...")

os.makedirs(
    EXTRACT_FOLDER,
    exist_ok=True
)

with zipfile.ZipFile(
    ZIP_FILE,
    "r"
) as zip_ref:

    zip_ref.extractall(
        EXTRACT_FOLDER
    )

print("Main ZIP extracted.")


# ============================================================
# 3. Find and Extract Nested ZIP Files
# ============================================================

print("\n[3] Searching for nested ZIP files...")

for root, directories, files in os.walk(
    EXTRACT_FOLDER
):

    for file in files:

        if file.lower().endswith(".zip"):

            nested_zip = os.path.join(
                root,
                file
            )

            print(
                "Extracting:",
                nested_zip
            )

            nested_extract_folder = os.path.join(
                root,
                os.path.splitext(file)[0]
            )

            os.makedirs(
                nested_extract_folder,
                exist_ok=True
            )

            with zipfile.ZipFile(
                nested_zip,
                "r"
            ) as zip_ref:

                zip_ref.extractall(
                    nested_extract_folder
                )

print("Nested ZIP extraction completed.")


# ============================================================
# 4. Find CSV File
# ============================================================

print("\n[4] Searching for bank-additional-full.csv...")

csv_file = None

for root, directories, files in os.walk(
    EXTRACT_FOLDER
):

    for file in files:

        if file.lower() == "bank-additional-full.csv":

            csv_file = os.path.join(
                root,
                file
            )

            break

    if csv_file is not None:
        break


if csv_file is None:

    print("\nERROR: CSV file was not found.")

    print("\nFiles currently available:")

    for root, directories, files in os.walk(
        EXTRACT_FOLDER
    ):

        for file in files:

            print(
                os.path.join(root, file)
            )

    raise FileNotFoundError(
        "bank-additional-full.csv was not found."
    )


print("\nCSV file found:")
print(csv_file)


# ============================================================
# 5. Load Dataset
# ============================================================

print("\n[5] Loading dataset...")

df = pd.read_csv(
    csv_file,
    sep=";"
)

print("Dataset loaded successfully.")


# ============================================================
# 6. Dataset Information
# ============================================================

print("\nDataset shape:")

print(
    df.shape
)

print("\nFirst 5 rows:")

print(
    df.head()
)

print("\nColumn names:")

print(
    df.columns.tolist()
)


# ============================================================
# 7. Separate Features and Target
# ============================================================

print("\n[6] Separating features and target...")

X = df.drop(
    "y",
    axis=1
)

y = df["y"].map({
    "no": 0,
    "yes": 1
})


# ============================================================
# 8. Identify Numerical and Categorical Columns
# ============================================================

numerical_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()


print("\nNumerical columns:")

print(
    numerical_columns
)

print("\nCategorical columns:")

print(
    categorical_columns
)


# ============================================================
# 9. Preprocessing
# ============================================================

print("\n[7] Creating preprocessing pipeline...")

preprocessor = ColumnTransformer(
    transformers=[

        (
            "numerical",
            StandardScaler(),
            numerical_columns
        ),

        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_columns
        )
    ]
)


# ============================================================
# 10. Create KNN Classifier
# ============================================================

knn = KNeighborsClassifier(
    n_neighbors=5
)


# ============================================================
# 11. Create Pipeline
# ============================================================

model = Pipeline(
    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "knn",
            knn
        )
    ]
)


# ============================================================
# 12. Split Dataset
# ============================================================

print("\n[8] Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print(
    "\nTraining samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


# ============================================================
# 13. Train KNN
# ============================================================

print("\n[9] Training KNN classifier...")

model.fit(
    X_train,
    y_train
)

print("Training completed.")


# ============================================================
# 14. Prediction
# ============================================================

print("\n[10] Making predictions...")

y_pred = model.predict(
    X_test
)

print("Prediction completed.")


# ============================================================
# 15. Calculate Accuracy
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n")
print("=" * 60)
print("KNN CLASSIFICATION RESULT")
print("=" * 60)

print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# 16. Classification Report
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "No Subscription",
            "Subscription"
        ]
    )
)


# ============================================================
# 17. Confusion Matrix
# ============================================================

print("\nConfusion Matrix:")

cm = confusion_matrix(
    y_test,
    y_pred
)

print(
    cm
)


# ============================================================
# 18. Final Result
# ============================================================

print("\n")
print("=" * 60)
print("PROGRAM COMPLETED SUCCESSFULLY")
print("=" * 60)