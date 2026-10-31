"""Train and save the five classifiers required by the assignment.

Run this script from the ML_Assignment_2 directory before deploying when the
model/*.joblib artifacts are not already present.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "model"
DATASET = PROJECT_DIR / "insurance_claims.csv"
MODEL_FILES = {
    "logistic_regression.joblib": LogisticRegression(max_iter=1000, random_state=42),
    "decision_tree.joblib": DecisionTreeClassifier(random_state=42),
    "k_nearest_neighbors.joblib": KNeighborsClassifier(n_neighbors=5),
    "naive_bayes.joblib": GaussianNB(),
    "support_vector_machine.joblib": SVC(
        kernel="rbf", probability=True, random_state=42
    ),
}
CONTINUOUS_COLUMNS = [
    "months_as_customer",
    "policy_deductable",
    "policy_annual_premium",
    "umbrella_limit",
    "capital-gains",
    "capital-loss",
    "incident_hour_of_the_day",
    "number_of_vehicles_involved",
    "bodily_injuries",
    "witnesses",
    "injury_claim",
    "property_claim",
    "vehicle_claim",
    "days_between_policy_and_incident",
]
COLUMNS_TO_DROP = [
    "policy_number",
    "policy_bind_date",
    "policy_state",
    "insured_zip",
    "incident_location",
    "incident_date",
    "incident_state",
    "incident_city",
    "insured_hobbies",
    "auto_make",
    "auto_model",
    "auto_year",
    "_c39",
]


data = pd.read_csv(DATASET)
data.replace(r"^[^A-Za-z0-9\s]+$", pd.NA, regex=True, inplace=True)
data.fillna(data.mode().iloc[0], inplace=True)
data["policy_bind_date"] = pd.to_datetime(data["policy_bind_date"], errors="coerce")
data["incident_date"] = pd.to_datetime(data["incident_date"], errors="coerce")
data["days_between_policy_and_incident"] = (
    data["incident_date"] - data["policy_bind_date"]
).dt.days
data.drop(columns=COLUMNS_TO_DROP + ["age", "total_claim_amount"], inplace=True)
target = data.pop("fraud_reported").map({"Y": 1, "N": 0, "Yes": 1, "No": 0}).astype(int)
features = pd.get_dummies(data, drop_first=True)
X_train, _, y_train, _ = train_test_split(
    features, target, test_size=0.2, random_state=42, stratify=target
)
scaler = StandardScaler()
X_train = X_train.copy()
X_train[CONTINUOUS_COLUMNS] = scaler.fit_transform(X_train[CONTINUOUS_COLUMNS])

MODEL_DIR.mkdir(exist_ok=True)
for filename, model in MODEL_FILES.items():
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_DIR / filename)
    print(f"Saved {MODEL_DIR / filename}")
