"""Streamlit application backed by the Insurance Fraud Detection notebook."""

from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = PROJECT_DIR / "insurance_claims.csv"
DEFAULT_TEST_DATASET = PROJECT_DIR / "insurance_claims_test.csv"
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


def save_uploaded_file(uploaded_file):
    """Persist an uploaded CSV briefly so pandas can read it."""
    temporary_file = NamedTemporaryFile(delete=False, suffix=".csv")
    temporary_file.write(uploaded_file.getvalue())
    temporary_file.close()
    return Path(temporary_file.name)


def prepare_training_data(dataset):
    """Reproduce the notebook's cleaning, encoding, split, and scaling steps."""
    data = dataset.copy()
    data.replace(r"^[^A-Za-z0-9\s]+$", pd.NA, regex=True, inplace=True)
    data.fillna(data.mode().iloc[0], inplace=True)
    data["policy_bind_date"] = pd.to_datetime(
        data["policy_bind_date"], errors="coerce"
    )
    data["incident_date"] = pd.to_datetime(data["incident_date"], errors="coerce")
    data["days_between_policy_and_incident"] = (
        data["incident_date"] - data["policy_bind_date"]
    ).dt.days
    data.drop(columns=COLUMNS_TO_DROP, inplace=True)
    data.drop(columns=["age", "total_claim_amount"], inplace=True)

    features = data.drop(columns="fraud_reported")
    target = data["fraud_reported"].map(
        {"Y": 1, "N": 0, "Yes": 1, "No": 0}
    ).astype(int)
    features = pd.get_dummies(features, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[CONTINUOUS_COLUMNS] = scaler.fit_transform(
        X_train[CONTINUOUS_COLUMNS]
    )
    X_test_scaled[CONTINUOUS_COLUMNS] = scaler.transform(
        X_test[CONTINUOUS_COLUMNS]
    )
    return X_train_scaled, X_test_scaled, y_train, y_test


def create_models():
    """Create the five models with the settings used in the notebook."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
        "K-Nearest Neighbor Classifier": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes Classifier": GaussianNB(),
        "Support Vector Machine": SVC(
            kernel="rbf", probability=True, random_state=42
        ),
    }


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Fit a model and return its predictions, probabilities, and metrics."""
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "AUC Score": roc_auc_score(y_test, probabilities),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1 Score": f1_score(y_test, predictions, zero_division=0),
        "MCC Score": matthews_corrcoef(y_test, predictions),
    }
    return model, predictions, metrics


def main():
    st.set_page_config(
        page_title="Insurance Fraud Detection",
        page_icon="🔎",
        layout="wide",
    )
    st.title("Insurance Claims Fraud Detection")
    st.write(
        "This app uses the preprocessing and five models from "
        "`Insurance Fraud Detection.ipynb`."
    )

    uploaded_file = st.file_uploader(
        "Upload the notebook-generated test CSV (optional)",
        type=["csv"],
        help=(
            "Run the final notebook export cell first. The uploaded file must "
            "contain the processed test feature columns and no target column."
        ),
    )

    temporary_path = None
    test_path = DEFAULT_TEST_DATASET
    if uploaded_file is not None:
        temporary_path = save_uploaded_file(uploaded_file)
        test_path = temporary_path

    try:
        raw_data = pd.read_csv(DEFAULT_DATASET)
        X_train, notebook_X_test, y_train, y_test = prepare_training_data(raw_data)
        if test_path.exists():
            test_features = pd.read_csv(test_path)
            missing_columns = set(X_train.columns) - set(test_features.columns)
            extra_columns = set(test_features.columns) - set(X_train.columns)
            if missing_columns or extra_columns:
                raise ValueError(
                    "Test CSV columns do not match the notebook export. "
                    f"Missing: {sorted(missing_columns)}; "
                    f"unexpected: {sorted(extra_columns)}"
                )
            if len(test_features) != len(y_test):
                raise ValueError(
                    "Test CSV must contain exactly the 20% held-out test split "
                    f"({len(y_test)} rows); received {len(test_features)} rows."
                )
            test_features = test_features[X_train.columns]
        else:
            st.info(
                "insurance_claims_test.csv is not present yet; using the in-memory "
                "notebook test split. Run the final notebook cell to create it."
            )
            test_features = notebook_X_test
    except (OSError, KeyError, ValueError, TypeError, pd.errors.ParserError) as error:
        st.error(f"Could not prepare the notebook data: {error}")
        return
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    models = create_models()
    results = []
    predictions = {}
    for model_name, model in models.items():
        _, prediction, metrics = evaluate_model(
            model, X_train, test_features, y_train, y_test
        )
        results.append({"Model": model_name, **metrics})
        predictions[model_name] = prediction

    results_df = pd.DataFrame(results)
    metric_columns = [
        "Accuracy",
        "AUC Score",
        "Precision",
        "Recall",
        "F1 Score",
        "MCC Score",
    ]

    st.sidebar.header("Model selection")
    selected_model_name = st.sidebar.selectbox(
        "Choose a model to inspect", options=list(models)
    )
    st.subheader("Notebook test-data predictions")
    st.caption(f"{len(test_features):,} processed test rows")
    st.dataframe(
        results_df.style.format({column: "{:.4f}" for column in metric_columns}),
        use_container_width=True,
        hide_index=True,
    )

    winner = results_df.sort_values(
        by=["F1 Score", "MCC Score", "Accuracy"], ascending=False
    ).iloc[0]
    st.success(
        f"Recommended overall model: {winner['Model']} "
        "(ranked by F1 score, MCC score, and accuracy)."
    )

    selected_prediction = predictions[selected_model_name]
    selected_metrics = results_df.loc[
        results_df["Model"] == selected_model_name, metric_columns
    ].iloc[0]
    st.subheader(f"Selected model: {selected_model_name}")
    metric_layout = st.columns(len(metric_columns))
    for column, metric_name in zip(metric_layout, metric_columns):
        column.metric(metric_name, f"{selected_metrics[metric_name]:.4f}")

    result_columns = st.columns(2)
    with result_columns[0]:
        st.markdown("#### Confusion matrix")
        matrix = confusion_matrix(y_test, selected_prediction, labels=[0, 1])
        st.dataframe(
            pd.DataFrame(
                matrix,
                index=["Actual: Not Fraud", "Actual: Fraud"],
                columns=["Predicted: Not Fraud", "Predicted: Fraud"],
            ),
            use_container_width=True,
        )
    with result_columns[1]:
        st.markdown("#### Classification report")
        report = classification_report(
            y_test,
            selected_prediction,
            labels=[0, 1],
            target_names=["Not Fraud", "Fraud"],
            output_dict=True,
            zero_division=0,
        )
        st.dataframe(pd.DataFrame(report).transpose().style.format("{:.4f}"))

    prediction_output = test_features.copy()
    prediction_output["predicted_fraud"] = predictions[selected_model_name]
    st.download_button(
        "Download predictions for the selected model",
        data=prediction_output.to_csv(index=False),
        file_name="insurance_claims_predictions.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
