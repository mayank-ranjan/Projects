"""Streamlit application backed by the Insurance Fraud Detection notebook."""

from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
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
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = PROJECT_DIR / "insurance_claims.csv"
MODEL_DIR = PROJECT_DIR / "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree Classifier": "decision_tree.joblib",
    "K-Nearest Neighbor Classifier": "k_nearest_neighbors.joblib",
    "Naive Bayes Classifier": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
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


def clean_dataset(dataset):
    """Apply the notebook's cleaning and encoding steps."""
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

    target = data["fraud_reported"].map(
        {"Y": 1, "N": 0, "Yes": 1, "No": 0}
    ).astype(int)
    features = pd.get_dummies(data.drop(columns="fraud_reported"), drop_first=True)
    return features, target


def prepare_training_data(dataset):
    """Reproduce the notebook's split and scaling steps."""
    features, target = clean_dataset(dataset)
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
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def prepare_uploaded_test_data(
    dataset, training_features, scaler, required_columns
):
    """Prepare labeled raw or notebook-exported test data."""
    if "fraud_reported" not in dataset.columns:
        raise ValueError(
            "The uploaded CSV must contain the fraud_reported target column."
        )
    if set(training_features.columns).issubset(dataset.columns):
        target = dataset["fraud_reported"].map(
            {"Y": 1, "N": 0, "Yes": 1, "No": 0}
        )
        target = target.fillna(pd.to_numeric(dataset["fraud_reported"], errors="coerce"))
        if target.isna().any():
            raise ValueError(
                "fraud_reported must contain Y/N, Yes/No, or 0/1 values."
            )
        return dataset[training_features.columns].copy(), target.astype(int)

    missing_columns = sorted(set(required_columns) - set(dataset.columns))
    if missing_columns:
        raise ValueError(
            "The uploaded CSV is missing required input columns: "
            f"{missing_columns}"
        )
    features, target = clean_dataset(dataset)
    features = features.reindex(columns=training_features.columns, fill_value=0)
    features[CONTINUOUS_COLUMNS] = scaler.transform(features[CONTINUOUS_COLUMNS])
    return features, target


def load_saved_models():
    """Load the five fitted model artifacts required by the assignment."""
    missing = [
        MODEL_DIR / filename
        for filename in MODEL_FILES.values()
        if not (MODEL_DIR / filename).is_file()
    ]
    if missing:
        names = ", ".join(path.name for path in missing)
        raise FileNotFoundError(
            f"Missing saved model files: {names}. "
            "Run train_models.py locally and commit model/*.joblib."
        )
    return {
        model_name: joblib.load(MODEL_DIR / filename)
        for model_name, filename in MODEL_FILES.items()
    }


def evaluate_model(model, X_test, y_test):
    """Evaluate a saved model without fitting it on test data."""
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
    return predictions, metrics


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
        "Upload labeled test data (CSV)",
        type=["csv"],
        help=(
            "Upload insurance_claims_test.csv from the notebook, or another "
            "CSV containing the original input columns and fraud_reported. "
            "Without an upload, the app uses the notebook's in-memory held-out "
            "test split. Saved models are evaluated without retraining."
        ),
    )

    try:
        raw_data = pd.read_csv(DEFAULT_DATASET)
        X_train, notebook_X_test, y_train, notebook_y_test, scaler = (
            prepare_training_data(raw_data)
        )
        if uploaded_file is not None:
            uploaded_data = pd.read_csv(uploaded_file)
            test_features, y_test = prepare_uploaded_test_data(
                uploaded_data, X_train, scaler, raw_data.columns
            )
            test_data_label = "uploaded labeled test data"
        else:
            st.info(
                "Upload a labeled CSV to evaluate the saved models on your own "
                "test data. The default view uses the notebook test split."
            )
            test_features, y_test = notebook_X_test, notebook_y_test
            test_data_label = "notebook test split"
        models = load_saved_models()
    except (OSError, KeyError, ValueError, TypeError, pd.errors.ParserError) as error:
        st.error(f"Could not prepare the notebook data: {error}")
        return

    results = []
    predictions = {}
    for model_name, model in models.items():
        prediction, metrics = evaluate_model(model, test_features, y_test)
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
    st.subheader("Test-data predictions")
    st.caption(f"Using {test_data_label}")
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
