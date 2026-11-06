# Insurance Claims Fraud Detection

## a. Problem statement

Insurance claim fraud increases financial losses and makes it difficult to
identify legitimate claims efficiently. This project builds a binary
classification solution to detect fraudulent insurance claims and deploys it
as an interactive Streamlit application. The application compares five
classifiers, reports the required evaluation metrics, and predicts labels for
held-out test data.

The notebook contains the implementation of the classification models. The
Streamlit app reuses its preprocessing steps, model settings, train/test
split, and evaluation approach.

## b. Dataset description

The application uses the public Kaggle insurance claims dataset:

<https://www.kaggle.com/code/niteshyadav3103/insurance-fraud-detection-using-12-models/input>

The supplied `insurance_claims.csv` contains 1,000 insurance claims and
39 usable input columns plus the `fraud_reported` target. The target is a
binary classification label (`Y` = fraud and `N` = not fraud), and the
dataset satisfies the assignment minimum of 12 features and 500 instances.

The preprocessing workflow:

1. Replaces cells containing only placeholder special characters with missing
   values.
2. Imputes missing values using each column's mode.
3. Creates `days_between_policy_and_incident` from the two date fields.
4. Removes identifiers, duplicate/unhelpful fields, and the target from the
   feature matrix.
5. One-hot encodes categorical variables.
6. Splits the data into stratified 80% training and 20% test sets.
7. Standardizes the same 14 continuous variables used in the notebook using
   training-set statistics only.

The split is exactly the notebook split: `test_size=0.2`,
`random_state=42`, and `stratify=y`. The categorical variables use
`pd.get_dummies(..., drop_first=True)`.

Run the notebook's final export cell to create `insurance_claims_test.csv`.
This file contains the processed held-out test features and the
`fraud_reported` target column, so it can be uploaded to the Streamlit app for
predictions and evaluation metrics.

## c. GitHub repository link

https://github.com/mayank-ranjan/Projects/tree/main/ML_Assignment_2

Streamlit app link: https://ml-insurance-fraud-detection.streamlit.app/

## d. Models used and evaluation metrics

The same processed dataset is used for all five required models:

- Logistic Regression
- Decision Tree Classifier
- K-Nearest Neighbor Classifier
- Gaussian Naive Bayes
- Support Vector Machine

The model settings match the notebook exactly:

- Logistic Regression: `max_iter=1000`, `random_state=42`
- Decision Tree: `random_state=42`
- KNN: `n_neighbors=5`
- Gaussian Naive Bayes: default settings
- SVM: `kernel="rbf"`, `probability=True`, `random_state=42`

For every model, the application displays:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

For the selected model, it also displays a confusion matrix and classification
report.

### Comparison table: results from the notebook test split

The notebook evaluates the models on the same stratified 20% test split used
by the application:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8350 | 0.7986 | 0.7000 | 0.5714 | 0.6292 | 0.5290 |
| Decision Tree Classifier | 0.6950 | 0.5706 | 0.3636 | 0.3265 | 0.3441 | 0.1465 |
| K-Nearest Neighbor Classifier | 0.7200 | 0.6329 | 0.3600 | 0.1837 | 0.2432 | 0.1011 |
| Naive Bayes Classifier | 0.5900 | 0.7058 | 0.3429 | 0.7347 | 0.4675 | 0.2392 |
| Support Vector Machine | 0.7650 | 0.7948 | 0.7500 | 0.0612 | 0.1132 | 0.1677 |

The values above are the reported notebook results; the deployed application
recomputes and displays the same complete metric table.

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall performer, with the highest accuracy (83.5%), AUC (0.7986), F1 score (0.6292), and MCC (0.5290). It provides the best balance between detecting fraud and avoiding false alarms. |
| Decision Tree Classifier | Weak performance overall, with 69.5% accuracy and an AUC of 0.5706. Its low F1 score (0.3441) and MCC (0.1465) indicate limited and imbalanced fraud detection. |
| K-Nearest Neighbor Classifier | Moderate accuracy (72.0%), but poor fraud detection. Recall is only 18.37% and the F1 score is 0.2432, so it misses most fraudulent claims in this configuration. |
| Naive Bayes | Best at finding fraudulent claims, with the highest recall (73.47%), but it produces many false positives. This results in the lowest accuracy (59.0%) and a modest F1 score (0.4675). |
| Support Vector Machine | Good ranking ability (AUC 0.7948) and precision (75.0%), but extremely low recall (6.12%) and F1 score (0.1132) mean that it misses almost all fraudulent claims. |
| **Overall Winner** | **Logistic Regression**, selected by descending F1 score, MCC score, and accuracy. |

## Saved trained models

The `model/` directory is included to satisfy the requirement for saved
trained models for all five classifiers. It contains one `.joblib` artifact
per classifier:

```text
model/
├── logistic_regression.joblib
├── decision_tree.joblib
├── k_nearest_neighbors.joblib
├── naive_bayes.joblib
└── support_vector_machine.joblib
```

The `train_models.py` script fits each model using the training data and saves
the fitted estimator to the corresponding file. The Streamlit app loads these
saved models and evaluates them without retraining on uploaded test data. The
binary files cannot contain comments; the model-to-file mapping and artifact
instructions are documented in `model/README.md`.

If the artifacts are not already present after cloning the repository, run
`python train_models.py` locally and commit the generated files before
submission. The app uses `insurance_claims.csv` to reproduce the preprocessing
needed for the default notebook split and uploaded raw test data; it does not
fit the saved models during app execution.

## Repository contents

```text
ML_Assignment_2/
├── app.py
├── train_models.py
├── requirements.txt
├── insurance_claims.csv
├── insurance_claims_test.csv
├── Insurance Fraud Detection.ipynb
├── model/
│   ├── README.md
│   └── *.joblib
└── README.md
```

## Project files

- `app.py`: Streamlit user interface using the notebook-aligned workflow and
  saved classifiers and uploaded labeled test data.
- `train_models.py`: Reproducible script for generating the five saved models.
- `insurance_claims.csv`: Dataset used for the experiment.
- `Insurance Fraud Detection.ipynb`: Notebook containing exploratory
  analysis, model results, winner selection, and test-data export.
- `insurance_claims_test.csv`: Processed held-out test features and the
  `fraud_reported` target generated by the notebook for upload and evaluation.
- `requirements.txt`: Python dependencies for local and cloud deployment.
- `model/`: Saved `.joblib` artifacts for all five trained classifiers.
