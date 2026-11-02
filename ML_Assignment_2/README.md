# Insurance Claims Fraud Detection

## Project overview

This project is related to the the BITS WILP M.Tech (AIML) Machine Learning
Assignment 2 classification problem and deploys it as an interactive
Streamlit application. The application trains and compares five classifiers
for detecting fraudulent insurance claims, displays the required evaluation
metrics, and provides predictions for the held-out test data.

The notebook contains the code for implementation of models for classification . The Streamlit app
reuses its preprocessing steps, model settings, train/test split, and
evaluation approach.

## Dataset

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

## Models and required metrics

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

### Results from the notebook test split

The notebook evaluates the models on the same stratified 20% test split used
by the application:

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8350 | 0.7986 | 0.7000 | 0.5714 | 0.6292 | 0.5290 |
| Decision Tree Classifier | 0.6950 | 0.5706 | 0.3636 | 0.3265 | 0.3441 | 0.1465 |
| K-Nearest Neighbor Classifier | 0.7200 | 0.6329 | 0.3600 | 0.1837 | 0.2432 | 0.1011 |
| Naive Bayes Classifier | 0.5900 | 0.7058 | 0.3429 | 0.7347 | 0.4675 | 0.2392 |
| Support Vector Machine | 0.7650 | 0.7948 | 0.7500 | 0.0612 | 0.1132 | 0.1677 |

The overall winner is **Logistic Regression**, selected by descending F1
score, MCC score, and accuracy. The values above are the reported notebook
results; the deployed application recomputes and displays the same complete
metric table.

### Model observations

- **Logistic Regression:** Best overall performer, with the highest accuracy
  (83.5%), AUC (0.7986), F1 score (0.6292), and MCC (0.5290). It provides the
  best balance between detecting fraud and avoiding false alarms.
- **Decision Tree Classifier:** Weak performance overall, with 69.5% accuracy
  and an AUC of 0.5706. Its low F1 score (0.3441) and MCC (0.1465) indicate
  limited and imbalanced fraud detection.
- **K-Nearest Neighbor Classifier:** Moderate accuracy (72.0%), but poor fraud
  detection. Recall is only 18.37% and the F1 score is 0.2432, so it misses
  most fraudulent claims in this configuration.
- **Naive Bayes Classifier:** Best at finding fraudulent claims, with the
  highest recall (73.47%), but it produces many false positives. This results
  in the lowest accuracy (59.0%) and a modest F1 score (0.4675).
- **Support Vector Machine:** Good ranking ability (AUC 0.7948) and precision
  (75.0%), but extremely low recall (6.12%) and F1 score (0.1132) mean that it
  misses almost all fraudulent claims.

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

## Run locally

From inside the `ML_Assignment_2` folder, use Python 3.11 or newer:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit. The default dataset is loaded
automatically. A CSV uploader is also available for
`insurance_claims_test.csv` generated by the notebook. The uploaded file must
contain the processed feature columns and the `fraud_reported` target column.
The app loads the saved models and computes predictions and evaluation metrics
without retraining on the uploaded rows. A raw labeled CSV containing the
original input columns and `fraud_reported` is also accepted.

To generate or refresh all five saved model artifacts, run
`python train_models.py` once. The artifacts are written to the `model/`
directory.

## Streamlit Community Cloud deployment

1. Create a GitHub repository and upload:
   - `ML_Assignment_2/app.py`
   - `ML_Assignment_2/train_models.py`
   - `insurance_claims.csv`
   - `insurance_claims_test.csv` (generated by the notebook)
   - `requirements.txt`
   - `README.md`
2. Sign in at <https://share.streamlit.io/> using GitHub.
3. Select **New app**.
4. Select the repository and branch.
5. Set the main file path to `ML_Assignment_2/app.py`.
6. Click **Deploy**.
7. Open the deployed URL and verify the CSV uploader, model dropdown,
   evaluation table, and confusion matrix/classification report.

The dependency versions in `requirements.txt` are pinned to versions with
wheels for current Streamlit Cloud Python runtimes. Keep this file in the
same folder as `app.py`.

The GitHub repository URL and deployed Streamlit URL must be added to the
final assignment PDF before submission. A screenshot of execution on BITS
Virtual Lab is also required by the assignment.

## Assignment submission checklist

- GitHub repository link containing the complete source code.
- `requirements.txt` containing the Python dependencies.
- `README.md` containing the project description, dataset details, models,
  metrics, execution instructions, deployment instructions, and file layout.
- Training and test CSV files used by the experiment.
- `model/` containing saved trained models for all five classifiers.
- Live Streamlit Community Cloud app link that opens the interactive frontend.
- Screenshot showing execution on BITS Virtual Lab.

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
