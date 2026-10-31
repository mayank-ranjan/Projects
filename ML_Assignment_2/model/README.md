# Saved models

This directory is the deployment artifact required by the assignment. The
`train_models.py` fits each classifier using the notebook-aligned training data
and saves the fitted estimator here as a `.joblib` file:

- `logistic_regression.joblib`
- `decision_tree.joblib`
- `k_nearest_neighbors.joblib`
- `naive_bayes.joblib`
- `support_vector_machine.joblib`

The files are binary serialized models, so comments cannot be stored inside
them. The comments and model-to-file mapping are documented here and in
`app.py`.

Run `python train_models.py` from the `ML_Assignment_2` directory to generate
the files. The Streamlit app loads these artifacts and never fits a model
using uploaded test data. They should not be manually edited. If the training
data or preprocessing changes, rerun the script to regenerate them.
