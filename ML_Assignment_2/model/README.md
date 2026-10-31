# Saved models

This directory is the deployment artifact required by the assignment. The
Streamlit app fits each classifier using the notebook-aligned training data
and saves the fitted estimator here as a `.joblib` file:

- `logistic_regression.joblib`
- `decision_tree.joblib`
- `k_nearest_neighbors.joblib`
- `naive_bayes.joblib`
- `support_vector_machine.joblib`

The files are binary serialized models, so comments cannot be stored inside
them. The comments and model-to-file mapping are documented here and in
`app.py`.

The files are generated automatically the first time the app runs. They
should not be manually edited. If the training data or preprocessing changes,
run the app again to regenerate them.
