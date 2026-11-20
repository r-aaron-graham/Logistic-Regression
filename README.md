# Logistic Regression

A local classification example using synthetic data, feature scaling, and logistic regression. The scaler fits on training data within a scikit-learn pipeline. A stratified held-out split measures prediction accuracy.

## Run

Use Python 3.12 in a virtual environment.

```sh
python -m pip install -r requirements.txt
python demo.py
```

The fixed seed produces 375 training rows and 125 test rows. The verified run returned accuracy 0.84 and confusion matrix [[56, 7], [13, 49]]. Library versions may affect exact results. This synthetic benchmark demonstrates the workflow. Real deployment requires representative data and evaluation against the intended decision criteria.

## Inspect the implementation

The demo shows dataset generation, stratified splitting, training-only scaling, model fitting, and held-out evaluation. All data is generated locally.
