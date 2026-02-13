"""
EECS 658 - Assignment 2
CompareMLModels.py

Compares 7 ML models on the Iris dataset using 2-fold cross-validation:
  1) Linear Regression (as classifier via rounding)
  2) Polynomial Regression degree 2 (as classifier via rounding)
  3) Polynomial Regression degree 3 (as classifier via rounding)
  4) Gaussian Naive Bayes
  5) kNN
  6) LDA
  7) QDA

For each model:
  - Confusion Matrix (must sum to 150)
  - Accuracy Score

Author: AbhiroopGoel
Date: 2002-02-12
"""

import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import confusion_matrix, accuracy_score


def print_results(model_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(model_name)
    print("Accuracy Score:", round(acc, 3))
    print("Confusion Matrix:")
    print(cm)
    print("")

    total = int(cm.sum())
    if total != 150:
        print("WARNING: Confusion matrix sum is", total, "but should be 150.\n")


def clamp_regression_predictions(pred: np.ndarray) -> np.ndarray:
    pred = pred.round()
    pred = np.where(pred >= 3.0, 2.0, pred)
    pred = np.where(pred <= -1.0, 0.0, pred)
    return pred.astype(int)


def run_regression_model(model_name: str, poly_degree: int,
                         X: np.ndarray, y: np.ndarray,
                         skf: StratifiedKFold) -> None:
    """
    Regression-as-classifier:
      For EACH fold:
        - fit PolynomialFeatures on TRAIN only
        - fit LinearRegression on TRAIN only
        - predict on TEST
      Then combine all test predictions (150 total).
    """
    y_pred_all = np.empty_like(y)

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        poly = PolynomialFeatures(degree=poly_degree)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)

        reg = LinearRegression()
        reg.fit(X_train_poly, y_train)

        preds = reg.predict(X_test_poly)
        y_pred_all[test_idx] = clamp_regression_predictions(preds)

    print_results(model_name, y, y_pred_all)


def run_classifier_model(model_name: str, clf,
                         X: np.ndarray, y: np.ndarray,
                         skf: StratifiedKFold) -> None:
    """
    True classifier:
      For EACH fold:
        - fit model on TRAIN
        - predict on TEST
      Then combine all test predictions (150 total).
    """
    y_pred_all = np.empty_like(y)

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        clf.fit(X_train, y_train)
        y_pred_all[test_idx] = clf.predict(X_test)

    print_results(model_name, y, y_pred_all)


def main() -> None:
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Fully reproducible folds + balanced classes in each fold
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=1)

    # 1) Linear Regression (degree 1)
    run_regression_model(
        "Linear Regression (LinearRegression as classifier)",
        poly_degree=1,
        X=X, y=y, skf=skf
    )

    # 2) Polynomial Regression degree 2
    run_regression_model(
        "Polynomial Regression Degree 2 (LinearRegression as classifier)",
        poly_degree=2,
        X=X, y=y, skf=skf
    )

    # 3) Polynomial Regression degree 3
    run_regression_model(
        "Polynomial Regression Degree 3 (LinearRegression as classifier)",
        poly_degree=3,
        X=X, y=y, skf=skf
    )

    # 4) Naive Bayes
    run_classifier_model(
        "Naive Bayesian (GaussianNB)",
        GaussianNB(),
        X=X, y=y, skf=skf
    )

    # 5) kNN (k from square-root rule using training size per fold)
    # Each fold's train size is 75, so k ~= round(sqrt(75)) = 9
    k = int(round(np.sqrt(75)))
    if k < 1:
        k = 1

    knn_pipeline = Pipeline([
        ("scaler", MinMaxScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k))
    ])

    run_classifier_model(
        f"kNN (KNeighborsClassifier, k={k}, Euclidean distance + MinMax scaling)",
        knn_pipeline,
        X=X, y=y, skf=skf
    )

    # 6) LDA
    run_classifier_model(
        "LDA (LinearDiscriminantAnalysis)",
        LinearDiscriminantAnalysis(),
        X=X, y=y, skf=skf
    )

    # 7) QDA
    run_classifier_model(
        "QDA (QuadraticDiscriminantAnalysis)",
        QuadraticDiscriminantAnalysis(),
        X=X, y=y, skf=skf
    )


if __name__ == "__main__":
    main()
