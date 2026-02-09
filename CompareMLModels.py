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

Author: YOUR NAME
Date: YYYY-MM-DD
"""

import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# -----------------------------
# Helper functions
# -----------------------------

def print_results(model_name: str, actual: np.ndarray, predicted: np.ndarray) -> None:
    """Print accuracy and confusion matrix, and verify CM sums to 150."""
    acc = accuracy_score(actual, predicted)
    cm = confusion_matrix(actual, predicted)

    print(model_name)
    print("Accuracy Score:", round(acc, 3))
    print("Confusion Matrix:")
    print(cm)
    print("")

    total = int(cm.sum())
    if total != 150:
        print("WARNING: Confusion matrix sum is", total, "but should be 150.\n")


def clamp_regression_predictions(pred: np.ndarray) -> np.ndarray:
    """
    For regression-as-classifier:
      - round to nearest integer
      - clamp to valid class labels {0,1,2}
    """
    pred = pred.round()
    pred = np.where(pred >= 3.0, 2.0, pred)
    pred = np.where(pred <= -1.0, 0.0, pred)
    return pred.astype(int)


def run_regression_classifier(model_name: str, poly_degree: int,
                              X1: np.ndarray, y1: np.ndarray,
                              X2: np.ndarray, y2: np.ndarray) -> None:
    """
    Regression-based classifier using 2-fold CV:
      Train fold1 -> predict fold2
      Train fold2 -> predict fold1
    Then concatenate to get 150 total predictions.
    """
    poly = PolynomialFeatures(degree=poly_degree)

    # Fit poly mapping on fold1, then apply to both folds (consistent feature expansion)
    X1_poly = poly.fit_transform(X1)
    X2_poly = poly.transform(X2)

    reg = LinearRegression()

    # Train on fold1, test on fold2
    reg.fit(X1_poly, y1)
    pred_fold2 = clamp_regression_predictions(reg.predict(X2_poly))

    # Train on fold2, test on fold1
    reg.fit(X2_poly, y2)
    pred_fold1 = clamp_regression_predictions(reg.predict(X1_poly))

    # Concatenate in the same order as predictions
    actual = np.concatenate([y2, y1])
    predicted = np.concatenate([pred_fold2, pred_fold1])

    print_results(model_name, actual, predicted)


def run_classifier(model_name: str, clf,
                   X1: np.ndarray, y1: np.ndarray,
                   X2: np.ndarray, y2: np.ndarray) -> None:
    """
    True classifier using 2-fold CV:
      Train fold1 -> predict fold2
      Train fold2 -> predict fold1
    Then concatenate to get 150 total predictions.
    """
    clf.fit(X1, y1)
    pred_fold2 = clf.predict(X2)

    clf.fit(X2, y2)
    pred_fold1 = clf.predict(X1)

    actual = np.concatenate([y2, y1])
    predicted = np.concatenate([pred_fold2, pred_fold1])

    print_results(model_name, actual, predicted)


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    # Load Iris dataset (150 samples, 4 features, 3 classes labeled 0/1/2)
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Two-fold split (50/50) with fixed random seed (reproducible)
    # fold1 ~ 75 samples, fold2 ~ 75 samples
    X_fold1, X_fold2, y_fold1, y_fold2 = train_test_split(
        X, y, test_size=0.50, random_state=1
    )

    # 1) Linear Regression (degree 1 polynomial features)
    run_regression_classifier(
        "Linear Regression (LinearRegression as classifier)",
        poly_degree=1,
        X1=X_fold1, y1=y_fold1,
        X2=X_fold2, y2=y_fold2
    )

    # 2) Polynomial Regression degree 2
    run_regression_classifier(
        "Polynomial Regression Degree 2 (LinearRegression as classifier)",
        poly_degree=2,
        X1=X_fold1, y1=y_fold1,
        X2=X_fold2, y2=y_fold2
    )

    # 3) Polynomial Regression degree 3
    run_regression_classifier(
        "Polynomial Regression Degree 3 (LinearRegression as classifier)",
        poly_degree=3,
        X1=X_fold1, y1=y_fold1,
        X2=X_fold2, y2=y_fold2
    )

    # 4) Naïve Bayes (GaussianNB)
    # Slides: Naïve Bayes assumes Gaussian distributions and feature independence;
    # GaussianNB implements the Gaussian probability density approach.
    run_classifier(
        "Naïve Bayesian (GaussianNB)",
        GaussianNB(),
        X1=X_fold1, y1=y_fold1,
        X2=X_fold2, y2=y_fold2
    )

    # 5) kNN
    # Slides: Euclidean distance is standard; scaling to [0,1] improves distance-based models.
    # Also: square-root rule for k ≈ sqrt(n_train). Here n_train ~ 75 per fold => sqrt(75) ~ 8.66 => use k=9.
    k = int(round(np.sqrt(len(y_fold1))))
    if k < 1:
        k = 1

    knn_pipeline = Pipeline([
        ("scaler", MinMaxScaler()),            # normalize features to [0,1]
        ("knn", KNeighborsClassifier(n_neighbors=k))
    ])

    run_classifier(
        f"kNN (KNeighborsClassifier, k={k}, Euclidean distance + MinMax scaling)",
        knn_pipeline,
        X1=X_fold1, y1=y_fold1,
        X2=X_fold2, y2=y_fold2
    )

    # 6) LDA
    # Slides: LDA assumes Gaussian class conditionals with shared covariance matrix across classes (linear boundary).
    run_classifier(
        "LDA (LinearDiscriminantAnalysis)",
        LinearDiscriminantAnalysis(),
        X1=X_fold1, y1=y_fold1,
        X2=X_fold2, y2=y_fold2
    )

    # 7) QDA
    # Slides: QDA assumes Gaussian class conditionals with different covariance matrices per class (quadratic boundary).
    run_classifier(
        "QDA (QuadraticDiscriminantAnalysis)",
        QuadraticDiscriminantAnalysis(),
        X1=X_fold1, y1=y_fold1,
        X2=X_fold2, y2=y_fold2
    )


if __name__ == "__main__":
    main()
