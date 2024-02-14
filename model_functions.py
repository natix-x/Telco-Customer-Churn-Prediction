from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_auc_score,
)
import numpy as np


def fitter(estimator, param_grid, X_train, y_train):
    """
    Finding the best estimator based on chosen parameters via GridSearchCV

    :param estimator: basic estimator chosen for GridSearchCV
    :param param_grid: param_grid with params associated with chosen estimator
    :param X_train: features to train on
    :param y_train: labels to train on
    :return:
    """
    grid_model = GridSearchCV(estimator, param_grid, scoring="roc_auc")
    grid_model.fit(X_train, y_train)
    grid_model = grid_model.best_estimator_
    return grid_model


def evaluate(estimator, X_test, y_test):
    """
    evaluation the performance of chosen estimator
    :param estimator: estimator chosen for evaluation
    :param X_test:  features to evaluate model on
    :param y_test: labels to evaluate model on
    :return: ConfusionMatrixDisplay, classification report, roc_auc_score
    """
    print(f"{estimator}")
    predictions = estimator.predict(X_test)
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    print(classification_report(y_test, predictions))

    def change(value):
        if value == "Yes":
            return True
        else:
            return False

    v_change = np.vectorize(change)
    print(
        f"roc_auc_score = {round(roc_auc_score(v_change(y_test), v_change(predictions)), 2)}"
    )
