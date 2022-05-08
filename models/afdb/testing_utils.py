import numpy as np


def testing_metrics(y_test: np.ndarray, y_pred: np.ndarray):
    y_test = np.ravel(y_test)
    y_pred = np.ravel(y_pred)

    true_positives = np.sum(y_test * y_pred)
    false_positives = np.sum(np.abs(y_test - 1) * y_pred)
    true_negatives = np.sum((y_test - 1) * (y_pred - 1))
    false_negatives = np.sum(y_test * np.abs(y_pred - 1))

    accuracy = round(
        (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives), 4)
    precision = round(true_positives / (true_positives + false_positives), 4)
    recall = round(true_positives / (true_positives + false_negatives), 4)
    specificity = round(true_negatives / (true_negatives + false_positives), 4)
    npv = round(true_negatives / (true_negatives + false_negatives), 4)
    f1_1 = round(2 * (precision * recall) / (precision + recall), 4)
    f1_0 = round(2 * (specificity * npv) / (specificity + npv), 4)
    f1_macro = round((f1_1 + f1_0) / 2, 4)

    return dict(
        accuracy = accuracy, f1_macro = f1_macro,
        f1_1 = f1_1, f1_0 = f1_0,
        Precision = precision, Recall = recall,
        Specificity = specificity, npv = npv,
        TP = int(true_positives), FP = int(false_positives), FN = int(false_negatives),
        TN = int(true_negatives),
    )
