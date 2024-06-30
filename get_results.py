import itertools
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, confusion_matrix, roc_curve

def plot_roc_auc(y_true, y_pred):
    """
    This function plots the ROC curves and provides the scores.
    """

    # Initialize dictionaries and array
    fpr = dict()
    tpr = dict()
    roc_auc = np.zeros(3)

    # Prepare for figure
    plt.figure()
    colors = ["aqua", "cornflowerblue"]

    # For both classification tasks (categories 1 and 2)
    for i in range(2):
        # Obtain ROC curve
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        # Obtain ROC AUC
        roc_auc[i] = auc(fpr[i], tpr[i])
        # Plot ROC curve
        plt.plot(
            fpr[i],
            tpr[i],
            color=colors[i],
            lw=2,
            label="ROC curve for task {d} (area = {f:.2f})".format(d=i + 1, f=roc_auc[i]),
        )
    # Get score for category 3
    roc_auc[2] = np.average(roc_auc[:2])

    # Format figure
    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curves")
    plt.legend(loc="lower right")
    plt.show()

    # Print scores
    for i in range(3):
        print("Category {d} Score: {f:.3f}".format(d=i + 1, f=roc_auc[i]))

def plot_confusion_matrix(y_true, y_pred, thresh, classes):
    """
    This function plots the (normalized) confusion matrix.
    """

    # Obtain class predictions from probabilities
    y_pred = (y_pred >= thresh) * 1
    # Obtain (unnormalized) confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Normalize confusion matrix
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure()
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j, i, format(cm[i, j], ".2f"), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black"
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()

if __name__ == "__main__":
    # Check for the correct number of command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python get_results.py <predictions_file> [threshold]")
        sys.exit(1)

    preds_path = sys.argv[1]
    thresh = float(sys.argv[2]) if len(sys.argv) == 3 else 0.5

    # Get ground truth labels for test dataset
    truth = pd.read_csv("C:/Users/naash/Documents/programming/skin-cancer-classification-main/ground_truth.csv")
    y_true = truth[["task_1", "task_2"]].to_numpy()

    # Get model predictions for test dataset
    y_pred = pd.read_csv(preds_path)
    y_pred = y_pred[["task_1", "task_2"]].to_numpy()

    # Plot ROC curves and print scores
    plot_roc_auc(y_true, y_pred)
    # Plot confusion matrix
    classes = ["benign", "malignant"]
    plot_confusion_matrix(y_true[:, 0], y_pred[:, 0], thresh, classes)

