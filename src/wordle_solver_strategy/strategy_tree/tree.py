"""Train a shallow decision tree on second-guess labels and export rules as text."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text

FittedTree = Tuple[DecisionTreeClassifier, LabelEncoder]


def weighted_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> float:
    """Fraction of mass where prediction equals the oracle label (0..1)."""
    correct = y_true == y_pred
    if sample_weight is None:
        return float(np.mean(correct))
    return float(np.sum(sample_weight * correct))


def fit_second_guess_tree(
    x: np.ndarray,
    y_labels: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    *,
    max_depth: int = 4,
    random_state: int = 0,
    min_samples_leaf: int = 1,
) -> FittedTree:
    """Fit a decision tree on string labels; return model and label encoder."""
    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y_labels)

    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=random_state,
        min_samples_leaf=min_samples_leaf,
    )
    clf.fit(x, y_enc, sample_weight=sample_weight)
    return clf, encoder


def tree_rules_as_text(
    model: DecisionTreeClassifier,
    feature_names: list,
    *,
    class_names: Optional[list] = None,
) -> str:
    """Return sklearn's text representation of the fitted tree."""
    return export_text(
        model,
        feature_names=list(feature_names),
        class_names=class_names,
        decimals=2,
    )
