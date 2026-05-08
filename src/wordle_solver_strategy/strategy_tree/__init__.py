"""Decision tree helpers for approximating second-guess strategy CSVs."""

from .features import build_feature_matrix, load_strategy_csv
from .humanize import humanize_decision_tree_export
from .tree import (
    fit_second_guess_tree,
    tree_rules_as_text,
    weighted_accuracy,
)

__all__ = [
    "build_feature_matrix",
    "load_strategy_csv",
    "fit_second_guess_tree",
    "humanize_decision_tree_export",
    "tree_rules_as_text",
    "weighted_accuracy",
]
