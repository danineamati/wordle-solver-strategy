"""Load second-guess strategy CSVs and build feature matrices for decision trees."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = (
    "first_word",
    "first_code",
    "assigned_second_guess",
    "bin_probability",
)


def load_strategy_csv(path: Path | str) -> pd.DataFrame:
    """Read a strategy CSV and validate required columns and types."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Strategy CSV not found: {p}")

    df = pd.read_csv(
        p,
        dtype={"first_code": str},
        keep_default_na=False,
    )
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df = df.copy()
    df["first_word"] = df["first_word"].astype(str).str.lower().str.strip()
    # Preserve leading zeros (CSV may store codes as integers without dtype=str).
    df["first_code"] = (
        pd.to_numeric(df["first_code"], errors="coerce")
        .fillna(0)
        .astype(np.int64)
        .astype(str)
        .str.zfill(5)
    )
    df["assigned_second_guess"] = (
        df["assigned_second_guess"].astype(str).str.lower().str.strip()
    )
    df["bin_probability"] = pd.to_numeric(df["bin_probability"], errors="coerce")
    if df["bin_probability"].isna().any():
        raise ValueError("bin_probability contains non-numeric or missing values")

    words = df["first_word"].unique()
    if len(words) != 1:
        raise ValueError(
            f"Expected a single first_word in CSV, got: {list(words)}"
        )

    first_word = words[0]
    if len(first_word) != 5:
        raise ValueError(f"first_word must have length 5, got {len(first_word)!r}")

    for idx, code in enumerate(df["first_code"]):
        if len(code) != 5:
            raise ValueError(
                f"Row {idx}: first_code must have length 5, got {code!r}"
            )
        for ch in code:
            if ch not in "012":
                raise ValueError(
                    f"Row {idx}: first_code must use only 0,1,2, got {code!r}"
                )

    return df


def parse_code_digits(first_code: str) -> np.ndarray:
    """Return shape (5,) int array of tile states 0=absent, 1=yellow, 2=green."""
    arr = np.empty(5, dtype=np.int64)
    for i, ch in enumerate(first_code):
        arr[i] = int(ch)
    return arr


def build_feature_matrix(
    df: pd.DataFrame,
    first_word_col: str = "first_word",
    code_col: str = "first_code",
    label_col: str = "assigned_second_guess",
    weight_col: str = "bin_probability",
) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray]:
    """
    Build X, y strings, and sample weights from a validated strategy dataframe.

    Features: five ordinal tile states per position (0/1/2) named by first-word
    letter, plus n_green, n_yellow, n_absent counts.
    """
    if df.empty:
        raise ValueError("DataFrame is empty")

    first_word = str(df[first_word_col].iloc[0]).lower().strip()
    letters = [c for c in first_word]

    feature_names: List[str] = (
        [f"{letter}_tile" for letter in letters]
        + ["n_green", "n_yellow", "n_absent"]
    )

    x_list = []
    y = df[label_col].to_numpy()

    weights = df[weight_col].to_numpy(dtype=np.float64)
    weight_sum = weights.sum()
    if weight_sum <= 0:
        raise ValueError("bin_probability weights must sum to a positive value")
    sample_weight = weights / weight_sum

    for code in df[code_col].astype(str).str.strip():
        digits = parse_code_digits(code)
        n_green = int(np.sum(digits == 2))
        n_yellow = int(np.sum(digits == 1))
        n_absent = int(np.sum(digits == 0))
        row = np.concatenate([digits.astype(np.float64), [n_green, n_yellow, n_absent]])
        x_list.append(row)

    x = np.vstack(x_list)
    return x, feature_names, sample_weight, y
