"""Train/validation/test splitting with actor-level isolation and simple splits.

This module provides two main splitting strategies:
  1. make_actorwise_splits: Isolates actors to prevent speaker leakage (recommended for RAVDESS)
  2. make_simple_stratified_split: Simple 80/20 train/test split with stratification by emotion
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, train_test_split


def _safe_group_shuffle_split(
    frame: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_groups = int(groups.nunique())
    if n_groups < 2:
        raise ValueError("Need at least two actors for a group-based split.")

    min_test_ratio = 1.0 / n_groups
    max_test_ratio = (n_groups - 1.0) / n_groups
    adjusted_test_size = min(max(test_size, min_test_ratio), max_test_ratio)

    splitter = GroupShuffleSplit(n_splits=1, test_size=adjusted_test_size, random_state=random_state)
    return next(splitter.split(frame, y, groups))


def _recommended_n_splits(y: pd.Series, groups: pd.Series, upper_bound: int) -> int:
    min_class_count = int(y.value_counts().min())
    unique_groups = int(groups.nunique())
    n_splits = min(upper_bound, min_class_count, unique_groups)
    if n_splits < 2:
        raise ValueError("Not enough samples/groups for stratified group split.")
    return n_splits


def _stratified_group_split(
    y: pd.Series,
    groups: pd.Series,
    n_splits: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    split_count = _recommended_n_splits(y, groups, n_splits)
    splitter = StratifiedGroupKFold(n_splits=split_count, shuffle=True, random_state=random_state)
    dummy_x = np.zeros((len(y), 1), dtype=np.float32)
    for train_idx, holdout_idx in splitter.split(dummy_x, y, groups):
        return train_idx, holdout_idx
    raise RuntimeError("Could not produce a stratified group split.")


def make_actorwise_splits(
    feature_table: pd.DataFrame,
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    if feature_table["actor"].nunique() < 3:
        raise ValueError(
            "Need at least 3 actors to create train/val/test actor-wise splits. "
            "Increase --max-files or use the full dataset."
        )

    y_all = feature_table["emotion"]
    g_all = feature_table["actor"]

    try:
        train_val_idx, test_idx = _stratified_group_split(y_all, g_all, n_splits=5, random_state=random_state)
    except Exception:
        # Fallback keeps actor isolation if stratification is infeasible.
        train_val_idx, test_idx = _safe_group_shuffle_split(
            feature_table,
            y_all,
            g_all,
            test_size=0.2,
            random_state=random_state,
        )

    train_val = feature_table.iloc[train_val_idx].reset_index(drop=True)
    test = feature_table.iloc[test_idx].reset_index(drop=True)

    y_tv = train_val["emotion"]
    g_tv = train_val["actor"]

    try:
        train_idx, val_idx = _stratified_group_split(y_tv, g_tv, n_splits=4, random_state=random_state)
    except Exception:
        train_idx, val_idx = _safe_group_shuffle_split(
            train_val,
            y_tv,
            g_tv,
            test_size=0.25,
            random_state=random_state,
        )

    train = train_val.iloc[train_idx].reset_index(drop=True)
    val = train_val.iloc[val_idx].reset_index(drop=True)

    return {
        "train": train,
        "val": val,
        "test": test,
    }


def make_simple_stratified_split(
    feature_table: pd.DataFrame,
    train_ratio: float = 0.8,
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    """Create a simple stratified train/test split without actor isolation.
    
    This is useful for quick experimentation on the full dataset.
    Warning: May have speaker leakage (same actor in train and test).
    
    Args:
        feature_table: DataFrame with emotion labels and features.
        train_ratio: Fraction of data for training (default 0.8 = 80% train, 20% test).
        random_state: Random seed for reproducibility.
    
    Returns:
        Dict with keys 'train' and 'test' (no validation split).
    """
    train, test = train_test_split(
        feature_table,
        train_size=train_ratio,
        stratify=feature_table["emotion"],
        random_state=random_state,
    )
    
    return {
        "train": train.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }
