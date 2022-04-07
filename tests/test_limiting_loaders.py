from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV

from authbench.loaders import (
    CrossValidatedSplitLoader,
    ImdbLoader,
    RedditLoader,
    TrainTestSplitLoader,
)


class ShapeCheckingDummyClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, expected_n_docs: int):
        self.expected_n_docs = expected_n_docs

    def fit(self, x: pd.DataFrame, y: np.ndarray) -> ShapeCheckingDummyClassifier:
        print("training on", x.author.nunique(), "authors")
        assert x.groupby("author").count().text_raw.min() == self.expected_n_docs
        assert x.groupby("author").count().text_raw.max() == self.expected_n_docs
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.array([0] * x.shape[0])


class FixedSplitLoader(TrainTestSplitLoader):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        data = np.random.random((1000, 10))
        self.df = pd.DataFrame(data)
        self.df["data_type"] = ["train"] * 500 + ["test"] * 500
        self.df["y"] = [str(x % 10) for x in range(data.shape[0])]

    def get_train_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        train_df = self.df[self.df.data_type == "train"]
        return train_df, train_df.y.values

    def get_test_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        test_df = self.df[self.df.data_type == "test"]
        return test_df, test_df.y.values


def test_train_test_split() -> None:
    dataloader = FixedSplitLoader(max_targets=4, max_docs_per_target=7)
    df, y, splits = dataloader.load()

    # same amount of rows and targets
    assert df.shape[0] == y.shape[0]

    # 4 different targets
    assert len(set(y)) == 4

    df["y"] = y
    # the reduced documents per target can only be measured in the train part of the df
    train_idx, test_idx = splits[0]
    train_df = df.iloc[train_idx]
    assert train_df.groupby("y").count()[0].min() == 7
    assert train_df.groupby("y").count()[0].max() == 7


class CrossValidationSplitTestLoader(CrossValidatedSplitLoader):
    def get_all_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        data = np.random.random((1000, 10))
        df = pd.DataFrame(data)
        df["y"] = [str(x % 10) for x in range(data.shape[0])]
        return df, df.y.values


def test_cross_validation_split() -> None:
    loader = CrossValidationSplitTestLoader(max_targets=4, max_docs_per_target=7)
    df, y, splits = loader.load()

    # same amount of rows and targets
    assert df.shape[0] == y.shape[0]

    # 4 different targets
    assert len(set(y)) == 4

    for train_idx, test_idx in splits:
        train_df = df.iloc[train_idx]
        assert train_df.groupby("y").count()[0].min() == 7
        assert train_df.groupby("y").count()[0].max() == 7


def test_limited_fixed_split_evaluator() -> None:
    dataloader = RedditLoader(
        path=f"data/R-CL5-FR/processed",
        train_languages=["en"],
        test_languages=["fr"],
        max_docs_per_target=8,
    )
    df, y, cv = dataloader.load()
    model = ShapeCheckingDummyClassifier(8)
    evaluator = GridSearchCV(model, {}, refit=False, cv=cv)
    evaluator.fit(df, y)


def test_limited_cv_evaluator() -> None:
    dataloader = ImdbLoader(max_docs_per_target=13)
    df, y, cv = dataloader.load()
    model = ShapeCheckingDummyClassifier(13)
    evaluator = GridSearchCV(model, {}, refit=False, cv=cv)
    evaluator.fit(df, y)
