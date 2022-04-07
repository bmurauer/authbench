from __future__ import annotations

import os
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def _load_df(path: str) -> pd.DataFrame:
    """Helper method to prepend current paths to paths in the dataset.csv files."""
    df = pd.read_csv(os.path.join(path, "dataset.csv"))
    unnamed_columns = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed_columns:
        df = df.drop(columns=unnamed_columns)
    df["text_raw"] = [os.path.join(path, x) for x in df.text_raw]
    df["stanza"] = [os.path.join(path, x) for x in df.stanza]
    return df


def _attach(df: pd.DataFrame, y: np.ndarray) -> Tuple[pd.DataFrame, str]:
    df = df.copy()
    # attach the column to the dataframe for grouping
    key_i = 0
    key = f"y_{key_i}"
    while key in df.columns:
        key_i += 1
        key = f"y_{key_i}"
    df[key] = y
    return df, key


def _limit(
    dataset_part: Tuple[pd.DataFrame, np.ndarray],
    remaining_targets: List[str],
    max_docs_per_target: Optional[int],
) -> Tuple[pd.DataFrame, np.ndarray]:
    df, key = _attach(dataset_part[0], dataset_part[1])
    sub_df = df[df[key].isin(remaining_targets)]
    if max_docs_per_target:
        sub_df = sub_df.groupby(key).sample(max_docs_per_target)
    return sub_df.drop(columns=key), sub_df[key].values


class Loader(ABC):
    """Abstract base class of a dataloader."""

    @abstractmethod
    def load(self) -> Any:
        """Returns the data loaded by the dataloader."""
        pass

    @property
    @abstractmethod
    def configuration(self) -> Dict:
        """Returns a dict-like representation of the configuration."""
        pass


class CrossValidatedSplitLoader(Loader):
    """
    Base class for all loaders that don't have an explicit train/test split.

    A Stratified K-Fold is used to split the data, and the resulting splits are
    used for the explicit splits which can be used by the grid search 'cv'
    parameter.
    """

    def __init__(
        self,
        n_splits: int = 5,
        max_targets: int = None,
        max_docs_per_target: int = None,
    ):
        """
        Initialize the loader.

        Args:
            n_splits: number of splits to be using for this CV-loader.
            max_targets: Maximum number of labels to be used. If this value is
                provided, a subset of all possible targets is used for both
                training and testing.
            max_docs_per_target: Maximum number of documents used for training
                each target. Does not influence testing data. Optional.
        """
        self.n_splits = n_splits
        self.max_targets = max_targets
        self.max_docs_per_target = max_docs_per_target

    def load(self) -> Tuple[pd.DataFrame, np.ndarray, List[np.ndarray]]:
        """
        Loads the data and the splits.

        This method gets all data from the abstract method `get_all_data`, and
        applies the stratified cv splitting as well as the optional limiting of
        targets or documents per target.

        Returns:
            A tuple of x, y, splits. The splits are something that can be
            passed to the GridSearchCV object as the 'cv' parameter.
        """
        x, y = self.get_all_data()
        x, key = _attach(x, y)
        all_targets = x[key].unique()

        if self.max_targets:
            selected_targets = random.sample(all_targets.tolist(), self.max_targets)
            # only take those rows with the selected targets
            x = x[x[key].isin(selected_targets)]
            x = x.reset_index(drop=True)

        all_splits = StratifiedKFold(n_splits=self.n_splits).split(
            # the first argument (X) is not used in a stratified k-fold split.
            np.zeros(x.shape[0]),
            x[key],
        )
        if not self.max_docs_per_target:
            splits = list(all_splits)
        else:
            splits = []
            for train_idx, test_idx in all_splits:
                df_train = pd.DataFrame(dict(idx=train_idx, y=x[key][train_idx]))
                df_train = df_train.groupby("y").sample(self.max_docs_per_target)
                splits.append((df_train.idx.values, test_idx))
        return x.drop(columns=[key]), x[key].values, splits

    @abstractmethod
    def get_all_data(self) -> pd.DataFrame:
        """
        Retrieves the entire data from which the splits are taken.

        Returns:
            A tuple of x, y, splits. The splits are something that can be
            passed to the GridSearchCV object as the 'cv' parameter.
        """
        pass

    @property
    def configuration(self) -> dict:
        """Returns the database representation of this loader."""
        return {
            "n_splits": self.n_splits,
            "max_targets": self.max_targets,
            "max_docs_per_target": self.max_docs_per_target,
        }


class TrainTestSplitLoader(Loader):
    """Base class for all Loaders that have an explicit Train/Test split."""

    def __init__(self, max_targets: int = None, max_docs_per_target: int = None):
        """
        Initialize the loader.

        Args:
            max_targets: Maximum number of labels to be used. If this value is
                provided, a subset of all possible targets is used for both
                training and testing.
            max_docs_per_target: Maximum number of documents used for training
                each target. Does not influence testing data. Optional.
        """
        self.max_targets = max_targets
        self.max_docs_per_target = max_docs_per_target

    def load(self) -> Tuple[pd.DataFrame, np.ndarray, List]:
        """
        Loads the data and the splits.

        This method gets all data from the abstract method `get_train_data` and
        `get_test_data`, and then calculates the appropriate split indices
        while considering the optional limiting of any targets or documents per
        target.

        Returns:
            A tuple of x, y, splits. The splits are something that can be
            passed to the GridSearchCV object as the 'cv' parameter.
        """
        train, test = self.get_train_data(), self.get_test_data()
        all_targets = set(train[1])
        if self.max_targets:
            selected_targets = random.sample(all_targets, self.max_targets)
        else:
            selected_targets = list(all_targets)
        train = _limit(train, selected_targets, self.max_docs_per_target)
        test = _limit(test, selected_targets, None)  # don't limit test data
        train_idx = list(range(train[0].shape[0]))
        test_idx = list(range(train[0].shape[0], train[0].shape[0] + test[0].shape[0]))
        splits = [(train_idx, test_idx)]
        df = pd.concat([train[0], test[0]])
        y = np.concatenate([train[1], test[1]])
        return df, y, splits

    @abstractmethod
    def get_train_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Retrieves the training data from the subclass.

        Returns:
            A tuple of training data in form of [DataFrame, np.ndarray]
        """
        pass

    @abstractmethod
    def get_test_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Retrieves the testing data from the subclass.

        Returns:
            A tuple of training data in form of [DataFrame, np.ndarray]
        """
        pass

    @property
    def configuration(self) -> dict:
        """Returns the database representation of this loader."""
        return {
            "max_targets": self.max_targets,
            "max_docs_per_target": self.max_docs_per_target,
        }


class ClNovelsLoader(Loader):
    def __init__(self, path: str, test_language: str, **kwargs: Any):
        self.df = _load_df(path)
        self.df = self.df.reset_index(drop=True)
        self.path = path
        self.test_language = test_language

    def load(self) -> Tuple[pd.DataFrame, np.ndarray, List[Any]]:
        # the speciality of this dataset is that some novels might be available in both
        # the train and test set. In theses cases, the novel should only be used for
        # testing, and be removed from the training set.
        test_df = self.df[self.df.language == self.test_language]
        train_df = self.df[self.df.language != self.test_language]
        train_titles = set(train_df.title.unique())
        test_titles = set(test_df.title.unique())

        splits = []

        # first, handle all overlapping titles.
        # every time a title is in both training and testing sets, a split as used where
        # only that title is tested, and it is removed from the training set.
        overlapping_titles = train_titles.intersection(test_titles)
        for title in overlapping_titles:
            sub_train_df = train_df[train_df.title != title]
            sub_test_df = test_df[test_df.title == title]
            # some authors only have one novel in a language, but that novel is also
            # in the testing set, no training data is left -> ignore those splits.
            if not set(sub_test_df.author.values).issubset(
                set(sub_train_df.author.values)
            ):
                continue
            splits.append((sub_train_df.index.values, sub_test_df.index.values))

        # then, for all titles that don't overlap, one single split can be added.
        non_overlapping_titles = test_titles - overlapping_titles
        train_idx = train_df[~train_df.title.isin(non_overlapping_titles)].index.values
        test_idx = test_df[test_df.title.isin(non_overlapping_titles)].index.values
        splits.append((train_idx, test_idx))

        return self.df, self.df.author.values, splits

    @staticmethod
    def get_all_configurations() -> List[ClNovelsLoader]:
        return [
            ClNovelsLoader("data/cl_novels/processed", "es"),
            ClNovelsLoader("data/cl_novels/processed", "en"),
        ]

    @property
    def configuration(self) -> dict:
        return {
            **super().configuration,
            "path": self.path,
            "test_language": self.test_language,
        }


class CMCCLoader(TrainTestSplitLoader):
    def __init__(
        self,
        domain: str,
        test_domain: str,
        path: str = "data/cmcc/processed",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.domain = domain
        self.test_domain = test_domain
        self.df = _load_df(path)

    @staticmethod
    def get_all_configurations() -> List[CMCCLoader]:
        loaders = []
        for test_domain in ["C", "G", "I", "M", "P", "S"]:
            loaders.append(CMCCLoader(domain="topic", test_domain=test_domain))
        for test_domain in ["B", "C", "D", "E", "P", "S"]:
            loaders.append(CMCCLoader(domain="genre", test_domain=test_domain))
        return loaders

    def get_train_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        train_df = self.df[self.df[self.domain] != self.test_domain]
        return train_df, train_df.author.values

    def get_test_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        test_df = self.df[self.df[self.domain] == self.test_domain]
        return test_df, test_df.author.values

    @property
    def configuration(self) -> dict:
        return {
            **super().configuration,
            "domain": self.domain,
            "test_domain": self.test_domain,
        }


class ReutersLoader(TrainTestSplitLoader):
    def __init__(self, path: str = "data/reuters-c50/processed", **kwargs: Any):
        super().__init__(**kwargs)
        self.df = _load_df(path)

    def get_train_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        train_df = self.df[self.df["category"] == "train"]
        return train_df, train_df["author"].values

    def get_test_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        test_df = self.df[self.df["category"] == "test"]
        return test_df, test_df["author"].values

    @staticmethod
    def limited() -> List[Loader]:
        return [
            ReutersLoader(max_docs_per_target=m) for m in [5, 10, 15, 20, 30, 40, 50]
        ]


class GuardianLoader(TrainTestSplitLoader):
    """
    Load the guardian dataset in five batches.

    This loader creates five batches in a leave-one-category out strategy.
    """

    def __init__(
        self,
        train_categories: List[str],
        test_categories: List[str],
        path: str = "data/guardian/processed",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        assert set(train_categories).isdisjoint(test_categories)

        self.test_categories = test_categories
        self.train_categories = train_categories
        self.df = _load_df(path)

    @staticmethod
    def get_all_configurations() -> List[GuardianLoader]:
        return [
            # cross-genre
            GuardianLoader(
                train_categories=["Politics", "Society", "World", "UK"],
                test_categories=["Books"],
            ),
            GuardianLoader(
                train_categories=["Books"],
                test_categories=["Politics", "Society", "World", "UK"],
            ),
            # cross-topic
            GuardianLoader(
                train_categories=["Society", "World", "UK"],
                test_categories=["Politics"],
            ),
            GuardianLoader(
                train_categories=["Politics", "World", "UK"],
                test_categories=["Society"],
            ),
            GuardianLoader(
                train_categories=["Politics", "Society", "UK"],
                test_categories=["World"],
            ),
            GuardianLoader(
                train_categories=["Politics", "Society", "World"],
                test_categories=["UK"],
            ),
        ]

    def get_train_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        train_df = self.df[self.df["category"].isin(self.train_categories)]
        return train_df, train_df.author.values

    def get_test_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        test_df = self.df[self.df["category"].isin(self.test_categories)]
        return test_df, test_df.author.values

    @property
    def configuration(self) -> dict:
        return {
            **super().configuration,
            "test_categories": self.test_categories,
            "train_categories": self.train_categories,
        }


class ImdbLoader(CrossValidatedSplitLoader):
    def __init__(self, path: str = "data/imdb62/processed", **kwargs: Any):
        super().__init__(**kwargs)
        self.df = _load_df(path)

    def get_all_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        return self.df, self.df.author.values

    @staticmethod
    def limited() -> List[ImdbLoader]:
        return [ImdbLoader(max_docs_per_target=m) for m in [5, 10, 15, 20, 30, 40, 50]]


class Pan18AttributionLoader(TrainTestSplitLoader):
    def __init__(
        self,
        problem: str,
        path: str = "data/pan18_attribution/processed",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        df = _load_df(path)
        self.df = df[df.problem == problem]
        self.problem = problem

    @staticmethod
    def get_all_configurations() -> List[Pan18AttributionLoader]:
        return [
            Pan18AttributionLoader(problem=f"problem{i + 1:05d}") for i in range(10)
        ]

    def get_train_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        train_df = self.df[self.df.data_type == "train"]
        return train_df, train_df.author.values

    def get_test_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        test_df = self.df[self.df.data_type == "test"]
        return test_df, test_df.author.values

    @property
    def configuration(self) -> dict:
        return {
            **super().configuration,
            "problem": self.problem,
        }


class RedditLoader(TrainTestSplitLoader):
    def __init__(
        self,
        path: str,
        train_languages: List[str],
        test_languages: List[str],
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        df = _load_df(path)
        # sneaky file managed to get a wrong language assigned, only one though.
        # just filter it out.
        df = df[
            df.text_raw != "data/R-CL5-FR/processed/text_raw/jeannaimard_en_000176.txt"
        ]
        available_languages = set(df.language.unique())
        if not set(train_languages).issubset(available_languages):
            raise ValueError("invalid train languages")
        if not set(test_languages).issubset(available_languages):
            raise ValueError("invalid test languages")
        if set(train_languages).intersection(set(test_languages)):
            raise ValueError("")

        self.train_languages = train_languages
        self.test_languages = test_languages
        self.path = path
        self.df = df

    @staticmethod
    def get_all_configurations() -> List[RedditLoader]:
        return [
            RedditLoader("data/R-CL1-DE/processed", ["en"], ["de"]),
            RedditLoader("data/R-CL1-DE/processed", ["de"], ["en"]),
            RedditLoader("data/R-CL2-ES/processed", ["en"], ["es"]),
            RedditLoader("data/R-CL2-ES/processed", ["es"], ["en"]),
            RedditLoader("data/R-CL3-PT/processed", ["en"], ["pt"]),
            RedditLoader("data/R-CL3-PT/processed", ["pt"], ["en"]),
            RedditLoader("data/R-CL4-NL/processed", ["en"], ["nl"]),
            RedditLoader("data/R-CL4-NL/processed", ["nl"], ["en"]),
            RedditLoader("data/R-CL5-FR/processed", ["en"], ["fr"]),
            RedditLoader("data/R-CL5-FR/processed", ["fr"], ["en"]),
        ]

    def get_train_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        train_df = self.df[self.df.language.isin(self.train_languages)]
        return train_df, train_df.author.values

    def get_test_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        test_df = self.df[self.df.language.isin(self.test_languages)]
        return test_df, test_df.author.values

    @property
    def configuration(self) -> dict:
        return {
            **super().configuration,
            "train_languages": self.train_languages,
            "test_languages": self.test_languages,
            "path": self.path,
        }


def get_all_auth_bench_loaders() -> List[Loader]:
    return [
        ReutersLoader(),
        *ReutersLoader.limited(),
        *CMCCLoader.get_all_configurations(),
        *GuardianLoader.get_all_configurations(),
        ImdbLoader(),
        *ImdbLoader.limited(),
        *ClNovelsLoader.get_all_configurations(),
        *Pan18AttributionLoader.get_all_configurations(),
        *RedditLoader.get_all_configurations(),
    ]
