from typing import List, Tuple, Union

import numpy as np
import pandas as pd


def check_explicit_splits(
    df: pd.DataFrame,
    y: np.ndarray,
    splits: Union[List, np.ndarray],
) -> None:
    for train_idx, test_idx in splits:
        train_df = df.iloc[train_idx]
        train_y = y[train_idx]
        test_df = df.iloc[test_idx]
        test_y = y[test_idx]

        #  at least two distinctive classes
        assert len(set(train_y)) > 1

        #  every test class must be present for training
        test_authors = set(test_y)
        train_authors = set(train_y)
        assert test_authors.issubset(train_authors)

        #  at least two samples for training, at least one sample for testing
        assert train_df.shape[0] >= 2
        assert test_df.shape[0] >= 1

        #  as many labels as training samples
        assert train_df.shape[0] == train_y.shape[0]
        assert test_df.shape[0] == test_y.shape[0]

        for column in ["text_raw", "stanza"]:
            train = train_df[column].values
            test = test_df[column].values

            # no document may appear multiple times
            assert train.shape[0] == len(set(train))
            assert test.shape[0] == len(set(test))

            # no document may be in both train and test set
            if set(train).intersection(set(test)):
                print(set(train).intersection(set(test)))
                assert False
