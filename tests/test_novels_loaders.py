import pandas as pd

from authbench.loaders import ClNovelsLoader


def test_novels_loader_2() -> None:
    available_languages = ["es", "en"]
    for language in available_languages:
        loader = ClNovelsLoader("data/cl_novels_2/processed", test_language=language)
        df, y, splits = loader.load()
        for train_idx, test_idx in splits:
            train_df = df.iloc[train_idx]
            train_y = y[train_idx]

            for (i, row), y_df in zip(train_df.iterrows(), train_y):
                assert row["author"] == y_df
                assert y_df in row["text_raw"]
                assert y_df in row["stanza"]

            test_df = df.iloc[test_idx]
            test_y = y[test_idx]

            for (i, row), y_df in zip(test_df.iterrows(), test_y):
                assert row["author"] == y_df
                assert y_df in row["text_raw"]
                assert y_df in row["stanza"]

            assert not train_df.empty
            assert not test_df.empty
            assert test_df.language.nunique() == 1
            assert test_df.language.unique()[0] == language
            assert language not in train_df.language.unique()
            assert set(test_df.author.unique()).issubset(set(train_df.author.unique()))


def test_all_novels_are_tested_once_2() -> None:
    df = pd.read_csv("data/cl_novels_2/processed/dataset.csv")
    all_titles = df.title.unique()

    available_languages = ["es", "en"]

    untested_titles = []
    for title in all_titles:
        was_tested = False
        for language in available_languages:
            loader = ClNovelsLoader(
                "data/cl_novels_2/processed", test_language=language
            )
            df, y, splits = loader.load()
            for train_idx, test_idx in splits:
                test_df = df.iloc[test_idx]
                if title in test_df.title.unique():
                    was_tested = True
                    break
            if was_tested:
                break
        if not was_tested:
            untested_titles.append(title)
    assert untested_titles == []
