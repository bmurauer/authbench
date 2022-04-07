from typing import Dict, List

from authbench.loaders import (
    ClNovelsLoader,
    CMCCLoader,
    GuardianLoader,
    ImdbLoader,
    Loader,
    Pan18AttributionLoader,
    RedditLoader,
    ReutersLoader,
)
from tests.utils import check_explicit_splits


def test_dataloaders() -> None:

    loader_lists: Dict[str, List] = {
        "C50": [ReutersLoader()],
        "CMCC": CMCCLoader.get_all_configurations(),
        "Guardian": GuardianLoader.get_all_configurations(),
        "IMDb": [ImdbLoader()],
        "Novels": ClNovelsLoader.get_all_configurations(),
        "Pan18": Pan18AttributionLoader.get_all_configurations(),
        "Reddit": RedditLoader.get_all_configurations(),
    }

    for name, loaders in loader_lists.items():
        print(name)
        for loader in loaders:
            df, y, splits = loader.load()
            check_explicit_splits(df, y, splits)
