from authbench.loaders import GuardianLoader
from tests.utils import check_explicit_splits


def test_guardian_loader() -> None:
    for loader in GuardianLoader.get_all_configurations():
        df, y, splits = loader.load()
        check_explicit_splits(df, y, splits)
