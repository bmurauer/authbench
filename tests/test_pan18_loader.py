from authbench.loaders import Pan18AttributionLoader
from tests.utils import check_explicit_splits


def test_pan18_loader() -> None:

    for i in range(10):
        loader = Pan18AttributionLoader(problem=f"problem{i+1:05d}")
        df, y, splits = loader.load()
        check_explicit_splits(df, y, splits)
