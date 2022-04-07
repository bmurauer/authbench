import langdetect
from tqdm import tqdm

from authbench.loaders import RedditLoader


def _reddit_loader_train(dataset: str, train_language: str, test_language: str) -> int:
    loader = RedditLoader(
        path=f"data/{dataset}/processed",
        train_languages=[train_language],
        test_languages=[test_language],
    )

    failed_count = 0

    (xtrain, _), (xtest, _) = loader.get_train_data(), loader.get_test_data()
    train_filenames = xtrain.text_raw.values
    train_texts = [open(f).read() for f in train_filenames]
    for text, filename in tqdm(
        zip(train_texts, train_filenames), total=len(train_filenames)
    ):
        detected = langdetect.detect(text)
        if detected != train_language:
            print(filename)
            print("expected it to be " + train_language + ", but was " + detected)
            failed_count += 1

    test_filenames = xtest.text_raw.values
    test_texts = [open(f).read() for f in test_filenames]
    for text, filename in tqdm(
        zip(test_texts, test_filenames), total=len(test_filenames)
    ):
        detected = langdetect.detect(text)
        if detected != test_language:
            print(filename)
            print("expected it to be " + test_language + ", but was " + detected)
            failed_count += 1

    return failed_count


def test_reddit_loaders_correct_languages() -> None:
    data = [
        ("R-CL5-FR", "en", "fr"),
        ("R-CL1-DE", "en", "de"),
    ]
    sum = 0
    for d in data:
        sum += _reddit_loader_train(*d)
    assert sum == 0
