import os
import xml.etree.cElementTree as cET
from glob import glob
from typing import List

import pandas as pd
from dbispipeline.utils import LOGGER
from tqdm import tqdm


def find_dir(base_directory: str, subdirectory_name_part: str) -> str:
    files = os.listdir(base_directory)
    subdirs = [f for f in files if os.path.isdir(os.path.join(base_directory, f))]
    for s in subdirs:
        if subdirectory_name_part in s:
            return os.path.join(base_directory, s)
    raise ValueError(
        'no subdirectory containing "' + subdirectory_name_part + '" found.'
    )


def set_languages(languages: List[str], train_dir: str, test_dir: str) -> List[str]:

    available_train_languages: List[str] = [
        os.path.basename(x) for x in glob(f"{train_dir}/*") if os.path.isdir(x)
    ]
    available_test_languages: List[str] = [
        os.path.basename(x) for x in glob(f"{test_dir}/*") if os.path.isdir(x)
    ]
    if languages is None:
        languages = available_train_languages
    for language in languages:
        if language not in available_train_languages:
            raise ValueError(
                f"language {language} not avail. f. training. Found these "
                f"training languages: {available_train_languages}"
            )
        if language not in available_test_languages:
            raise ValueError(
                f"language {language} not avail. f. testing. Found these "
                f"testing languages: {available_test_languages}",
                test_dir,
            )
    if not languages:
        raise ValueError("no languages found in", train_dir, test_dir)
    print("set languages to", languages)
    return languages


def parse_truth(lang: str, directory: str, truth_columns: List[str]) -> dict:
    result = {}
    possible_locations = [
        os.path.join(directory, lang, "truth.txt"),
        os.path.join(directory, f"truth-{lang}.txt"),
        # e.g. for 2013
    ]
    truth_found = False
    for f in possible_locations:
        if not os.path.isfile(f):
            continue
        LOGGER.info("reading truth file from %s", f)
        truth_found = True
        with open(f) as i_f:
            for line in i_f:
                parts = line.strip().split(":::")
                if len(parts) == len(truth_columns):
                    result[parts[0]] = {k: v for k, v in zip(truth_columns, parts)}
                else:
                    raise ValueError(
                        "truth columns don't match: %s vs. %s",
                        str(truth_columns),
                        str(parts),
                    )

        break
    if not truth_found:
        raise Exception("no truth file found in " + str(possible_locations))
    return result


def _load(
    directory: str,
    languages: List[str],
    truth_columns: List[str],
    xml_search_path: str,
) -> pd.DataFrame:
    documents = []
    for lang in languages:
        truth = parse_truth(lang, directory, truth_columns)
        files = sorted(glob(f"{directory}/{lang}/*.xml"))
        for f in files:
            author_id = os.path.splitext(os.path.basename(f))[0]
            try:
                tree = cET.parse(f).getroot()
                for document in tree.findall(xml_search_path):
                    doc = {
                        "author": author_id,
                        "language": lang,
                        "text_raw": document.text,
                    }
                    if truth:
                        doc.update(truth[author_id])
                    documents.append(doc)
            except Exception as e:
                LOGGER.error("error on parsing file: %s", f)
                raise e
    df = pd.DataFrame(documents)
    LOGGER.debug("loaded dataframe with shape: %s", df.shape)
    return df


def load_pan_dataframe_from_xml(
    base_directory: str,
    truth_columns: List[str],
    xml_search_path: str,
    languages: List[str],
) -> pd.DataFrame:
    if not os.path.isdir(base_directory):
        raise ValueError(f"Directory does not exist: {base_directory}")
    train_dir = find_dir(base_directory, "train")
    test_dir = find_dir(base_directory, "test")

    languages = set_languages(languages, train_dir, test_dir)

    df_train = _load(train_dir, languages, truth_columns, xml_search_path)
    df_train["data_type"] = ["train"] * df_train.shape[0]
    df_test = _load(test_dir, languages, truth_columns, xml_search_path)
    df_test["data_type"] = ["test"] * df_test.shape[0]
    return pd.concat([df_train, df_test])


def extract_texts_to_files(
    dataset_path: str,
    text_column_name: str,
    output_directory: str,
) -> None:
    directory = os.path.dirname(dataset_path)
    df = pd.read_csv(dataset_path, index_col=False).dropna()
    df = df.drop(columns=[c for c in df.columns if "Unnamed" in c])
    target_directory = os.path.join(directory, output_directory)
    if not os.path.isdir(target_directory):
        os.makedirs(target_directory)

    filenames = []
    for index, x in tqdm(df.iterrows(), total=df.shape[0]):
        filename = f"{index}_{x.author}.txt"
        output_filename = os.path.join(directory, output_directory, filename)
        with open(output_filename, "w") as o_fh:
            o_fh.write(x[text_column_name])
            filenames.append(os.path.join(output_directory, filename))
    df.text_raw = filenames
    if "language" not in df.columns:
        df["language"] = ["en"] * df.shape[0]
    df.to_csv(dataset_path)
