#!/usr/bin/env python3

import json
import os
from glob import glob

import click
import pandas as pd

from scripts.unify_common import extract_texts_to_files


def read_problem_train(problem: dict, base_directory: str) -> pd.DataFrame:
    directory = problem["problem-name"]
    language = problem["language"]
    if language == "sp":
        language = "es"
    candidates = glob(f"{base_directory}/{directory}/candidate*")
    documents = []
    for candidate in candidates:
        for text_file in glob(f"{candidate}/known*.txt"):
            with open(text_file) as doc_i_f:
                documents.append(
                    {
                        "author": os.path.basename(candidate),
                        "language": language,
                        "text_raw": doc_i_f.read(),
                    }
                )
    return pd.DataFrame(documents)


def read_problem_test(problem: dict, base_directory: str) -> pd.DataFrame:
    directory = os.path.join(base_directory, problem["problem-name"])
    language = problem["language"]
    if language == "sp":
        language = "es"
    truth = json.load(open(os.path.join(directory, "ground-truth.json")))
    documents = []
    for document in truth["ground_truth"]:
        filename = os.path.join(directory, "unknown", document["unknown-text"])
        with open(filename) as doc_i_f:
            documents.append(
                {
                    "author": document["true-author"],
                    "language": language,
                    "text_raw": doc_i_f.read(),
                }
            )
    return pd.DataFrame(documents)


def load_pan18_dataframe(base_directory: str) -> pd.DataFrame:
    dfs = []
    with open(os.path.join(base_directory, "collection-info.json")) as i_f:
        for problem in json.load(i_f):
            df_train = read_problem_train(problem, base_directory)
            df_train["data_type"] = ["train"] * df_train.shape[0]
            df_test = read_problem_test(problem, base_directory)
            df_test["data_type"] = ["test"] * df_test.shape[0]
            df = pd.concat([df_train, df_test])
            df["problem"] = [problem["problem-name"]] * df.shape[0]
            dfs.append(df)
    return pd.concat(dfs)


@click.command()
@click.argument("directory")
@click.option("-f", "--force", help="Overwrite existing data", is_flag=True)
def unify_pan18(directory: str, force: bool) -> None:
    processed_dir = os.path.join(directory, "processed")
    if not os.path.isdir(processed_dir):
        os.makedirs(processed_dir)
    directory = os.path.join(directory, "raw")

    output = os.path.join(processed_dir, "dataset.csv")
    if os.path.isfile(output) and not force:
        raise ValueError("not overwriting existing data.")

    load_pan18_dataframe(directory).to_csv(output)
    print("wrote output to " + output)
    extract_texts_to_files(output, "text_raw", "text_raw")


if __name__ == "__main__":
    unify_pan18()
