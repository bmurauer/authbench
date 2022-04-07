#!/usr/bin/env python3

import os
from glob import glob
from typing import Dict, List

import click
import pandas as pd

from scripts.unify_common import extract_texts_to_files, find_dir


@click.command()
@click.argument("directory")
@click.option("-f", "--force", help="Overwrite existing data", is_flag=True)
def unify_reuters_c50(directory: str, force: bool) -> None:
    processed_dir = os.path.join(os.path.dirname(directory), "processed")
    if not os.path.isdir(processed_dir):
        os.makedirs(processed_dir)
    directory = os.path.join(directory, "raw")

    output = os.path.join(processed_dir, "dataset.csv")
    if os.path.isfile(output) and not force:
        raise ValueError("not overwriting existing data.")

    train = find_dir(directory, "train")
    test = find_dir(directory, "test")

    def read(subdir: str, category: str) -> List[Dict]:
        posts = []
        authors = os.listdir(subdir)
        for author in authors:
            author_dir = os.path.join(subdir, author)
            files = glob(author_dir + "/*.txt")
            for f in files:
                with open(f) as i_f:
                    text = i_f.read()
                    posts.append(dict(category=category, author=author, text_raw=text))
        return posts

    rows_train = read(train, "train")
    rows_test = read(test, "test")

    pd.DataFrame.from_records(rows_train + rows_test).to_csv(output)
    print("wrote output to " + output)
    extract_texts_to_files(output, "text_raw", "text_raw")


if __name__ == "__main__":
    unify_reuters_c50()
