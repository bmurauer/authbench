#!/usr/bin/env python3

import os

import click
import pandas as pd

from scripts.unify_common import extract_texts_to_files


@click.command()
@click.argument("directory")
@click.option("-f", "--force", help="Overwrite existing data", is_flag=True)
def unify_imdb62(directory: str, force: bool) -> None:
    processed_dir = os.path.join(directory, "processed")
    if not os.path.isdir(processed_dir):
        os.makedirs(processed_dir)
    directory = os.path.join(directory, "raw")

    output = os.path.join(processed_dir, "dataset.csv")
    if os.path.isfile(output) and not force:
        raise ValueError("not overwriting existing data.")

    # in the original zip (md5 sum: e75089d9a050e6e119d4989d591ec900), the data
    # is stored in a tab-separated file called 'imdb62.txt'.

    columns = [
        "review_id",
        "author",
        "item_id",
        "rating",
        "title",
        "text_raw",
    ]
    df = pd.read_csv(os.path.join(directory, "imdb62.txt"), sep="\t", names=columns)
    df.to_csv(output)
    print("wrote output to " + output)
    extract_texts_to_files(output, "text_raw", "text_raw")


if __name__ == "__main__":
    unify_imdb62()
