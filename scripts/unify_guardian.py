#!/usr/bin/env python3

import os
from glob import glob

import click
import pandas as pd

from scripts.unify_common import extract_texts_to_files


@click.command()
@click.argument("directory")
@click.option("-f", "--force", help="Overwrite existing data", is_flag=True)
def unify_guardian(directory: str, force: bool) -> None:
    processed_dir = os.path.join(directory, "processed")
    if not os.path.isdir(processed_dir):
        os.makedirs(processed_dir)
    directory = os.path.join(directory, "raw")

    output = os.path.join(processed_dir, "dataset.csv")
    if os.path.isfile(output) and not force:
        raise ValueError("not overwriting existing data.")

    posts = []
    categories = os.listdir(directory)
    for category in categories:
        c_dir = os.path.join(directory, category)
        if not os.path.isdir(c_dir):
            continue
        authors = os.listdir(c_dir)
        for author in authors:
            author_dir = os.path.join(c_dir, author)
            files = glob(author_dir + "/*.txt")
            for f in files:
                # files are windows-encoded
                with open(f, "rb") as i_f:
                    try:
                        posts.append(
                            dict(
                                category=category,
                                author=author,
                                text_raw=i_f.read().decode("cp1252"),
                            )
                        )
                    except Exception as e:
                        print(f)
                        raise e

    pd.DataFrame.from_records(posts).to_csv(output)
    print("wrote output to " + output)
    extract_texts_to_files(output, "text_raw", "text_raw")


if __name__ == "__main__":
    unify_guardian()
