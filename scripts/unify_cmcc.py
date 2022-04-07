#!/usr/bin/env python3

import os
import re
from glob import glob

import click
import pandas as pd

from scripts.unify_common import extract_texts_to_files


def check_or_fix_dataset_typo(directory: str) -> None:
    """
    There is one typo in the dataset which might have not been corrected yet:
    there is one file 'Discussions/Correlated/S1D113.txt'
    Which is the only file in the corpus that does not comply to the
    naming convention explained in FileCodingSchemes3.doc.

    It should be called S1D1I3.txt with an upper case i instead of a digit one.
    This code was tested on CMCCData.zip with a md5 checksum of:
        157586057cf4ad3dc1876890e94373a5
    """
    wrong = os.path.join(directory, "Discussion", "Correlated", "S1D113.txt")
    right = os.path.join(directory, "Discussion", "Correlated", "S1D1I3.txt")

    if os.path.isfile(wrong):
        print("renaming " + wrong + " to " + right)
        os.rename(wrong, right)


@click.command()
@click.argument("directory")
@click.option("-f", "--force", help="Overwrite existing data", is_flag=True)
def unify_cmcc(directory: str, force: bool) -> None:
    processed_dir = os.path.join(directory, "processed")
    if not os.path.isdir(processed_dir):
        os.makedirs(processed_dir)
    directory = os.path.join(directory, "raw")

    output = os.path.join(processed_dir, "dataset.csv")
    if os.path.isfile(output) and not force:
        raise ValueError("not overwriting existing data.")

    check_or_fix_dataset_typo(directory)

    posts = []
    categories = ["Blogs", "Chat", "Discussion", "Emails", "Essays", "Interviews"]

    for category in categories:
        correlated_dir = os.path.join(directory, category, "Correlated")
        files = glob(correlated_dir + "/*.txt")
        pattern = re.compile(
            r"(?P<author>[A-Z]\d+)(?P<genre>[A-Z])\d+(?P<topic>[A-Z])\d+.txt"
        )

        for f in files:
            # the files are windows-1252-encoded.
            with open(f, "rb") as i_f:
                try:
                    text_raw = i_f.read().decode("cp1252")
                except Exception as e:
                    print(f)
                    raise e

            name = os.path.basename(f)
            match = pattern.match(name)
            if not match:
                raise ValueError("no match found for file: " + f)

            posts.append(
                dict(category=category, text_raw=text_raw, **match.groupdict())
            )
    pd.DataFrame.from_records(posts).to_csv(output)
    print("wrote output to " + output)
    extract_texts_to_files(output, "text_raw", "text_raw")


if __name__ == "__main__":
    unify_cmcc()
