#!/usr/bin/env python3

import os
from glob import glob
from typing import List

import click
import nltk
import pandas as pd
from dbispipeline.utils import LOGGER

tokenizers = {
    "en": nltk.data.load("tokenizers/punkt/english.pickle"),
    "es": nltk.data.load("tokenizers/punkt/spanish.pickle"),
    "fr": nltk.data.load("tokenizers/punkt/french.pickle"),
    "de": nltk.data.load("tokenizers/punkt/german.pickle"),
}


def chunkify(sentences: List[str], sentences_per_chunk: int) -> List[List[str]]:
    result = []
    for i in range(0, len(sentences), sentences_per_chunk):
        result.append(sentences[i : i + sentences_per_chunk])
    return result


@click.command()
@click.argument("directory")
@click.option("-f", "--force", help="Overwrite existing data", is_flag=True)
@click.option(
    "-cs", "--chunk-size", type=int, help="how many sentenc es per chunk", default=500
)
def unify_cl_novels(directory: str, force: bool, chunk_size: int) -> None:
    processed_dir = os.path.join(directory, "processed")
    if not os.path.isdir(processed_dir):
        LOGGER.info("creating %s", processed_dir)
        os.makedirs(processed_dir)
    text_raw_dir = os.path.join(directory, "processed", "text_raw")
    if not os.path.isdir(text_raw_dir):
        LOGGER.info("creating %s", text_raw_dir)
        os.makedirs(text_raw_dir)
    directory = os.path.join(directory, "raw")

    output = os.path.join(processed_dir, "dataset.csv")
    if os.path.isfile(output) and not force:
        raise ValueError("not overwriting existing data:", output)

    entries = []
    for author_dir in glob(f"{directory}/*"):
        author = os.path.basename(author_dir)
        for language_dir in glob(f"{author_dir}/*"):
            language = os.path.basename(language_dir)
            for text_file in glob(f"{language_dir}/*.txt"):
                text_file_name = os.path.splitext(os.path.basename(text_file))[0]
                with open(text_file) as input_fh:
                    text = input_fh.read()
                    sentences = tokenizers[language].tokenize(text)
                    chunks = chunkify(sentences, chunk_size)
                    for i, chunk in enumerate(chunks):
                        chunk_file = f"{author}_{language}_{text_file_name}_{i}.txt"
                        chunk_file_full = os.path.join("text_raw", chunk_file)
                        with open(os.path.join(text_raw_dir, chunk_file), "w") as o_fh:
                            o_fh.write(" ".join(chunk))
                        entries.append(
                            {
                                "author": author,
                                "language": language,
                                "title": text_file_name,
                                "text_raw": chunk_file_full,
                                "chunk": i,
                            }
                        )

    df = pd.DataFrame.from_records(entries)
    df.to_csv(output)


if __name__ == "__main__":
    unify_cl_novels()
