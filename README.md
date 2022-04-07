# Authorship Attribution Benchmark

Dataset tools for the paper "Developing a Benchmark for Reducing Data Bias in Authorship Attribution".


## Requirements
- Python >= 3.8, < 4.
If you don't have this version on your computer, you might want to check out [pyenv](https://github.com/pyenv/pyenv).

- This project uses and requires [poetry](https://python-poetry.org) to manage dependencies. You can install it using:

  ```curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -```

  Then, install all dependencies of this project in a virtual environment by calling `poetry install`.
- Not all datasets described in the paper can't be distributed due to license agreements.
  Please ask the authors of the respective corpora to gain access.

## Dataset preparation

All steps in the following section are valid for all datasets. For simplicity, the examples will use the `cl_novels` dataset, which is shipped in this repo.

### 1. Bring to common format
The `scripts` directory contains scripts to transform each dataset into  common format.
They expect the raw dataset (as you would receive it from the original author) to be in the following position:

```
data/<name>/raw
```

For the cl_novels dataset, this means that the following instructions are required:

```bash
mkdir -p data/cl_novels/{raw,processed}
unzip datasets/cl_novels.zip -d data/cl_novels/raw
```

Then, the script to bring the raw dataset to the common format can be run:

```bash
poetry run python scripts/unify_cl_novels.py data/cl_novels
```

### 2. Parse dependencies

If you want to run an experiment depending on POS tags or dependency information, the dataset must be parsed using the stanza parser.
The parser requires some models to be downloaded beforehand.
This script will download the required models for all languages mentioned in the paper.

```bash
poetry run scripts/download_stanza_resources.sh
```

Then, the parser can be run on the unified data using the following command:

```bash
poetry run parse_dependency data/cl_novels_processed
```

## Usage

The basic usage is:

```python
loader = RedditLoader('path/to/reddit/dataset')
df, targets, splits = loader.load()
model = make_pipeline(...)   # some sklearn model
grid = GridSearchCV(model, params, cv=splits)
grid.fit(df, targets)
```

A working example is included in `tests/test_limiting_loaders.py`


## Cite

If you use this software, please consider citing my paper:

Benjamin Murauer and Günther Specht. 2021. Developing a Benchmark for Reducing Data Bias in Authorship Attribution. In Proceedings of the 2nd Workshop on Evaluation and Comparison of NLP Systems, pages 179–188, Punta Cana, Dominican Republic. Association for Computational Linguistics.

https://aclanthology.org/2021.eval4nlp-1.18/

Copyright (c) 2022 Benjamin Murauer
