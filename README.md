# Authorship Attribution Benchmark

Experiments for the paper "Developing a Benchmark for Reducing Data Bias in Authorship Attribution".


## Requirements
- Python >= 3.6, < 3.9.
If you don't have this version on your computer, you might want to check out [pyenv](https://github.com/pyenv/pyenv).

- A GPU with at least 12GB RAM + CUDA >= 10.2

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


## Usage/Reproduction
This project uses the `dbispipeline` library, the configuration of this library is documented [here](https://git.uibk.ac.at/dbis/software/dbispipeline).
Thereby, each experiment is in a plan file located in `plans`.
For example, the file `plans/cpu/cmcc.py` contains the instructions for the CMCC dataset with all models that should run on a cpu.

To run an experiment, run:

```bash
poetry run dbispipeline plans/cpu/cmcc.py
```

If you don't want to store the results in a database, you can add the `--dryrun` parameter to locally print any results.

The plans are split into cpu and gpu to be able to run some experiments on machines without gpus in parallel.

Copyright (c) 2021 Benjamin Murauer
