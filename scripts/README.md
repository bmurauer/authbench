# Prepratation Scripts

This folder contains the scripts that aim to bring datasets to a common format, which is expected by the loader classes.
The script expect the original data of dataset _foobar_ to be in a directory called
```
data/foobar/raw
```
and will write the unified representation into
```
data/foobar/processed
```

The loader expect the data to be split into meta data and content. The meta data is stored in a file called `dataset.csv`, and for most datasets, the content is split into separate files, one for each document, e.g.:
```
# dataset.csv

author,text_raw,stanza
author1,text_raw/document001.txt,stanza/document001.pckl
author1,text_raw/document002.txt,stanza/document002.pckl
```

In this example, each document has a raw text part (which is stored in the directroy `text_raw`) and a pickle file containing a parsed stanza document object (in `stanza/...`).
The meta data contains only the paths to those files, as some models will require the raw text while others will only required other files.

For small datasets, this is probably not required, and all data could be stored in the csv file directly. 
