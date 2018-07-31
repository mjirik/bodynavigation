# Datasets

Load all dataset infos with:
```
from bodynavigation.files import loadDatasetsInfo
datasets = loadDatasetsInfo()
```

## 3Dircadb1

Download with:
```
wget http://www.ircad.fr/softwares/3Dircadb/3Dircadb1/3Dircadb1.zip -O 3Dircadb1.zip
```
or
```
python -m io3d.datasets -l 3Dircadb1
```
Expected data structure is defined in `3Dircadb1.json` and `3Dircadb1_no-tumors.json`.

## 3Dircadb2

Download with:
```
wget https://www.ircad.fr/softwares/3Dircadb/3Dircadb2/3Dircadb2.zip -O 3Dircadb2.zip
```
Expected data structure is defined in `3Dircadb2.json`.

## sliver07

Download training data from `http://www.sliver07.org` (sets 001-020).

Expected data structure is defined in `sliver07.json`.
