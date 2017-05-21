#!/bin/bash
# downloads and prepares one data set required for testing
# http://www.ircad.fr/softwares/3Dircadb/3Dircadb1/3Dircadb1.1.zip
rm -f -r ./test_data && mkdir -p ./test_data # remove old test data
wget http://www.ircad.fr/softwares/3Dircadb/3Dircadb1/3Dircadb1.1.zip -O test_data.zip
unzip test_data.zip -d ./test_data && rm test_data.zip

mkdir -p ./test_data/1 # make set dir
unzip ./test_data/3Dircadb1.1/MASKS_DICOM.zip -d ./test_data/1
unzip ./test_data/3Dircadb1.1/PATIENT_DICOM.zip -d ./test_data/1
rm -f -r ./test_data/3Dircadb1.1

# link first set
ln -s $(readlink -m ./test_data/1/MASKS_DICOM) $(readlink -m ./test_data/MASKS_DICOM)
ln -s $(readlink -m ./test_data/1/PATIENT_DICOM) $(readlink -m ./test_data/PATIENT_DICOM)
