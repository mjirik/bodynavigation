#!/bin/bash
# downloads and prepares data required for testing
# http://www.ircad.fr/softwares/3Dircadb/3Dircadb1/3Dircadb1.1.zip
rm -f -r ./test_data && mkdir -p ./test_data # remove old test data
wget http://www.ircad.fr/softwares/3Dircadb/3Dircadb1/3Dircadb1.1.zip -O test_data.zip
unzip test_data.zip -d ./test_data && rm test_data.zip
unzip ./test_data/3Dircadb1.1/MASKS_DICOM.zip -d ./test_data
unzip ./test_data/3Dircadb1.1/PATIENT_DICOM.zip -d ./test_data
rm -f -r ./test_data/3Dircadb1.1
