#!/bin/bash
# downloads and prepares all data sets required for testing
# http://www.ircad.fr/softwares/3Dircadb/3Dircadb1/3Dircadb1.zip
rm -f -r ./test_data && mkdir -p ./test_data # remove old test data
wget http://www.ircad.fr/softwares/3Dircadb/3Dircadb1/3Dircadb1.zip -O test_data.zip
unzip test_data.zip -d ./test_data && rm test_data.zip

for i in {1..20}
do
    mkdir -p "./test_data/$i" # make set dir
    unzip "./test_data/3Dircadb1.$i/MASKS_DICOM.zip" -d "./test_data/$i"
    unzip "./test_data/3Dircadb1.$i/PATIENT_DICOM.zip" -d "./test_data/$i"
    rm -f -r "./test_data/3Dircadb1.$i"
done

# link first set
ln -s $(readlink -m ./test_data/1/MASKS_DICOM) $(readlink -m ./test_data/MASKS_DICOM)
ln -s $(readlink -m ./test_data/1/PATIENT_DICOM) $(readlink -m ./test_data/PATIENT_DICOM)



