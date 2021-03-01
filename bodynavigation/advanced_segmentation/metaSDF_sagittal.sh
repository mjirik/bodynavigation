#PBS -l select=1:ncpus=2:ngpus=2:mem=32gb:scratch_local=0gb:cluster=adan -q gpu
#PBS -l walltime=10:00:00
# modify/delete the above given guidelines according to your job's needs
# Please note that only one select= argument is allowed at a time.

# add to qsub with:
# qsub metaSDF_sagittal.sh

# nastaveni domovskeho adresare, v promenne $LOGNAME je ulozeno vase prihlasovaci jmeno
DATADIR="/storage/plzen1/home/$LOGNAME/bodynavigation/bodynavigation/advanced_segmentation"
# nebo snad "/storage/plzen4-ntis/home/$LOGNAME/"  ?

# nacteni aplikacniho modulu, ktery zpristupni aplikaci Gaussian verze 3
# module add g03

echo "job: $PBS_JOBID running on: `uname -n`" >result # just an example computation

#SOURCE="${BASH_SOURCE[0]}"
#DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"

# nastaveni automatickeho vymazani adresare SCRATCH pro pripad chyby pri behu ulohy
trap 'clean_scratch' TERM EXIT

# vstup do adresare SCRATCH, nebo v pripade neuspechu ukonceni s chybovou hodnotou rovnou 1
cd $SCRATCHDIR || exit 1

# priprava vstupnich dat (kopirovani dat na vypocetni uzel)
# cp $DATADIR/gaussian_test.com $SCRATCHDIR

# spusteni aplikace - samotny vypocet
export PATH=/storage/plzen1/home/$LOGNAME/miniconda/bin:$PATH
#echo /storage/plzen1/home/$LOGNAME/projects/scaffan/experiments/metacentrum/SA_experiments.xlsx
# this is because of python click
#export LC_ALL=C.UTF-8
#export LANG=C.UTF-8

module add cuda
activate tf_gpu

# python -m scaffan set --common-spreadsheet-file /storage/plzen1/home/$LOGNAME/projects/scaffan/experiments/metacentrum/SA_experiments.xlsx
python /storage/plzen1/home/javorek/bodynavigation/bodynavigation/advanced_segmentation/metasagittal.py > /storage/plzen1/home/javorek/meta_sdf_output_sagittal.txt

# kopirovani vystupnich dat z vypocetnicho uzlu do domovskeho adresare,
# pokud by pri kopirovani doslo k chybe, nebude adresar SCRATCH vymazan pro moznost rucniho vyzvednuti dat
#cp results.out $DATADIR && cp -r SA_* $DATADIR || export CLEAN_SCRATCH=false
#cp results.out $DATADIR && cp scaffan.log $DATADIR && cp -r test_run_lobuluses_output_dir* $DATADIR || export CLEAN_SCRATCH=false
#cp results.out $DATADIR || export CLEAN_SCRATCH=false
