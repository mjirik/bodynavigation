import pytest
import io3d
import sed3
from . import kidney_segmentation

def test_trenovani():
    kidney_segmentation.vytiskni("ahoj")
    dp = io3d.read_dataset("3Dircadb1", 'data3d', 1)
    # ed = sed3.sed3(dp['data3d'])
    # ed.show()
    sed3.show_slices(dp['data3d'], slice_number=24)
def test_trainer():
    ol=kidney_segmentation.train([30, 30])
    kidney_segmentation.klasification(30, ol)
def test_train_and_save():
    ol = kidney_segmentation.train([30], show=True)
    import joblib
    joblib.dump(ol, 'ol.joblib')

def test_train_load():
    import joblib
    ol=joblib.load('ol.joblib')
    print(ol)
    kidney_segmentation.klasification(30, ol)