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