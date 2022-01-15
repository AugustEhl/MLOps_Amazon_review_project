import pytest
import os
import torch
from tests import _PATH_DATA

@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data does not exist")
def test_data_length():
    test_data = torch.load(_PATH_DATA +'test.pth')
    assert len(test_data) == 27844, "Dataset did not have the correct number of samples"
