"""
Test pycatchmod.utils module

"""
from pycatchmod.io.json import catchment_from_json
import pytest
import os

@pytest.fixture
def data_dir():
    this_dir = os.path.dirname(__file__)
    return os.path.join(this_dir, 'data')


@pytest.fixture
def thames_json(data_dir):
    return os.path.join(data_dir, 'thames.json')

def test_json(thames_json):
    catchment = catchment_from_json(thames_json)
    assert(catchment.name == 'Thames')
    assert(len(catchment.subcatchments) == 3)
    assert(catchment.subcatchments[0].area == 3900.0)
