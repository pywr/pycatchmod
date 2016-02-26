"""
Test pycatchmod.utils module

"""
import pycatchmod.utils
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

    catchment = pycatchmod.utils.catchment_from_json(thames_json)

    assert catchment.name == 'Thames'