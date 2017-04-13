import pytest
from click.testing import CliRunner
import os
import numpy as np
import pandas
import pycatchmod
from pycatchmod.__main__ import version, run, pandas_read

TEST_FOLDER = os.path.dirname(__file__)

@pytest.fixture
def runner():
    return CliRunner()

@pytest.fixture
def sample_data(tmpdir):
    """Create some sample rainfall and PET data"""
    shape = (200, 5)
    folder = str(tmpdir)
    rainfall = np.random.random(shape) * 100
    pet = np.zeros(shape, dtype=np.float64)
    index = pandas.date_range("1920-01-01", periods=200, freq="D")
    rainfall = pandas.DataFrame(rainfall, index=index)
    pet = pandas.DataFrame(pet, index=index)
    rainfall.to_csv(os.path.join(folder, "rainfall.csv"))
    pet.to_csv(os.path.join(folder, "pet.csv"))
    return {
        "folder": folder,
        "rainfall": os.path.join(folder, "rainfall.csv"),
        "pet": os.path.join(folder, "pet.csv"),
    }

def test_version(runner):
    result = runner.invoke(version, [])
    assert(result.exit_code == 0)
    assert(result.output == "pycatchmod "+pycatchmod.__version__+"\n")

@pytest.mark.parametrize("extension", ["csv", "h5"])
def test_run(runner, sample_data, extension):
    result = runner.invoke(run, [
        "--parameters", os.path.join(TEST_FOLDER, "data", "thames.json"),
        "--rainfall", sample_data["rainfall"],
        "--pet", sample_data["pet"],
        "--output", os.path.join(sample_data["folder"], "results."+extension),
    ])
    assert(result.exit_code == 0)
    assert(os.path.exists(os.path.join(sample_data["folder"], "results."+extension)))
    df = pandas_read(os.path.join(sample_data["folder"], "results."+extension))
    assert(df.shape == (200, 5))
    assert(pandas.Period(df.index[0]).year == 1920)
    assert(pandas.Period(df.index[0]).month == 1)
    assert(pandas.Period(df.index[0]).day == 1)

@pytest.mark.parametrize("extension", ["csv", "h5"])
def test_run3000(runner, tmpdir, extension):
    output_filename = str(tmpdir.join("results."+extension))
    result = runner.invoke(run, [
        "--parameters", os.path.join(TEST_FOLDER, "data", "thames.json"),
        "--rainfall", os.path.join(TEST_FOLDER, "data", "rainfall3000.csv"),
        "--pet", os.path.join(TEST_FOLDER, "data", "pet3000.csv"),
        "--output", output_filename,
    ])
    assert(result.exit_code == 0)
    assert(os.path.exists(output_filename))
    df = pandas_read(output_filename)
    assert(df.shape == (31, 1))
    assert(pandas.Period(df.index[0]).year == 3005)
    assert(pandas.Period(df.index[0]).month == 1)
    assert(pandas.Period(df.index[0]).day == 1)
