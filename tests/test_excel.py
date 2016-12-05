from pycatchmod.io.excel import (read_parameters, read_results,
    excel_parameter_adjustment, compare, open_workbook)
import json
import os
import pytest
import numpy as np

JSON_SAMPLE = os.path.join(os.path.dirname(__file__), "data", "thames.json")
EXCEL_SAMPLE = os.path.join(os.path.dirname(__file__), "data", "thames.xls")
@pytest.fixture
def wb():
    return open_workbook(EXCEL_SAMPLE)

def test_read_parameters(wb):
    name, subcatchments = read_parameters(wb)
    assert(name == "Thames")
    assert(len(subcatchments) == 3)

    # check all parameters have been read correctly by comparing them to the
    # json version of the model
    json_data = json.load(open(JSON_SAMPLE,"r"))
    parameter_names = list(json_data["subcatchments"][0].keys())
    for n in range(0, len(subcatchments)):
        for key in parameter_names:
            assert(subcatchments[n][key] == json_data["subcatchments"][n][key])

def test_read_results(wb):
    results = read_results(wb)

    # all arrays should have the same length
    for a in results.values():
        assert(len(a) == 31)

    assert(results["dates"][0] == np.datetime64("1920-01-01"))
    assert(results["dates"][-1] == np.datetime64("1920-01-31"))

    # we use some dummy values here to make testing easy

    assert(results["recharge"][-1] == 31)
    assert(results["smd_upper"][-1] == 62)
    assert(results["smd_lower"][-1] == 93)

    assert(results["flow_sim"][-1] == 124)
    assert(results["flow_obs"][-1] == 278)

    assert(results["rainfall"][-1] == 185)
    assert(results["pet"][-1] == 216)
