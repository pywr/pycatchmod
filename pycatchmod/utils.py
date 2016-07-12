import json
import numpy as np
from ._catchmod import Catchment, SubCatchment


def catchment_from_json(filename, n=1):

    with open(filename) as fh:
        data = json.load(fh)


    subcatchments = []
    for subdata in data['subcatchments']:
        subcatchments.append(SubCatchment(
            subdata['area'],
            np.ones(n)*subdata['initial_upper_deficit'],
            np.ones(n)*subdata['initial_lower_deficit'],
            np.ones(n)*subdata['initial_linear_outflow'],
            np.ones(n)*subdata['initial_nonlinear_outflow'],
            direct_percolation=subdata['direct_percolation']/100.0,
            potential_drying_constant=subdata['potential_drying_constant'],
            gradient_drying_curve=subdata['gradient_drying_curve'],
            linear_storage_constant=subdata['linear_storage_constant'],
            nonlinear_storage_constant=subdata['nonlinear_storage_constant']
        ))

    return Catchment(subcatchments, name=data['name'])