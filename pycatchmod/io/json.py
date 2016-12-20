from __future__ import absolute_import
import json

import numpy as np
from .._catchmod import Catchment, SubCatchment, OudinCatchment
from past.builtins import basestring

def catchment_from_json(filename, n=1):
    if isinstance(filename, dict):
        # argument is already a python dictionary
        data = filename
    elif isinstance(filename, basestring):
        # argument is a filename
        with open(filename) as fh:
            data = json.load(fh)
    elif hasattr(filename, "read"):
        # argument is file-like
        data = json.load(filename.read())
    else:
        raise TypeError("Unexpected input type: \"{}\"".format(filename.__class__.__name__))

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
            nonlinear_storage_constant=subdata['nonlinear_storage_constant'],
            legacy=data.get("legacy", False)
        ))

    klass = data.get('class', 'Catchment')
    if klass == 'Catchment':
        return Catchment(subcatchments, name=data['name'])
    elif klass == 'OudinCatchment':
        return OudinCatchment(subcatchments, name=data['name'], latitude=np.deg2rad(data['latitude']))
    else:
        raise ValueError("Catchment class ({}) not recognised.".format(klass))
