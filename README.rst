***************************************************************************************
pycatchmod: A Cython implementation of the rainfall runoff model CATCHMOD (Wilby, 1994)
***************************************************************************************

CATCHMOD is widely used rainfall runoff model in the United Kingdom. It was introduced by Wilby (1994). This
version is developed in Python and utilises Cython to allow efficient execution of long time series.

========
Features
========

pycatchmod includes the following features,

- Core CATCHMOD algorithm written in Cython for speed,
- Simultaneous simulation of multiple input timeseries with the same catchment parameters, and
- Integration with numpy arrays.

====
TODO
====

The following features are planned, but yet to be completed (patches welcome!),

- Input-output routines to load input timeseries and save outputs,
- Integration with pandas.Series and/or DataFrames, and
- Implementation of Oudin (2005) PET formula to calculate PET from temperature.

=======
Licence
=======

This work is licenced under the GPLv3. Please see LICENCE for details.

=======
Authors
=======

James Tomlinson (<james.tomlinson@postgrad.manchester.ac.uk>)

================
Acknowledgements
================

This work was funded by the University of Manchester, United Kingdom.
