***************************************************************************************
pycatchmod: A Cython implementation of the rainfall runoff model CATCHMOD (Wilby, 1994)
***************************************************************************************

CATCHMOD is widely used rainfall runoff model in the United Kingdom. It was introduced by Wilby (1994). This
version is developed in Python and utilises Cython to allow efficient execution of long time series.


.. image:: https://travis-ci.org/pywr/pycatchmod.svg?branch=master
   :target: https://travis-ci.org/pywr/pycatchmod


========
Features
========

pycatchmod includes the following features:

- Core CATCHMOD algorithm written in Cython for speed,
- Simultaneous simulation of multiple input timeseries with the same catchment parameters, and
- Integration with numpy arrays.
- Implementation of Oudin (2005) PET formula to calculate PET from temperature.
- A command line interface

============
Installation
============

The module is primarily written in Cython and will need a C compiler installed to build. The code has been tested successfully with GCC (Linux), MSVC (Windows) and clang (OS X).

To install, use :code:`setup.py` as you normally would:

.. code-block:: console

    python setup.py install

Once installed, tests can be run using the :code:`py.test` command:

.. code-block:: console

    py.test tests

======================
Command line interface
======================

A command line interface has been written for convenience. This is installed as the :code:`pycatchmod` command. See:

.. code-block:: console

    pycatchmod --help

You can access the help for each of the sub-commands using the :code:`--help` switch, e.g.:

.. code-block:: console

    pycatchmod run --help

To run a model, use the :code:`run` sub-command:

.. code-block:: console

    pycatchmod run --parameters thames.json --rainfall thames_rainfall.csv --pet thames_pet.csv --output thames_flow.csv

The command line interface also provides some tools for working with the Excel implementation of CATCHMOD. The parameters from a model can be extracted from an Excel file using :code:`dump` e.g.:

.. code-block:: console

    pycatchmod dump --filename thames.xls

The parameters are printed in JSON format to the standard output (STDOUT). An example of this format can be found in the :code:`tests` directory. This data can be redirected into a file using a pipe:

.. code-block:: console

    pycatchmod dump --filename thames.xls > thames.json

You can use the :code:`compare` command to compare the results of pycatchmod and an Excel model. Any (significant) differences between the outputs is considered a bug (and should be reported via GitHub).

.. code-block:: console

    pycatchmod compare --filename thames.xls --plot

=======
Changes
=======

Version 1.1.0
=============

- Added feature to output flow per area.

Version 1.0.0
=============

- Initial release


=======
Licence
=======

This work is licenced under the GNU General Public Licence Version 3 (GNU GPLv3). Please see LICENCE for details.

=======
Authors
=======

- James Tomlinson (<james.tomlinson@manchester.ac.uk>)
- Joshua Arnott (<josh@snorfalorpagus.net>)
- Lauren Petch

================
Acknowledgements
================

This work was funded by the University of Manchester, United Kingdom.

==========
References
==========
Wilby, R., Greenfield, B., Glenny, C., 1994. A coupled synoptic-hydrological model for climate change impact assessment. Journal of Hydrology. 153. p265-290.
