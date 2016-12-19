#!/bin/bash

# clean up before building
# assumes all .c files were compiled from .pyx and should be removed
rm -rf build dist
find . | grep -E "(__pycache__|\.pyc|\.pyo|\.pyd|\.so|\.c$)" | xargs rm -rf

$PYTHON setup.py install
