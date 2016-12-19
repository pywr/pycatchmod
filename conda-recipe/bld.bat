# clean up before building
# assumes all .c files were compiled from .pyx and should be removed
rmdir /s /q build dist
del /s __pycache__ *.pyc *.pyo *.pyd *.so *.c

"%PYTHON%" setup.py install
if errorlevel 1 exit 1
