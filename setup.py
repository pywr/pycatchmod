from setuptools import setup, Extension
from Cython.Distutils import build_ext
import numpy

catchmod = Extension('pycatchmod._catchmod', ['pycatchmod/_catchmod.pyx'],
                     include_dirs=[numpy.get_include()])
weather_generator = Extension('pycatchmod._weather_generator', ['pycatchmod/_weather_generator.pyx'],
                     include_dirs=[numpy.get_include()])

setup(name='pycatchmod',
      version='0.1',
      description='Python implementation of the rainfall runoff model CATCHMOD.',
      author='James E Tomlinson',
      author_email='tomo.bbe@gmail.com',
      packages=['pycatchmod'],
      install_requires=['cython', 'numpy'],
      setup_requires=['cython', 'numpy'],
      tests_require=['pytest'],
      ext_modules=[catchmod, weather_generator],
      cmdclass = {'build_ext': build_ext},
      )