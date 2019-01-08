#!/bin/bash
# Build script for use with manylinux1 docker image to construct Linux wheels
# Note this script does not build the wheels with lpsolve support.
set -e -x

export PYCATCHMOD_TAG=v1.1.0
export WORKDIR=${HOME}

cd ${WORKDIR}

rm -rf pycatchmod
git clone https://github.com/pywr/pycatchmod.git
cd pycatchmod
git checkout ${PYCATCHMOD_TAG}


# Support binaries
PYBINS=( /opt/python/cp36-cp36m/bin )

# Compile wheels
for PYBIN in "${PYBINS[@]}"; do
    "${PYBIN}/pip" install -r /io/dev-requirements.txt --no-build-isolation --only-binary pandas
    "${PYBIN}/pip" wheel --no-deps ./ -w ${WORKDIR}/wheelhouse/
done

# Bundle external shared libraries into the wheels
for whl in ${WORKDIR}/wheelhouse/pycatchmod*.whl; do
    auditwheel repair "$whl" -w /io/wheelhouse/
done

# Install packages and test
for PYBIN in "${PYBINS[@]}"; do
    "${PYBIN}/pip" install pycatchmod --no-index -f /io/wheelhouse
    "${PYBIN}/py.test" ${WORKDIR}/pycatchmod/tests
done
