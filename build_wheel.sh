#!/bin/bash

cd /io/

/opt/python/cp39-cp39/bin/pip install -U pip setuptools wheel Cython
/opt/python/cp39-cp39/bin/pip install -r requirements.txt
/opt/python/cp39-cp39/bin/python setup.py bdist_wheel

/opt/python/cp310-cp310/bin/pip install -U pip setuptools wheel Cython
/opt/python/cp310-cp310/bin/pip install -r requirements.txt
/opt/python/cp310-cp310/bin/python setup.py bdist_wheel

/opt/python/cp311-cp311/bin/pip install -U pip setuptools wheel Cython
/opt/python/cp311-cp311/bin/pip install -r requirements.txt
/opt/python/cp311-cp311/bin/python setup.py bdist_wheel

/opt/python/cp312-cp312/bin/pip install -U pip setuptools wheel Cython
/opt/python/cp312-cp312/bin/pip install -r requirements.txt
/opt/python/cp312-cp312/bin/python setup.py bdist_wheel

# Repair the wheels with auditwheel
for whl in dist/*.whl; do
    auditwheel repair "$whl" --plat manylinux2014_x86_64 -w /io/wheelhouse/
done