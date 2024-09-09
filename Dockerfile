FROM quay.io/pypa/manylinux2014_x86_64

ARG PYTHON_VERSION

RUN /opt/python/cp310-cp310/bin/pip install -U pip setuptools wheel Cython
RUN /opt/python/cp311-cp311/bin/pip install -U pip setuptools wheel Cython
RUN /opt/python/cp312-cp312/bin/pip install -U pip setuptools wheel Cython

COPY build_wheel.sh /build_wheel.sh
RUN chmod +x /build_wheel.sh
