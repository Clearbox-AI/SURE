FROM quay.io/pypa/manylinux2014_x86_64

ARG PYTHON_VERSION

RUN /opt/python/cp${PYTHON_VERSION}/bin/pip install -U pip setuptools wheel Cython

COPY build_wheels.sh /build_wheel.sh
RUN chmod +x /build_wheel.sh
