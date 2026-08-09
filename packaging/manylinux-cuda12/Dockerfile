# syntax=docker/dockerfile:1.7

ARG MANYLINUX_IMAGE=quay.io/pypa/manylinux_2_28_x86_64@sha256:f854c50adf7b7a325bc4794316f3758d387a41d61f9e2ebca0f26c7dc8f761d4

FROM ${MANYLINUX_IMAGE} AS cuda-base

ARG CUDA_SERIES=12-8
ARG PYTHON_TAG=cp312-cp312
ENV PYTHON_TAG=${PYTHON_TAG}
RUN dnf install -y dnf-plugins-core \
    && dnf config-manager --add-repo \
       https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo \
    && dnf install -y \
       "cuda-cudart-devel-${CUDA_SERIES}" \
       "cuda-driver-devel-${CUDA_SERIES}" \
       "cuda-nvcc-${CUDA_SERIES}" \
       "libcufft-devel-${CUDA_SERIES}" \
       "libcurand-devel-${CUDA_SERIES}" \
    && dnf clean all

ENV CUDA_HOME=/usr/local/cuda-12.8
ENV CUDACXX=/usr/local/cuda-12.8/bin/nvcc
ENV PATH=/opt/python/${PYTHON_TAG}/bin:/usr/local/cuda-12.8/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/wheel-deps/lib64:/opt/wheel-deps/lib:/opt/loki/lib64:/opt/loki/lib:/usr/local/cuda-12.8/lib64

RUN for python_tag in cp39-cp39 cp310-cp310 cp311-cp311 cp312-cp312; do \
      /opt/python/${python_tag}/bin/python -m pip install --no-cache-dir \
        auditwheel build ninja pybind11 scikit-build-core wheel typing_extensions; \
    done

FROM cuda-base AS toolchain

COPY packaging/manylinux-cuda12/build-dependencies.sh /usr/local/bin/build-gaffa-dependencies
RUN /usr/local/bin/build-gaffa-dependencies

FROM toolchain AS loki-builder

COPY --from=loki / /src/loki
COPY packaging/manylinux-cuda12/build-loki.sh /usr/local/bin/build-loki
RUN --mount=type=cache,target=/opt/cpm-cache /usr/local/bin/build-loki

FROM toolchain AS builder

COPY --from=loki-builder /opt/loki /opt/loki
COPY --from=loki /LICENSE /src/loki/LICENSE
COPY . /src/gaffa
WORKDIR /src/gaffa
RUN packaging/manylinux-cuda12/build-wheel.sh

FROM scratch AS wheel
COPY --from=builder /wheelhouse/ /

FROM ${MANYLINUX_IMAGE} AS cuda-runtime-base

ARG CUDA_RUNTIME_SERIES=12-1
ARG PYTHON_TAG=cp312-cp312
ENV PYTHON_TAG=${PYTHON_TAG}
RUN curl --fail --location \
       https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo \
       --output /etc/yum.repos.d/cuda-rhel8.repo \
    && dnf install -y \
       "cuda-cudart-${CUDA_RUNTIME_SERIES}" \
       "libcufft-${CUDA_RUNTIME_SERIES}" \
       "libcurand-${CUDA_RUNTIME_SERIES}" \
    && dnf clean all

FROM cuda-runtime-base AS runtime-test

ENV PATH=/opt/python/${PYTHON_TAG}/bin:/usr/local/cuda-12.1/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64

COPY --from=builder /wheelhouse /wheelhouse
COPY packaging/manylinux-cuda12/runtime-smoke.py /opt/gaffa/runtime-smoke.py
COPY packaging/manylinux-cuda12/verify-wheel.py /opt/gaffa/verify-wheel.py
ARG RUNTIME_PIP_INDEX_URL=https://pypi.org/simple
RUN /opt/python/${PYTHON_TAG}/bin/python /opt/gaffa/verify-wheel.py \
      /wheelhouse/*.whl \
      --cuda-runtime-root /usr/local/cuda-12.1/targets/x86_64-linux/lib
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/python/${PYTHON_TAG}/bin/python -m pip install \
      --index-url "${RUNTIME_PIP_INDEX_URL}" \
      /wheelhouse/*.whl

ENTRYPOINT ["/bin/sh", "-c", "exec /opt/python/${PYTHON_TAG}/bin/python /opt/gaffa/runtime-smoke.py"]
