# SPDX-FileCopyrightText: Copyright 2022, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE

# -----------------------------------------------------------------------------
# How to build
#
#   $ cd ..  # go to root dir of package (it must run from the parent dir)
#   $ docker build . -t sameli/glearn -f docker/Dockerfile
#
# If you run `docker build` multiple times to fix issues, it is better to add
# `--no-cache` option, so that in each run it starts from fresh:
#
#   $ docker build . --no-cache -t sameli/glearn -f docker/Dockerfile
#
# How to run:
#
#   $ docker run -it -v/host_dir:/image_dir glearn
#
# Notes:
#
# The `ADD` command copies the contents of the package to a directory in the
# comtainer, but it does not copy all files/folders. See `.dockerignore` to
# see which files are ingored.
# -----------------------------------------------------------------------------

# -----
# Build
# -----

FROM nvidia/cuda:12.3.1-devel-ubuntu22.04 as builder
MAINTAINER Siavash Ameli <samei@berkeley.edu>
LABEL Description="glearn python package with cuda-12 support"

# Install python3
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-dev \
        python3-pip \
        python3-venv \
        wget \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Build package wheel
RUN mkdir /tmp/glearn

# Note: ADD does not copy all files. See .dockerignore
ADD . /tmp/glearn/
WORKDIR /tmp/glearn
RUN rm -rf build dist

# Note: run setup.py clean to remove in-source cpp build files (if there are any)
RUN /usr/bin/python3 setup.py clean
RUN /usr/bin/python3 -m pip install --upgrade pip
RUN /usr/bin/python3 -m pip install build
RUN /usr/bin/python3 -m build --wheel --outdir dist

# -------
# Runtime
# -------

FROM sameli/imate

# Install python3 and libcudablas, libcusparse, since they are not in the NVIDIA
# cuda's "base" image.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
        libcublas-12-3 \
        libcusparse-12-3 \
        python3-pip \
        libgomp1 \
        libsuitesparse-dev \
        vim \
        texlive-latex-recommended \
        cm-super \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the wheel from the previous build (on NVIDIA cuda-devel image) and
# install it on this image.
COPY --from=builder /tmp/glearn/dist/*.whl /tmp/
RUN /usr/bin/python3 -m pip --no-cache-dir install /tmp/*.whl
RUN rm /tmp/*.whl

# Create a startup file to auto-import package when python starts
RUN echo "import glearn" > /home/.start_python.py
ENV PYTHONSTARTUP=/home/.start_python.py
ENV PATH=$PATH:/usr/bin/
RUN /usr/bin/python3 -m pip install --upgrade pip ipython jupyter

# Change working directory for entrypoint
VOLUME ["/root"]
WORKDIR /root

# ENTRYPOINT ["/bin/bash"]
CMD ["/usr/bin/python3"]
