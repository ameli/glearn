.. _optional-dependencies:

Optional Runtime Dependencies
=============================

The followings are dependencies used during the runtime of |project|. Note that, among these dependencies, `OpenMP` is **required**, while the rest of the dependencies are optional.

.. _dependencies_openmp:

OpenMP (`Required`)
-------------------

|project| requires OpenMP, which is typically included with most C++ compilers.

.. glossary::

    For **Linux** users:

        By installing a C++ compiler such as GCC, Clang, or Intel, you also obtain OpenMP as well. You may alternatively install ``libgomp`` (see below) without the need to install a full compiler.

    For **macOS** users:

        It's crucial to note that OpenMP is not part of the default Apple Xcode's LLVM compiler. Even if you have Apple Xcode LLVM compiler readily installed on macOS, you will still need to install OpenMP separately via ``libomp`` Homebrew package (see below) or as part of the *open source* `LLVM compiler <https://llvm.org/>`__, via ``llvm`` Homebrew package.

    For **Windows** users:

        OpenMP support depends on the compiler you choose; Microsoft Visual C++ supports OpenMP, but you may need to enable it explicitly.

Below are the specific installation for each operating system:

.. tab-set::

    .. tab-item:: Ubuntu/Debian
        :sync: ubuntu

        .. prompt:: bash

            sudo apt install libgomp1 -y

    .. tab-item:: CentOS 7
        :sync: centos

        .. prompt:: bash

            sudo yum install libgomp -y

    .. tab-item:: RHEL 9
        :sync: rhel

        .. prompt:: bash

            sudo dnf install libgomp -y

    .. tab-item:: macOS
        :sync: osx

        .. prompt:: bash

            sudo brew install libomp

.. note::

    In *macOS*, for ``libomp`` versions ``15`` and above, Homebrew installs OpenMP as *keg-only*. To utilize the OpenMP installation, you should establish the following symbolic links:

    .. prompt:: bash

        libomp_dir=$(brew --prefix libomp)
        ln -sf ${libomp_dir}/include/omp-tools.h  /usr/local/include/omp-tools.h
        ln -sf ${libomp_dir}/include/omp.h        /usr/local/include/omp.h
        ln -sf ${libomp_dir}/include/ompt.h       /usr/local/include/ompt.h
        ln -sf ${libomp_dir}/lib/libomp.a         /usr/local/lib/libomp.a
        ln -sf ${libomp_dir}/lib/libomp.dylib     /usr/local/lib/libomp.dylib

CUDA Toolkit and NVIDIA Graphic Driver (`Optional`)
---------------------------------------------------

To use GPU devices, install CUDA runtime libraries and NVIDIA graphic driver. See the instructions below.

* :ref:`Install CUDA runtime libraries <install-cuda-runtime-lib>`.
* :ref:`Install NVIDIA Graphic Driver <install-graphic-driver>`.

Sparse Suite (`Optional`)
-------------------------

`Suite Sarse <https://people.engr.tamu.edu/davis/suitesparse.html>`_ is a library for efficient calculations on sparse matrices. |project| does not require this library as it has its own library for sparse matrices. However, if this library is available, |project| uses it.

.. note::

    The Sparse Suite library is only used for those functions in |project| that uses the Cholesky decomposition method by passing ``method=cholesky`` argument to the functions. See :ref:`API reference for Functions <Functions>` for details. 

1. Install Sparse Suite development library by

   .. tab-set::

       .. tab-item:: Ubuntu/Debian
          :sync: ubuntu

          .. prompt:: bash

              sudo apt install libsuitesparse-dev

       .. tab-item:: CentOS 7
          :sync: centos

          .. prompt:: bash

              sudo yum install libsuitesparse-devel

       .. tab-item:: RHEL 9
          :sync: rhel

          .. prompt:: bash

              sudo dnf install libsuitesparse-devel

       .. tab-item:: macOS
          :sync: osx

          .. prompt:: bash

              sudo brew install suite-sparse

   Alternatively, if you are using *Anaconda* python distribution (on either of the operating systems), install Suite Sparse by:

   .. prompt:: bash

       sudo conda install -c conda-forge suitesparse

2. Install ``scikit-sparse`` python package:

   .. prompt:: bash
       
       python -m pip install scikit-sparse
