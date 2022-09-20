.. module:: glearn

|project| Documentation
***********************

|deploy-docs|

|project| is a modular and high-performance Python package for machine learning using **G**\ aussian process regression with novel algorithms and is capable of petascale computation on multi-GPU devices.

.. .. toctree::
    :maxdepth: 1

    old/ComputeLogDeterminant.rst
    old/ComputeTraceOfInverse.rst
    old/examples.rst
    old/generate_matrix.rst
    old/InterpolateTraceOfInverse.rst
    old/introduction.rst

.. grid:: 4

    .. grid-item-card:: GitHub
        :link: https://github.com/ameli/glearn
        :text-align: center
        :class-card: custom-card-link

    .. grid-item-card:: PyPI
        :link: https://pypi.org/project/glearn/
        :text-align: center
        :class-card: custom-card-link

    .. grid-item-card:: Anaconda Cloud
        :link: https://anaconda.org/s-ameli/glearn
        :text-align: center
        :class-card: custom-card-link

    .. grid-item-card:: Docker Hub
        :link: https://hub.docker.com/r/sameli/glearn
        :text-align: center
        :class-card: custom-card-link

.. grid:: 4

    .. grid-item-card:: Install
        :link: install
        :link-type: ref
        :text-align: center
        :class-card: custom-card-link

    .. grid-item-card:: Tutorials
        :link: index_tutorials
        :link-type: ref
        :text-align: center
        :class-card: custom-card-link

    .. grid-item-card:: API reference
        :link: api
        :link-type: ref
        :text-align: center
        :class-card: custom-card-link

    .. grid-item-card:: Performance
        :link: index_performance
        :link-type: ref
        :text-align: center
        :class-card: custom-card-link

Overview
========

To learn more about |project| functionality, see:

.. toctree::

    overview

Supported Platforms
===================

Successful installation and tests performed on the following operating systems, architectures, and Python versions:

.. |y| unicode:: U+2714
.. |n| unicode:: U+2716

+----------+--------+--------+-------+-------+-------+-------+-------+-----------------+
| Platform | Arch   | Device | Python Version                        | Continuous      |
+          |        +        +-------+-------+-------+-------+-------+ Integration     +
|          |        |        |  3.6  |  3.7  |  3.8  |  3.9  |  3.10 |                 |
+==========+========+========+=======+=======+=======+=======+=======+=================+
| Linux    | X86-64 | CPU    |  |y|  |  |y|  |  |y|  |  |y|  |  |y|  | |build-linux|   |
+          +        +--------+-------+-------+-------+-------+-------+                 +
|          |        | GPU    |  |y|  |  |y|  |  |y|  |  |y|  |  |y|  |                 |
+----------+--------+--------+-------+-------+-------+-------+-------+-----------------+
| macOS    | X86-64 | CPU    |  |y|  |  |y|  |  |y|  |  |y|  |  |y|  | |build-macos|   |
+          +        +--------+-------+-------+-------+-------+-------+                 +
|          |        | GPU    |  |n|  |  |n|  |  |n|  |  |n|  |  |n|  |                 |
+----------+--------+--------+-------+-------+-------+-------+-------+-----------------+
| Windows  | X86-64 | CPU    |  |y|  |  |y|  |  |y|  |  |y|  |  |y|  | |build-windows| |
+          +        +--------+-------+-------+-------+-------+-------+                 +
|          |        | GPU    |  |y|  |  |y|  |  |y|  |  |y|  |  |y|  |                 |
+----------+--------+--------+-------+-------+-------+-------+-------+-----------------+

.. |build-linux| image:: https://img.shields.io/github/workflow/status/ameli/glearn/build-linux
   :target: https://github.com/ameli/glearn/actions?query=workflow%3Abuild-linux 
.. |build-macos| image:: https://img.shields.io/github/workflow/status/ameli/glearn/build-macos
   :target: https://github.com/ameli/glearn/actions?query=workflow%3Abuild-macos
.. |build-windows| image:: https://img.shields.io/github/workflow/status/ameli/glearn/build-windows
   :target: https://github.com/ameli/glearn/actions?query=workflow%3Abuild-windows

Python wheels for |project| for all supported platforms and versions in the above are available through `PyPI <https://pypi.org/project/glearn/>`_ and `Anaconda Cloud <https://anaconda.org/s-ameli/glearn>`_. If you need |project| on other platforms, architectures, and Python versions, `raise an issue <https://github.com/ameli/glearn/issues>`_ on GitHub and we build its Python Wheel for you.

Install
=======

|conda-downloads|

.. grid:: 2

    .. grid-item-card:: 

        Install with ``pip`` from `PyPI <https://pypi.org/project/glearn/>`_:

        .. prompt:: bash
            
            pip install glearn

    .. grid-item-card::

        Install with ``conda`` from `Anaconda Cloud <https://anaconda.org/s-ameli/glearn>`_:

        .. prompt:: bash
            
            conda install -c s-ameli glearn

For complete installation guide, see:

.. toctree::
    :maxdepth: 2
    :numbered:

    Install <tutorials/install>

Docker
======

|docker-pull| |deploy-docker|

The docker image comes with a pre-installed |project|, an NVIDIA graphic driver, and a compatible version of CUDA Toolkit libraries.

.. grid:: 1

    .. grid-item-card::

        Pull docker image from `Docker Hub <https://hub.docker.com/r/sameli/glearn>`_:

        .. prompt:: bash
            
            docker pull sameli/glearn

For a complete guide, see:

.. toctree::
    :maxdepth: 2
    :numbered:

    Docker <tutorials/docker>

GPU
===

|project| can run on CUDA-capable **multi**-GPU devices, which can be set up in several ways. Using the **docker container** is the easiest way to run |project| on GPU devices. For a comprehensive guide, see:

.. toctree::
    :maxdepth: 2
    :name: Using GPU
    :numbered:

    GPU <tutorials/gpu>

The supported GPU micro-architectures and CUDA version are as follows:

+-----------------+---------+---------+---------+---------+---------+---------+---------+--------+
| Version \\ Arch | Fermi   | Kepler  | Maxwell | Pascal  | Volta   | Turing  | Ampere  | Hopper |
+=================+=========+=========+=========+=========+=========+=========+=========+========+
| CUDA 9          |   |n|   |   |n|   |   |n|   |   |n|   |   |n|   |   |n|   |   |n|   |   |n|  |
+-----------------+---------+---------+---------+---------+---------+---------+---------+--------+
| CUDA 10         |   |n|   |   |y|   |   |y|   |   |y|   |   |y|   |   |y|   |   |y|   |   |y|  |
+-----------------+---------+---------+---------+---------+---------+---------+---------+--------+
| CUDA 11         |   |n|   |   |n|   |   |n|   |   |y|   |   |y|   |   |y|   |   |y|   |   |y|  |
+-----------------+---------+---------+---------+---------+---------+---------+---------+--------+

.. _index_tutorials:

Tutorials
=========

|binder|

Launch online interactive demonstration of examples with

API Reference
=============

Check the list of functions, classes, and modules of |project| with their usage, options, and examples.

.. toctree::
   :maxdepth: 2
   
   API Reference <api>

.. _index_performance:

Performance
===========

|project| is scalable to **very large matrices**. Its core library for basic linear algebraic operations is **faster than OpenBLAS**, and its **pseudo-random generator** is a hundred-fold faster than the implementation in the standard C++ library.

Read about the performance of |project| in practical applications:

.. toctree::
    :maxdepth: 1
    :hidden:

    performance <performance/performance>

.. Performance on GPU Farm <performance/gpu>
.. Performance on CPU <performance/scalability>
.. Comparison With and Without OpenBLAS <performance/openblas>
.. Interpolation of Affine Matrix Functions <performance/interpolation>


.. grid:: 2

   .. grid-item-card:: :ref:`Performance on GPU Farm <perf-gpu>`
        :link: perf-gpu
        :link-type: ref
        :text-align: center
        :class-card: custom-card-link-2

        .. image:: _static/images/performance/benchmark_speed_time.png
           :align: center
           :width: 320px
           :height: 200px
           :class: custom-dark

   .. grid-item-card:: :ref:`Comparison of Randomized Algorithms <perf-algorithms>`
        :link: perf-algorithms
        :link-type: ref
        :text-align: center
        :class-card: custom-card-link-2

        .. image:: _static/images/performance/compare_methods_practical_matrix_logdet_time.png
           :align: center
           :width: 320px
           :height: 200px
           :class: custom-dark

.. grid:: 2

   .. grid-item-card:: :ref:`Comparison With and Without OpenBLAS <perf-openblas>` 
        :link: perf-openblas
        :link-type: ref
        :text-align: center
        :class-card: custom-card-link-2

        .. image:: _static/images/performance/benchmark_openblas_sparse_time.png
           :align: center
           :width: 320px
           :height: 200px
           :class: custom-dark


   .. grid-item-card:: :ref:`Interpolation of Affine Matrix Functions <interpolation>` 
        :link: interpolation
        :link-type: ref
        :text-align: center
        :class-card: custom-card-link-2

        .. image:: _static/images/performance/affine_matrix_function_logdet.png
           :align: center
           :width: 320px
           :height: 200px
           :class: custom-dark

Features
========

* Matrices can be dense or sparse (`CSR` or `CSC` format), with 32-bit, 64-bit, or 128-bit data types, and stored either by row-ordering (`C` style) or column-ordering (`Fortran` style).
* Matrices can be **linear operators** with parameters (see :class:`imate.Matrix` and :class:`imate.AffineMatrixFunction` classes).
* **Randomized algorithms** using Hutchinson and stochastic Lanczos quadrature algorithms (see :ref:`Overview <overview>`)
* Novel method to **interpolate** matrix functions. See :ref:`Interpolation of Affine Matrix Functions <interpolation>`.
* Parallel processing both on **shared memory** and CUDA Capable **multi-GPU** devices.

* GPU, parallel
* Sparse covariance
* Mixed covariance model, object
* Jacobian and Hessian based optimization
* Learn hyperparameters in reduced space (profile likelihood)
* Prediction in Dual, O(n)

Technical Notes
===============

|tokei-2| |languages|

The core of |project|, which is implemented in C++ and NVIDIA CUDA framework, is a standalone modular library for high-performance low-level algebraic operations on linear operators (including matrices and affine matrix functions). This library provides a unified interface for computations on both CPU and GPU, a unified interface for dense and sparse matrices, a unified container for various data types, and fully automatic memory management and data transfer between CPU and GPU devices on demand. This library can be employed independently for projects other than |project|. The Doxygen generated reference of `C++/CUDA Classes and Namespaces <doxygen/html/annotated.html>`_ of |project| is available for developers.

The front-end interface of |project| is implemented in Cython and Python (see Python :ref:`API Reference <api>` for end-users).

Some notable implementation techniques used to develop |project| are:

* Polymorphic and curiously recurring template pattern programming (CRTP) technique.
* OS-independent customized `dynamic loading` of CUDA libraries.
* Static dispatching enables executing |project| with and without CUDA on the user's machine with the same pre-compiled |project| installation.
* Completely `GIL <https://en.wikipedia.org/wiki/Global_interpreter_lock>`_-*free* Cython implementation.
* Providing `manylinux wheels <https://pypi.org/project/imate/#files>`_ build upon customized docker images with CUDA support (see `manylinux CUDA 10 <https://hub.docker.com/r/sameli/manylinux2014_x86_64_cuda_10>`_ and `manylinux CUDA 11 <https://hub.docker.com/r/sameli/manylinux2014_x86_64_cuda_11>`_ docker images on Docker Hub).

How to Contribute
=================

We welcome contributions via `GitHub's pull request <https://github.com/ameli/glearn/pulls>`_. If you do not feel comfortable modifying the code, we also welcome feature requests and bug reports as `GitHub issues <https://github.com/ameli/glearn/issues>`_.

Publications
============

For information on how to cite |project|, publications, and software packages that used |project|, see:

.. toctree::
    :maxdepth: 2

    Publications <cite>

License
=======

|license|

This project uses a `BSD 3-clause license <https://github.com/ameli/glearn/blob/main/LICENSE.txt>`_, in hopes that it will be accessible to most projects. If you require a different license, please raise an `issue <https://github.com/ameli/glearn/issues>`_ and we will consider a dual license.

Related Projects
================

.. grid:: 3

   .. grid-item-card:: |imate-light| |imate-dark|
       :link: https://ameli.github.io/imate/index.html
       :text-align: center
       :class-card: custom-card-link
   
       A high-performance python package for machine learning using Gaussian process regression.

   .. grid-item-card:: |detkit-light| |detkit-dark|
       :link: https://ameli.github.io/detkit/index.html
       :text-align: center
       :class-card: custom-card-link

       A python package for matrix determinant functions used in machine learning.

   .. grid-item-card:: |special-light| |ortho-dark|
      :link: https://ameli.github.io/ortho/index.html
      :text-align: center
      :class-card: custom-card-link

      A python package providing both Python and Cython interface for special mathematical functions.

.. |deploy-docs| image:: https://img.shields.io/github/workflow/status/ameli/glearn/deploy-docs?label=docs
   :target: https://github.com/ameli/glearn/actions?query=workflow%3Adeploy-docs
.. |deploy-docker| image:: https://img.shields.io/github/workflow/status/ameli/glearn/deploy-docker?label=build%20docker
   :target: https://github.com/ameli/glearn/actions?query=workflow%3Adeploy-docker
.. |codecov-devel| image:: https://img.shields.io/codecov/c/github/ameli/glearn
   :target: https://codecov.io/gh/ameli/glearn
.. |license| image:: https://img.shields.io/github/license/ameli/glearn
   :target: https://opensource.org/licenses/BSD-3-Clause
.. |implementation| image:: https://img.shields.io/pypi/implementation/glearn
.. |pyversions| image:: https://img.shields.io/pypi/pyversions/glearn
.. |format| image:: https://img.shields.io/pypi/format/glearn
.. |pypi| image:: https://img.shields.io/pypi/v/glearn
.. |conda| image:: https://anaconda.org/s-ameli/traceinv/badges/installer/conda.svg
   :target: https://anaconda.org/s-ameli/traceinv
.. |platforms| image:: https://img.shields.io/conda/pn/s-ameli/traceinv?color=orange?label=platforms
   :target: https://anaconda.org/s-ameli/traceinv
.. |conda-version| image:: https://img.shields.io/conda/v/s-ameli/traceinv
   :target: https://anaconda.org/s-ameli/traceinv
.. |binder| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/ameli/glearn/HEAD?filepath=notebooks%2FInterpolateTraceOfInverse.ipynb
.. |conda-downloads| image:: https://img.shields.io/conda/dn/s-ameli/glearn
   :target: https://anaconda.org/s-ameli/glearn
.. |tokei| image:: https://tokei.rs/b1/github/ameli/glearn?category=lines
   :target: https://github.com/ameli/glearn
.. |tokei-2| image:: https://img.shields.io/badge/code%20lines-26.4k-blue
   :target: https://github.com/ameli/glearn
.. |languages| image:: https://img.shields.io/github/languages/count/ameli/glearn
   :target: https://github.com/ameli/glearn
.. |docker-pull| image:: https://img.shields.io/docker/pulls/sameli/glearn?color=green&label=downloads
   :target: https://hub.docker.com/r/sameli/glearn
.. |imate-light| image:: _static/images/icons/logo-imate-light.svg
   :height: 23
   :class: only-light
.. |imate-dark| image:: _static/images/icons/logo-imate-dark.svg
   :height: 23
   :class: only-dark
.. |detkit-light| image:: _static/images/icons/logo-detkit-light.svg
   :height: 27
   :class: only-light
.. |detkit-dark| image:: _static/images/icons/logo-detkit-dark.svg
   :height: 27
   :class: only-dark
.. |special-light| image:: _static/images/icons/logo-special-light.svg
   :height: 24
   :class: only-light
.. |ortho-dark| image:: _static/images/icons/logo-ortho-dark.svg
   :height: 24
   :class: only-dark
.. |special_functions-light| image:: _static/images/icons/logo-special-functions-light.svg
   :height: 35
   :class: only-light
.. |special_functions-dark| image:: _static/images/icons/logo-special-functions-dark.svg
   :height: 35
   :class: only-dark
