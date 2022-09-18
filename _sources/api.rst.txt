.. _api:

API Reference
*************

The API reference contains:

* :ref:`Functions <Functions>`: compute log-determinant and trace of functions
  of matrices.
* :ref:`Interpolators <Interpolators>`: interpolate functions of one-parameter
  family of affine matrix functions.
* :ref:`Linear Operators <Linear Operators>`: classes that represent matrices
  and affine matrix functions.
* :ref:`Sample Matrices <Sample Matrices>`: generate matrices for test
  purposes.
* :ref:`Device Inquiry <Device Inquiry>`: inquiry information about CPU and GPU devices.

.. _Functions:

Functions
=========

The functions of this package are:

* :ref:`Log-Determinant <Log-Determinant>`: computes log-determinant of matrix.
* :ref:`Trace of Inverses <Trace of Inverses>`: computes trace of the inverse of
  a matrix or any negative power of the matrix.
* :ref:`Trace <Trace>`: computes the trace of matrix or any positive power of
  the matrix.
* :ref:`Schatten Norm <Schatten Norm>`: computes the Schatten norm of order
  :math:`p`, which includes the above three functions. 

Each of the above functions are implemented using both direct and randomized algorithms, suitable for various matrices sizes.

.. _Log-Determinant:

Log-Determinant
---------------

.. autosummary::
    :toctree: generated
    :caption: logdet
    :recursive:
    :template: autosummary/member.rst

    imate.logdet

This function computes the log-determinant of :math:`\mathbf{A}^p` or the Gramian matrix :math:`(\mathbf{A}^{\intercal} \mathbf{A})^p` where :math:`p` is a real exponent.

The `imate.logdet` function has the following methods:

.. toctree::

    api/imate.logdet.eigenvalue
    api/imate.logdet.cholesky
    api/imate.logdet.slq

.. _Linear Operators:

Linear Operators
================

Create linear operator objects as container for various matrix types with a unified interface, establish a fully automatic dynamic buffer to allocate, deallocate, and transfer data between CPU and multiple GPU devices on demand, as well as perform basic matrix-vector operations with high performance on both CPU or GPU devices. These objects can be passed to |project| functions as input matrices.

.. autosummary::
    :toctree: generated
    :caption: Classes
    :recursive:
    :template: autosummary/class.rst

    glearn.LinearModel
    glearn.Covariance
    glearn.GaussianProcess

.. _Sample Matrices:

Sample Matrices
===============

Generate sample matrices for test purposes, such as correlation matrix and Toeplitz matrix. The matrix functions of Toeplitz matrix (such as its log-determinant, trace of its inverse, etc) are known analytically, making Toeplitz matrix suitable for benchmarking the result of randomized methods with analytical solutions.
   
.. autosummary::
    :toctree: generated
    :caption: Sample Matrices
    :recursive:
    :template: autosummary/member.rst

    imate.correlation_matrix
    imate.toeplitz
    imate.sample_matrices.toeplitz_logdet
    imate.sample_matrices.toeplitz_trace
    imate.sample_matrices.toeplitz_traceinv
    imate.sample_matrices.toeplitz_schatten

.. _Device Inquiry:

Device Inquiry
==============

Measure the process time and consumed memory of the Python process during computation with the following classes.

.. autosummary::
    :toctree: generated
    :caption: Device Inquiry
    :recursive:
    :template: autosummary/class.rst

    imate.Timer
    imate.Memory

Inquiry hardware information, including CPU and GPU devices employed during computation and get information about the CUDA Toolkit installation with the following functions.

.. autosummary::
    :toctree: generated
    :recursive:
    :template: autosummary/member.rst

    imate.info
    imate.device.get_processor_name
    imate.device.get_gpu_name
    imate.device.get_num_cpu_threads
    imate.device.get_num_gpu_devices
    imate.device.get_nvidia_driver_version
    imate.device.locate_cuda
    imate.device.restrict_to_single_processor
