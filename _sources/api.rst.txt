.. _api:

API Reference
*************

The API reference contains:

* :ref:`Linear Model <linear_model>`: classes to create linear model, covariance, and Gaussian process objects.
* :ref:`Kernels <kernels>`: classes to create kernel functions for the covariance model.
* :ref:`Priors <priors>`: Classes to create prior distributions for hyperparameters of covariance model.
* :ref:`Sample Data <sample_data>`: Functions to generate sample points and data for test purposes.
* :ref:`Device Inquiry <device_inquiry>`: Functions and classes to inquiry hardware device information.

.. _linear_model:

Linear Model
============

Define a customized linear model by creating modular objects representing correlation, covariance, mean, and a Gaussian process prior.

.. autosummary::
    :toctree: generated
    :caption: Linear Model
    :recursive:
    :template: autosummary/class.rst

    glearn.LinearModel
    glearn.Covariance
    glearn.GaussianProcess

.. _kernels:

Kernels
=======

Defines various kernel functions for the covariance model.

.. autosummary::
    :toctree: generated
    :caption: Kernels
    :recursive:
    :template: autosummary/class.rst

    glearn.kernels.Kernel
    glearn.kernels.Exponential
    glearn.kernels.SquareExponential
    glearn.kernels.Linear
    glearn.kernels.RationalQuadratic
    glearn.kernels.Matern

.. _priors:

Prior Distributions
===================

Define various prior distributions for the hyperparameters of the covariance model.

.. autosummary::
    :toctree: generated
    :caption: Prior Distributions
    :recursive:
    :template: autosummary/class.rst

    glearn.priors.Prior
    glearn.priors.Uniform
    glearn.priors.Normal
    glearn.priors.Cauchy
    glearn.priors.StudentT
    glearn.priors.Erlang
    glearn.priors.Gamma
    glearn.priors.InverseGamma
    glearn.priors.BetaPrime

.. _sample_data:

Sample Data
===========

Generate sample data for test purposes, such as multi-dimensional points and stochastic data on the points.
   
.. autosummary::
    :toctree: generated
    :caption: Sample Data
    :recursive:
    :template: autosummary/member.rst

    glearn.sample_data.generate_points
    glearn.sample_data.generate_data

.. _device_inquiry:

Device Inquiry
==============

Measure the process time and consumed memory of the Python process during computation with the following classes.

.. autosummary::
    :toctree: generated
    :caption: Device Inquiry
    :recursive:
    :template: autosummary/class.rst

    glearn.Timer
    glearn.Memory

Inquiry hardware information, including CPU and GPU devices employed during computation and get information about the CUDA Toolkit installation with the following functions.

.. autosummary::
    :toctree: generated
    :recursive:
    :template: autosummary/member.rst

    glearn.info
    glearn.device.get_processor_name
    glearn.device.get_gpu_name
    glearn.device.get_num_cpu_threads
    glearn.device.get_num_gpu_devices
    glearn.device.get_nvidia_driver_version
    glearn.device.locate_cuda
    glearn.device.restrict_to_single_processor
