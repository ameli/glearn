******
|logo|
******

``glearn`` is a modular and high-performance Python package for machine learning using **G**\ aussian process regression with novel algorithms capable of petascale computation on multi-GPU devices.

Links
=====

* `Documentation <https://ameli.github.io/glearn>`__
* `PyPI <https://pypi.org/project/glearn/>`__
* `Anaconda <https://anaconda.org/s-ameli/glearn>`__
* `Docker Hub <https://hub.docker.com/r/sameli/glearn>`__
* `Github <https://github.com/ameli/glearn>`__

Install
=======

Install with ``pip``
--------------------

|pypi|

::

    pip install glearn

Install with ``conda``
----------------------

|conda-version|

::

    conda install -c s-ameli glearn

Docker Image
------------

|docker-pull| |deploy-docker|

::

    docker pull sameli/glearn

Supported Platforms
===================

Successful installation and tests performed on the following operating systems, architectures, and Python and `PyPy <https://www.pypy.org/>`__ versions:

.. |y| unicode:: U+2714
.. |n| unicode:: U+2716

+----------+-------------------+--------+-------+-------+-------+-------+-------+-------+-------+-----------------+
| Platform | Arch              | Device | Python Version                | PyPy Version :sup:`1` | Continuous      |
+          |                   +        +-------+-------+-------+-------+-------+-------+-------+ Integration     +
|          |                   |        |  3.9  |  3.10 |  3.11 |  3.12 |  3.8  |  3.9  |  3.10 |                 |
+==========+===================+========+=======+=======+=======+=======+=======+=======+=======+=================+
| Linux    | X86-64            | CPU    |  |y|  |  |y|  |  |y|  |  |y|  |  |y|  |  |y|  |  |y|  | |build-linux|   |
+          +                   +--------+-------+-------+-------+-------+-------+-------+-------+                 +
|          |                   | GPU    |  |y|  |  |y|  |  |y|  |  |y|  |  |y|  |  |y|  |  |y|  |                 |
+          +-------------------+--------+-------+-------+-------+-------+-------+-------+-------+                 +
|          | AARCH-64          | CPU    |  |y|  |  |y|  |  |y|  |  |y|  |  |y|  |  |y|  |  |y|  |                 |
+          +                   +--------+-------+-------+-------+-------+-------+-------+-------+                 +
|          |                   | GPU    |  |y|  |  |y|  |  |y|  |  |y|  |  |y|  |  |y|  |  |y|  |                 |
+----------+-------------------+--------+-------+-------+-------+-------+-------+-------+-------+-----------------+
| macOS    | X86-64            | CPU    |  |y|  |  |y|  |  |y|  |  |y|  |  |y|  |  |y|  |  |y|  | |build-macos|   |
+          +                   +--------+-------+-------+-------+-------+-------+-------+-------+                 +
|          |                   | GPU    |  |n|  |  |n|  |  |n|  |  |n|  |  |n|  |  |n|  |  |n|  |                 |
+          +-------------------+--------+-------+-------+-------+-------+-------+-------+-------+                 +
|          | ARM-64            | CPU    |  |y|  |  |y|  |  |y|  |  |y|  |  |y|  |  |y|  |  |y|  |                 |
+          +                   +--------+-------+-------+-------+-------+-------+-------+-------+                 +
|          |                   | GPU    |  |n|  |  |n|  |  |n|  |  |n|  |  |n|  |  |n|  |  |n|  |                 |
+----------+-------------------+--------+-------+-------+-------+-------+-------+-------+-------+-----------------+
| Windows  | X86-64            | CPU    |  |y|  |  |y|  |  |y|  |  |y|  |  |n|  |  |n|  |  |n|  | |build-windows| |
+          +                   +--------+-------+-------+-------+-------+-------+-------+-------+                 +
|          |                   | GPU    |  |y|  |  |y|  |  |y|  |  |y|  |  |n|  |  |n|  |  |n|  |                 |
+----------+-------------------+--------+-------+-------+-------+-------+-------+-------+-------+-----------------+

.. |build-linux| image:: https://img.shields.io/github/actions/workflow/status/ameli/glearn/build-linux.yml
   :target: https://github.com/ameli/glearn/actions?query=workflow%3Abuild-linux 
.. |build-macos| image:: https://img.shields.io/github/actions/workflow/status/ameli/glearn/build-macos.yml
   :target: https://github.com/ameli/glearn/actions?query=workflow%3Abuild-macos
.. |build-windows| image:: https://img.shields.io/github/actions/workflow/status/ameli/glearn/build-windows.yml
   :target: https://github.com/ameli/glearn/actions?query=workflow%3Abuild-windows

Python wheels for ``glearn`` for all supported platforms and versions in the above are available through `PyPI <https://pypi.org/project/glearn/>`__ and `Anaconda Cloud <https://anaconda.org/s-ameli/glearn>`__. If you need ``glearn`` on other platforms, architectures, and Python or PyPy versions, `raise an issue <https://github.com/ameli/glearn/issues>`__ on GitHub and we build its Python Wheel for you.

.. line-block::

    :sup:`1. Wheels for PyPy are exclusively available for installation through pip and cannot be installed using conda.`
    :sup:`2. Wheels for Windows on ARM-64 architecture are exclusively available for installation through pip and cannot be installed using conda.`

Supported GPU Architectures
===========================

``glearn`` can run on CUDA-capable **multi**-GPU devices. Using the **docker container** is the easiest way to run ``glearn`` on GPU devices. The supported GPU micro-architectures and CUDA version are as follows:

+-----------------+---------+---------+---------+---------+---------+---------+---------+--------+
| Version \\ Arch | Fermi   | Kepler  | Maxwell | Pascal  | Volta   | Turing  | Ampere  | Hopper |
+=================+=========+=========+=========+=========+=========+=========+=========+========+
| CUDA 9          |   |n|   |   |n|   |   |n|   |   |n|   |   |n|   |   |n|   |   |n|   |   |n|  |
+-----------------+---------+---------+---------+---------+---------+---------+---------+--------+
| CUDA 10         |   |n|   |   |y|   |   |y|   |   |y|   |   |y|   |   |y|   |   |y|   |   |y|  |
+-----------------+---------+---------+---------+---------+---------+---------+---------+--------+
| CUDA 11         |   |n|   |   |n|   |   |n|   |   |y|   |   |y|   |   |y|   |   |y|   |   |y|  |
+-----------------+---------+---------+---------+---------+---------+---------+---------+--------+
| CUDA 12         |   |n|   |   |n|   |   |n|   |   |y|   |   |y|   |   |y|   |   |y|   |   |y|  |
+-----------------+---------+---------+---------+---------+---------+---------+---------+--------+

Documentation
=============

|deploy-docs| |binder|

See `documentation <https://ameli.github.io/glearn/index.html>`__, including:

* `What This Packages Does? <https://ameli.github.io/glearn/overview.html>`__
* `Comprehensive Installation Guide <https://ameli.github.io/glearn/tutorials/install.html>`__
* `How to Work with Docker Container? <https://ameli.github.io/glearn/tutorials/docker.html>`__
* `How to Deploy on GPU Devices? <https://ameli.github.io/glearn/tutorials/gpu.html>`__
* `API Reference <https://ameli.github.io/glearn/api.html>`__
* `Interactive Notebook Tutorials <https://mybinder.org/v2/gh/ameli/glearn/HEAD?filepath=notebooks%2Fquick_start.ipynb>`__
* `Publications <https://ameli.github.io/glearn/cite.html>`__

How to Contribute
=================

We welcome contributions via `GitHub's pull request <https://github.com/ameli/glearn/pulls>`__. If you do not feel comfortable modifying the code, we also welcome feature requests and bug reports as `GitHub issues <https://github.com/ameli/glearn/issues>`__.

How to Cite
===========

If you publish work that uses ``glearn``, please consider citing the manuscripts available `here <https://ameli.github.io/glearn/cite.html>`__.

License
=======

|license|

This project uses a `BSD 3-clause license <https://github.com/ameli/glearn/blob/main/LICENSE.txt>`__, in hopes that it will be accessible to most projects. If you require a different license, please raise an `issue <https://github.com/ameli/glearn/issues>`__ and we will consider a dual license.

.. |logo| image:: https://raw.githubusercontent.com/ameli/glearn/main/docs/source/_static/images/icons/logo-glearn-light.svg
   :width: 160
.. |license| image:: https://img.shields.io/github/license/ameli/glearn
   :target: https://opensource.org/licenses/BSD-3-Clause
.. |deploy-docs| image:: https://img.shields.io/github/actions/workflow/status/ameli/glearn/deploy-docs.yml?label=docs
   :target: https://github.com/ameli/glearn/actions?query=workflow%3Adeploy-docs
.. |binder| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/ameli/glearn/HEAD?filepath=notebooks%2Fquick_start.ipynb
.. |pypi| image:: https://img.shields.io/pypi/v/glearn
   :target: https://pypi.org/project/glearn/
.. |codecov-devel| image:: https://img.shields.io/codecov/c/github/ameli/glearn
   :target: https://codecov.io/gh/ameli/glearn
.. |deploy-docker| image:: https://img.shields.io/github/actions/workflow/status/ameli/glearn/deploy-docker.yml?label=build%20docker
   :target: https://github.com/ameli/glearn/actions?query=workflow%3Adeploy-docker
.. |docker-pull| image:: https://img.shields.io/docker/pulls/sameli/glearn?color=green&label=downloads
   :target: https://hub.docker.com/r/sameli/glearn
.. |conda-version| image:: https://img.shields.io/conda/v/s-ameli/glearn
   :target: https://anaconda.org/s-ameli/glearn
