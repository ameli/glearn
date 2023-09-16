******
|logo|
******

``glearn`` is a modular and high-performance Python package for machine learning using **G**\ aussian process regression with novel algorithms capable of petascale computation on multi-GPU devices.

Links
=====

* `Documentation <https://ameli.github.io/glearn>`_
* `PyPI <https://pypi.org/project/glearn/>`_
* `Anaconda <https://anaconda.org/s-ameli/glearn>`_
* `Docker Hub <https://hub.docker.com/r/sameli/glearn>`_
* `Github <https://github.com/ameli/glearn>`_

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

Successful installation and tests performed on the following operating systems, architectures, and Python and `PyPy <https://www.pypy.org/>`_ versions:

.. |y| unicode:: U+2714
.. |n| unicode:: U+2716

+----------+--------+--------+-------+-------+-------+-----------------+
| Platform | Arch   | Device | Python Version        | Continuous      |
+          |        +        +-------+-------+-------+ Integration     +
|          |        |        |  3.9  |  3.10 |  3.11 |                 |
+==========+========+========+=======+=======+=======+=================+
| Linux    | X86-64 | CPU    |  |y|  |  |y|  |  |y|  | |build-linux|   |
+          +        +--------+-------+-------+-------+                 +
|          |        | GPU    |  |y|  |  |y|  |  |y|  |                 |
+----------+--------+--------+-------+-------+-------+-----------------+
| macOS    | X86-64 | CPU    |  |y|  |  |y|  |  |y|  | |build-macos|   |
+          +        +--------+-------+-------+-------+                 +
|          |        | GPU    |  |n|  |  |n|  |  |n|  |                 |
+----------+--------+--------+-------+-------+-------+-----------------+
| Windows  | X86-64 | CPU    |  |y|  |  |y|  |  |y|  | |build-windows| |
+          +        +--------+-------+-------+-------+                 +
|          |        | GPU    |  |y|  |  |y|  |  |y|  |                 |
+----------+--------+--------+-------+-------+-------+-----------------+

.. |build-linux| image:: https://img.shields.io/github/actions/workflow/status/ameli/glearn/build-linux.yml
   :target: https://github.com/ameli/glearn/actions?query=workflow%3Abuild-linux 
.. |build-macos| image:: https://img.shields.io/github/actions/workflow/status/ameli/glearn/build-macos.yml
   :target: https://github.com/ameli/glearn/actions?query=workflow%3Abuild-macos
.. |build-windows| image:: https://img.shields.io/github/actions/workflow/status/ameli/glearn/build-windows.yml
   :target: https://github.com/ameli/glearn/actions?query=workflow%3Abuild-windows

Python wheels for ``glearn`` for all supported platforms and versions in the above are available through `PyPI <https://pypi.org/project/glearn/>`_ and `Anaconda Cloud <https://anaconda.org/s-ameli/glearn>`_. If you need ``glearn`` on other platforms, architectures, and Python or PyPy versions, `raise an issue <https://github.com/ameli/glearn/issues>`_ on GitHub and we build its Python Wheel for you.

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

* `What This Packages Does? <https://ameli.github.io/glearn/overview.html>`_
* `Comprehensive Installation Guide <https://ameli.github.io/glearn/tutorials/install.html>`_
* `How to Work with Docker Container? <https://ameli.github.io/glearn/tutorials/docker.html>`_
* `How to Deploy on GPU Devices? <https://ameli.github.io/glearn/tutorials/gpu.html>`_
* `API Reference <https://ameli.github.io/glearn/api.html>`_
* `Interactive Notebook Tutorials <https://mybinder.org/v2/gh/ameli/glearn/HEAD?filepath=notebooks%2Fquick_start.ipynb>`_
* `Publications <https://ameli.github.io/glearn/cite.html>`_

How to Contribute
=================

We welcome contributions via `GitHub's pull request <https://github.com/ameli/glearn/pulls>`_. If you do not feel comfortable modifying the code, we also welcome feature requests and bug reports as `GitHub issues <https://github.com/ameli/glearn/issues>`_.

How to Cite
===========

If you publish work that uses ``glearn``, please consider citing the manuscripts available `here <https://ameli.github.io/glearn/cite.html>`_.

License
=======

|license|

This project uses a `BSD 3-clause license <https://github.com/ameli/glearn/blob/main/LICENSE.txt>`_, in hopes that it will be accessible to most projects. If you require a different license, please raise an `issue <https://github.com/ameli/glearn/issues>`_ and we will consider a dual license.

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
