****************
gaussian process
****************

|licence| |docs|

This is a python package for machine leanring using Gaussian process regression.

*This package is under developement and has not been released yet.*

========
Features
========

* **Lightweight:** This package requires *no python dependency* at runtime.
* **Cython interface:** Both Python and Cython interfaces are available.
* **Releasing GIL:** Most importantly, the functions can be used in ``with nogil:`` environment, which is essential in parallel OpenMP applications with Cython.

====================
Interactive Tutorial
====================

|binder|

Launch an online interactive tutorial in `Jupyter notebook <https://mybinder.org/v2/gh/ameli/gaussian_process/HEAD?filepath=notebooks%2FSpecial%20Functions.ipynb>`_.

.. toctree::
    :maxdepth: 1
    :caption: Documentation

    Install <install>
    List of Functions <list>

.. toctree::
    :maxdepth: 1
    :caption: Modules User Guide

.. toctree::
    :maxdepth: 1
    :caption: Development
              
    Package API <_modules/modules>
    Running Tests <tests>
    Change Log <changelog>

.. =======
.. Modules
.. =======
..
.. .. autosummary::
..    :toctree: _autosummary
..    :recursive:
..    :nosignatures:
..
..    gaussian_process

=====
Links
=====

* `Package on Anaconda Cloud <https://anaconda.org/s-ameli/gaussian_process>`_
* `Package on PyPi <https://pypi.org/project/gaussian_process/>`_
* `Source code on Github <https://github.com/ameli/gaussian_process>`_
.. * `Interactive Jupyter notebook <https://mybinder.org/v2/gh/ameli/gaussian_process/HEAD?filepath=notebooks%2FSpecial%20Functions.ipynb>`_.
.. * `API <https://ameli.github.io/gaussian_process/_modules/modules.html>`_

=================
How to Contribute
=================

We welcome contributions via `Github's pull request <https://github.com/ameli/gaussian_process/pulls>`_. If you do not feel comfortable modifying the code, we also welcome feature request and bug report as `Github issues <https://github.com/ameli/gaussian_process/issues>`_.

================
Related Packages
================

* TODO

================
Acknowledgements
================

* National Science Foundation #1520825
* American Heart Association #18EIA33900046

======
Credit
======

* TODO

==================
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. |codecov-devel| image:: https://img.shields.io/codecov/c/github/ameli/gaussian_process
   :target: https://codecov.io/gh/ameli/gaussian_process
.. |docs| image:: https://github.com/ameli/gaussian_process/workflows/docs/badge.svg
   :target: https://ameli.github.io/gaussian_process/index.html
.. |licence| image:: https://img.shields.io/github/license/ameli/gaussian_process
   :target: https://opensource.org/licenses/MIT
.. |travis-devel-linux| image:: https://img.shields.io/travis/com/ameli/gaussian_process?env=BADGE=linux&label=build&branch=main
   :target: https://travis-ci.com/github/ameli/gaussian_process
.. |travis-devel-osx| image:: https://img.shields.io/travis/com/ameli/gaussian_process?env=BADGE=osx&label=build&branch=main
   :target: https://travis-ci.com/github/ameli/gaussian_process
.. |travis-devel-windows| image:: https://img.shields.io/travis/com/ameli/gaussian_process?env=BADGE=windows&label=build&branch=main
   :target: https://travis-ci.com/github/ameli/gaussian_process
.. |implementation| image:: https://img.shields.io/pypi/implementation/gaussian_process
.. |pyversions| image:: https://img.shields.io/pypi/pyversions/gaussian_process
.. |format| image:: https://img.shields.io/pypi/format/gaussian_process
.. |pypi| image:: https://img.shields.io/pypi/v/gaussian_process
   :target: https://pypi.org/project/gaussian-process/
.. |conda| image:: https://anaconda.org/s-ameli/gaussian_process/badges/installer/conda.svg
   :target: https://anaconda.org/s-ameli/gaussian_process
.. |platforms| image:: https://img.shields.io/conda/pn/s-ameli/gaussian_process?color=orange?label=platforms
   :target: https://anaconda.org/s-ameli/gaussian_process
.. |conda-version| image:: https://img.shields.io/conda/v/s-ameli/gaussian_process
   :target: https://anaconda.org/s-ameli/gaussian_process
.. |binder| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/ameli/gaussian_process/HEAD?filepath=notebooks%2FSpecial%20Functions.ipynb
