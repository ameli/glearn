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

Launch an online interactive tutorial in `Jupyter notebook <https://mybinder.org/v2/gh/ameli/glearn/HEAD?filepath=notebooks%2FSpecial%20Functions.ipynb>`_.

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
..    glearn

=====
Links
=====

* `Package on Anaconda Cloud <https://anaconda.org/s-ameli/glearn>`_
* `Package on PyPi <https://pypi.org/project/glearn/>`_
* `Source code on Github <https://github.com/ameli/glearn>`_
.. * `Interactive Jupyter notebook <https://mybinder.org/v2/gh/ameli/glearn/HEAD?filepath=notebooks%2FSpecial%20Functions.ipynb>`_.
.. * `API <https://ameli.github.io/glearn/_modules/modules.html>`_

=================
How to Contribute
=================

We welcome contributions via `Github's pull request <https://github.com/ameli/glearn/pulls>`_. If you do not feel comfortable modifying the code, we also welcome feature request and bug report as `Github issues <https://github.com/ameli/glearn/issues>`_.

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


.. |codecov-devel| image:: https://img.shields.io/codecov/c/github/ameli/glearn
   :target: https://codecov.io/gh/ameli/glearn
.. |docs| image:: https://github.com/ameli/glearn/workflows/docs/badge.svg
   :target: https://ameli.github.io/glearn/index.html
.. |licence| image:: https://img.shields.io/github/license/ameli/glearn
   :target: https://opensource.org/licenses/MIT
.. |travis-devel-linux| image:: https://img.shields.io/travis/com/ameli/glearn?env=BADGE=linux&label=build&branch=main
   :target: https://travis-ci.com/github/ameli/glearn
.. |travis-devel-osx| image:: https://img.shields.io/travis/com/ameli/glearn?env=BADGE=osx&label=build&branch=main
   :target: https://travis-ci.com/github/ameli/glearn
.. |travis-devel-windows| image:: https://img.shields.io/travis/com/ameli/glearn?env=BADGE=windows&label=build&branch=main
   :target: https://travis-ci.com/github/ameli/glearn
.. |implementation| image:: https://img.shields.io/pypi/implementation/glearn
.. |pyversions| image:: https://img.shields.io/pypi/pyversions/glearn
.. |format| image:: https://img.shields.io/pypi/format/glearn
.. |pypi| image:: https://img.shields.io/pypi/v/glearn
   :target: https://pypi.org/project/gaussian-process/
.. |conda| image:: https://anaconda.org/s-ameli/glearn/badges/installer/conda.svg
   :target: https://anaconda.org/s-ameli/glearn
.. |platforms| image:: https://img.shields.io/conda/pn/s-ameli/glearn?color=orange?label=platforms
   :target: https://anaconda.org/s-ameli/glearn
.. |conda-version| image:: https://img.shields.io/conda/v/s-ameli/glearn
   :target: https://anaconda.org/s-ameli/glearn
.. |binder| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/ameli/glearn/HEAD?filepath=notebooks%2FSpecial%20Functions.ipynb
