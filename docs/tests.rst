.. _Run_Tests:

*************
Running Tests
*************

|codecov-devel|

The package can be tested by running the `test scripts <https://github.com/ameli/gaussian_process/tree/main/tests>`_, which tests all `modules <https://github.com/ameli/gaussian_process/tree/main/gaussian_process>`_. 

=============================
Running Tests with ``pytest``
=============================

1. Install ``pytest-cov``:

   ::

       python -m pip install pytest-cov

2. Install this package by either of the methods described in the :ref:`installation instructions <Install_Package>`.

2. Clone the package source code and install the test dependencies:

   ::

       git clone https://github.com/ameli/gaussian_process.git
       cd gaussian_process
       python -m pip install -r tests/requirements.txt

3. Test the package by:

   ::

       cd tests
       pytest

   .. warning::

       Do not run tests in the root directory of the package ``/gaussian_process``. To properly run tests, change current working directory to ``/gaussian_process/tests`` sub-directory.

==========================
Running Tests with ``tox``
==========================

To run a test in a virtual environment, use ``tox`` as follows:

1. Clone the source code from the repository:
   
   ::
       
       git clone https://github.com/ameli/gaussian_process.git

2. Install ``tox``:
   
   ::
       
       python -m pip install tox

3. run tests by
   
   ::
       
       cd gaussian_process
       tox
  
.. |codecov-devel| image:: https://img.shields.io/codecov/c/github/ameli/gaussian_process
   :target: https://codecov.io/gh/ameli/gaussian_process
.. |build-linux| image:: https://github.com/ameli/gaussian_process/workflows/build-linux/badge.svg
   :target: https://github.com/ameli/gaussian_process/actions?query=workflow%3Abuild-linux 
.. |build-macos| image:: https://github.com/ameli/gaussian_process/workflows/build-macos/badge.svg
   :target: https://github.com/ameli/gaussian_process/actions?query=workflow%3Abuild-macos
.. |build-windows| image:: https://github.com/ameli/gaussian_process/workflows/build-windows/badge.svg
   :target: https://github.com/ameli/gaussian_process/actions?query=workflow%3Abuild-windows
