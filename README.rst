*******
G-Learn
*******

|licence| |docs|

Gaussian Process for Machine Learning.

*This package is under development and has not been released.*

========
Features
========

* TODO

====================
Interactive Tutorial
====================

|binder|

Launch an online interactive tutorial in `Jupyter notebook <https://mybinder.org/v2/gh/ameli/glearn/main?labpath=notebooks%2Fdemo.ipynb>`_.


=====
Links
=====

* `Documentation <https://ameli.github.io/glearn/index.html>`_
* `Package on Anaconda Cloud <https://anaconda.org/s-ameli/glearn>`_
* `Package on PyPi <https://pypi.org/project/glearn/>`_

=======
Install
=======

-------------------
Supported Platforms
-------------------

Successful installation and tests have been performed on the following platforms and Python/PyPy versions shown in the table below.

.. |y| unicode:: U+2714
.. |n| unicode:: U+2716

+----------+-----+-----+-----+-----+-----+-----+-----+-----+-----------------+
| Platform | Python version              | PyPy version    | Status          |
+          +-----+-----+-----+-----+-----+-----+-----+-----+                 +
|          | 2.7 | 3.6 | 3.7 | 3.8 | 3.9 | 2.7 | 3.6 | 3.7 |                 |
+==========+=====+=====+=====+=====+=====+=====+=====+=====+=================+
| Linux    | |y| | |y| | |y| | |y| | |y| | |y| | |y| | |y| | |build-linux|   |
+----------+-----+-----+-----+-----+-----+-----+-----+-----+-----------------+
| macOS    | |y| | |y| | |y| | |y| | |y| | |n| | |n| | |n| | |build-macos|   |
+----------+-----+-----+-----+-----+-----+-----+-----+-----+-----------------+
| Windows  | |n| | |y| | |y| | |y| | |y| | |n| | |n| | |n| | |build-windows| |
+----------+-----+-----+-----+-----+-----+-----+-----+-----+-----------------+

.. |build-linux| image:: https://github.com/ameli/glearn/workflows/build-linux/badge.svg
   :target: https://github.com/ameli/glearn/actions?query=workflow%3Abuild-linux 
.. |build-macos| image:: https://github.com/ameli/glearn/workflows/build-macos/badge.svg
   :target: https://github.com/ameli/glearn/actions?query=workflow%3Abuild-macos
.. |build-windows| image:: https://github.com/ameli/glearn/workflows/build-windows/badge.svg
   :target: https://github.com/ameli/glearn/actions?query=workflow%3Abuild-windows


* For the Python/PyPy versions indicated by |y| in the above, this package can be installed using either ``pip`` or ``conda`` (see `Install Package`_ below.)
* This package cannot be installed via ``pip`` or ``conda`` on the Python/PyPy versions indicated by |n| in the above table.
* To install on the older Python 3 versions that are not listed in the above (for example Python 3.5), you should *build* this package from the source code (see `Build and Install from Source Code`_).


------------
Dependencies
------------

* **At runtime:** TODO
* **For tests:** To run `Test`_, ``scipy`` package is required and can be installed by

  ::

      python -m pip install -r tests/requirements.txt

---------------
Install Package
---------------

Either `Install from PyPi`_, `Install from Anaconda Cloud`_, or `Build and Install from Source Code`_.

.. _Install_PyPi:

~~~~~~~~~~~~~~~~~
Install from PyPi
~~~~~~~~~~~~~~~~~

|pypi| |format| |implementation| |pyversions|

The recommended installation method is through the package available at `PyPi <https://pypi.org/project/glearn>`_ using ``pip``.

1. Ensure ``pip`` is installed within Python and upgrade the existing ``pip`` by

   ::

       python -m ensurepip
       python -m pip install --upgrade pip

   If you are using PyPy instead of Python, ensure ``pip`` is installed and upgrade the existing ``pip`` by

   ::

       pypy -m ensurepip
       pypy -m pip install --upgrade pip

2. Install this package in Python by
   
   ::
       
       python -m pip install glearn

   or, in PyPy by

   ::
       
       pypy -m pip install glearn

.. _Install_Conda:

~~~~~~~~~~~~~~~~~~~~~~~~~~~
Install from Anaconda Cloud
~~~~~~~~~~~~~~~~~~~~~~~~~~~

|conda-version| |conda-platform|

Alternatively, the package can be installed through `Anaconda could <https://anaconda.org/s-ameli/glearn>`_.

* In **Linux** and **Windows**:
  
  ::
      
      conda install -c s-ameli glearn

* In **macOS**:
  
  ::
      
      conda install -c s-ameli -c conda-forge glearn

.. _Build_Locally:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Build and Install from Source Code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

|release|

**Build dependencies:** To build the package from the source code, ``numpy`` and ``cython`` are required. These dependencies are installed automatically during the build process and no action is needed.

1. Install both C and Fortran compilers as follows.

   * **Linux:** Install ``gcc``, for instance, by ``apt`` (or any other package manager on your Linux distro)

     ::

         sudo apt install gcc

   * **macOS:** Install ``gcc`` via Homebrew:

     ::

         sudo brew install gcc

     *Note:* If ``gcc`` is already installed, but Fortran compiler is yet not available on macOS, you may resolve this issue by reinstalling:
     
     ::
         
         sudo brew reinstall gcc

   * **Windows:** Install both `Microsoft Visual C++ compiler <https://visualstudio.microsoft.com/vs/features/cplusplus/>`_ and Intel Fortran compiler (`Intel oneAPI <https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/fortran-compiler.html>`_). Open the command prompt (where you will enter the installation commands in the next step) and load the Intel compiler variables by

     ::

         C:\Program Files (x86)\Intel\oneAPI\setvars.bat

     Here, we assumed the Intel Fortran compiler is installed in ``C:\Program Files (x86)\Intel\oneAPI``. You may set this directory accordingly to the directory of your Intel compiler.


2. Clone the source code and install this package by
   
   ::

       git clone https://github.com/ameli/glearn.git
       cd glearn
       python -m pip install .

**Warning:** After the package is built and installed from the source code, the package cannot be imported properly if the current working directory is the same as the source code directory. To properly import the package, change the current working directory to a directory anywhere else **outside** of the source code directory. For instance:
    
.. code-block:: python
   
   cd ..
   python
   >>> import glearn

====
Test
====

|codecov-devel|

To test package, install ``tox``:

::

    python -m pip install tox

and test the package with

::

    tox

=======
Modules
=======

========================  ===============================================================================================================
Syntax                    User guide
========================  ===============================================================================================================
``todo(nu, z, n)``        Module name todo  <https://ameli.github.io/glearn/module_name.html>`_
========================  ===============================================================================================================

**Typed Arguments:**

========  ==============================  ==============================================================
Argument   Type                           Description
========  ==============================  ==============================================================
``nu``    ``double``                      Parameter
========  ==============================  ==============================================================


.. |image01| image:: https://raw.githubusercontent.com/ameli/glearn/main/docs/images/image01.svg
.. |image02| image:: https://raw.githubusercontent.com/ameli/glearn/main/docs/images/image02.svg
.. |image03| image:: https://raw.githubusercontent.com/ameli/glearn/main/docs/images/image03.svg
.. |image04| image:: https://raw.githubusercontent.com/ameli/glearn/main/docs/images/image04.svg
.. |image05| image:: https://raw.githubusercontent.com/ameli/glearn/main/docs/images/image05.svg
.. |image06| image:: https://raw.githubusercontent.com/ameli/glearn/main/docs/images/image06.svg
.. |image07| image:: https://raw.githubusercontent.com/ameli/glearn/main/docs/images/image07.svg
.. |image08| image:: https://raw.githubusercontent.com/ameli/glearn/main/docs/images/image08.svg
.. |image09| image:: https://raw.githubusercontent.com/ameli/glearn/main/docs/images/image09.svg
.. |image10| image:: https://raw.githubusercontent.com/ameli/glearn/main/docs/images/image10.svg
.. |image11| image:: https://raw.githubusercontent.com/ameli/glearn/main/docs/images/image11.svg

========
Examples
========
 

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

* TODO.

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
   :target: https://pypi.org/project/special-functions/
.. |conda| image:: https://anaconda.org/s-ameli/glearn/badges/installer/conda.svg
   :target: https://anaconda.org/s-ameli/glearn
.. |platforms| image:: https://img.shields.io/conda/pn/s-ameli/glearn?color=orange?label=platforms
   :target: https://anaconda.org/s-ameli/glearn
.. |conda-version| image:: https://img.shields.io/conda/v/s-ameli/glearn
   :target: https://anaconda.org/s-ameli/glearn
.. |conda-platform| image:: https://anaconda.org/s-ameli/glearn/badges/platforms.svg
   :target: https://anaconda.org/s-ameli/glearn
.. |release| image:: https://img.shields.io/github/v/tag/ameli/glearn
   :target: https://github.com/ameli/glearn/releases/
.. |binder| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/ameli/glearn/main?labpath=notebooks%2Fdemo.ipynb
