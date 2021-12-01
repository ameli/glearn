.. _Install_Package:

*******
Install
*******

===================
Supported Platforms
===================

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

.. note::

    * For the Python/PyPy versions indicated by |y| in the above, this package can be installed using either ``pip`` or ``conda`` (see :ref:`installation instructions <InstallationMethods>` below.)
    * This package cannot be installed via ``pip`` or ``conda`` on the Python/PyPy versions indicated by |n| in the above table.
    * To install on the older Python 3 versions that are not listed in the above (for example Python 3.5), you should *build* this package from the source code (see :ref:`build instructions <Build_Locally>`).


============
Dependencies
============

* **At runtime:** This package does not have any dependencies at runtime.
* **For tests:** To :ref:`run tests <Run_Tests>`, ``scipy`` package is required and can be installed by

  ::

      python -m pip install -r tests/requirements.txt

.. _InstallationMethods:

===============
Install Package
===============

Install by either through :ref:`PyPi <Install_PyPi>`, :ref:`Conda <Install_Conda>`, or :ref:`build locally <Build_Locally>`.

.. _Install_PyPi:

-----------------
Install from PyPi
-----------------

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

---------------------------
Install from Anaconda Cloud
---------------------------

|conda-version| |conda-platform|

Alternatively, the package can be installed through `Anaconda could <https://anaconda.org/s-ameli/glearn>`_.

* In **Linux** and **Windows**:
  
  ::
      
      conda install -c s-ameli glearn

* In **macOS**:
  
  ::
      
      conda install -c s-ameli -c conda-forge glearn

.. note::

    On macOS and Python 2.7, install this package with ``pip`` instead of ``conda``.

.. _Build_Locally:

----------------------------------
Build and Install from Source Code
----------------------------------

|release|

**Build dependencies:** To build the package from the source code, ``numpy`` and ``cython`` are required. These dependencies are installed automatically during the build process and no action is needed.

1. Install both C and Fortran compilers as follows.

   * **Linux:** Install ``gcc``, for instance, by ``apt`` (or any other package manager on your Linux distro)

     ::

         sudo apt install gcc

   * **macOS:** Install ``gcc`` via Homebrew:

     ::

         sudo brew install gcc

     .. note::
         
         If ``gcc`` is already installed, but Fortran compiler is yet not available on macOS, you may resolve this issue by reinstalling:
         
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

.. warning::

    After the package is built and installed from the source code, the package cannot be imported properly if the current working directory is the same as the source code directory.
    To properly import the package, change the current working directory to a directory anywhere else **outside** of the source code directory. For instance:

    .. code-block:: python

        cd ..
        python
        >>> import glearn


==============================
Install in Virtual Environment
==============================

If you do not want the installation to occupy your main python's site-packages, you may install the package in an isolated virtual environment. Below we describe the installation procedure in two common virtual environments, namely, :ref:`virtualenv <virtualenv_env>` and :ref:`conda <conda_env>`.

.. _virtualenv_env:

-------------------------------------
Install in ``virtualenv`` Environment
-------------------------------------

1. Install ``virtualenv``:

   ::

       python -m pip install virtualenv

2. Create a virtual environment and give it a name, such as ``glearn_env``

   ::

       python -m virtualenv glearn_env

3. Activate python in the new environment

   ::

       source glearn_env/bin/activate

4. Install ``glearn`` package with any of the :ref:`above methods <InstallationMethods>`. For instance:

   ::

       python -m pip install glearn
   
   Then, use the package in this environment.

5. To exit from the environment

   ::

       deactivate

.. _conda_env:

--------------------------------
Install in ``conda`` Environment
--------------------------------

In the following, it is assumed `anaconda <https://www.anaconda.com/products/individual#Downloads>`_ (or `miniconda <https://docs.conda.io/en/latest/miniconda.html>`_) is installed.

1. Initialize conda (if it was not initialized before)

   ::

       conda init

   You may need to close and reopen the terminal after the above command. Alternatively, instead of the above, you can do

   ::

       sudo sh $(conda info --root)/etc/profile.d/conda.sh

2. Create a virtual environment and give it a name, such as ``glearn_env``

   ::

       conda create --name glearn_env -y

   The command ``conda info --envs`` shows the list of all environments. The current environment is marked by an asterisk in the list, which should be the default environment at this stage. In the next step, we will change the current environment to the one we created.

3. Activate the new environment

   ::

       conda activate glearn_env

4. Install ``glearn`` with any of the :ref:`above methods <InstallationMethods>`. For instance:

   ::

       conda install -c s-ameli glearn
   
   Then, use the package in this environment.

5. To exit from the environment

   ::

       conda deactivate

.. |implementation| image:: https://img.shields.io/pypi/implementation/glearn
.. |pyversions| image:: https://img.shields.io/pypi/pyversions/glearn
.. |format| image:: https://img.shields.io/pypi/format/glearn
.. |pypi| image:: https://img.shields.io/pypi/v/glearn
.. |conda| image:: https://anaconda.org/s-ameli/glearn/badges/installer/conda.svg
   :target: https://anaconda.org/s-ameli/glearn
.. |platforms| image:: https://img.shields.io/conda/pn/s-ameli/glearn?color=orange?label=platforms
   :target: https://anaconda.org/s-ameli/glearn
.. |conda-version| image:: https://img.shields.io/conda/v/s-ameli/glearn
   :target: https://anaconda.org/s-ameli/glearn
.. |release| image:: https://img.shields.io/github/v/tag/ameli/glearn
   :target: https://github.com/ameli/glearn/releases/
.. |conda-platform| image:: https://anaconda.org/s-ameli/glearn/badges/platforms.svg
   :target: https://anaconda.org/s-ameli/glearn
