.. _install-wheels:

Install from Wheels
===================

|project| offers Python wheels for a variety of operating systems and Python versions. These wheels are available on both `PyPI <https://pypi.org/project/glearn>`_ and `Anaconda Cloud <https://anaconda.org/s-ameli/glearn>`_, providing a convenient way to install the package using either ``pip`` or ``conda``.

Install with ``pip``
--------------------

|pypi|

First, ensure that you have ``pip`` installed

.. prompt:: bash

    python -m ensurepip --upgrade

For further detail on installing ``pip``, refer to `pip installation documentation <https://pip.pypa.io/en/stable/installation/>`__.

To install |project| and its Python dependencies using ``pip`` by

.. prompt:: bash
    
    python -m pip install --upgrade pip
    python -m pip install glearn

If you are using `PyPy <https://www.pypy.org/>`__ instead of Python, you can first ensure ``pip`` is installed by

.. prompt:: bash

    pypy -m ensurepip --upgrade

Next, you can install |project| as follows:

.. prompt:: bash
    
    pypy -m pip install --upgrade pip
    pypy -m pip install glearn

Install with ``conda``
----------------------

|conda-version|

Alternatively, you can install |project| via ``conda``. To do so, you may refer to the `conda instalation documentation <https://conda.io/projects/conda/en/latest/user-guide/install/index.html>`__ to set up ``conda`` on your system. Once ``conda`` is ready, you can install |project| along with its Python dependencies by using the following command

.. prompt:: bash

    conda install s-ameli::glearn

.. |pypi| image:: https://img.shields.io/pypi/v/glearn
   :target: https://pypi.org/project/glearn
.. |conda-version| image:: https://img.shields.io/conda/v/s-ameli/glearn
   :target: https://anaconda.org/s-ameli/glearn
