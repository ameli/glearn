# SPDX-FileCopyrightText: Copyright 2022, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the license found in the LICENSE.txt file in the root directory
# of this source tree.


# =======
# Imports
# =======

from .c_get_config cimport is_use_openmp, is_use_cuda, \
        is_cuda_dynamic_loading, is_debug_mode, is_cython_build_in_source, \
        is_cython_build_for_doc


# ==========
# get config
# ==========

def get_config(key=None):
    """
    Returns the definitions used in the compile-time of the package.

    Parameters
    ----------

    key : str, default=None
        A string with one of the following values:
        
        * ``'use_openmp'``: inquiries if the package is compiled with OpenMP
          support.
        * ``'use_cuda'``: inquiries if the package is compiled with CUDA
          support.
        * ``'cuda_dynamic_loading'``: inquiries if the package is compiled with
          enabling the dynamic loading of CUDA libraries.
        * ``'debug_mode'``: inquiries if the package is compiled with the
          debugging mode enabled.
        * ``'cython_build_in_source'``: inquiries if the Cython source files
          were generated in the source directory during compilation.
        * ``'cython_build_for_doc'``: inquiries if the docstring for Cython
          functions are generated for the purpose of Sphinx documentation.

        If `None`, the full list of all above configurations is returned. 

    Returns
    -------

    config : dict
        If a ``key`` input argument is given, a boolean value corresponding to
        the status of the key is returned. If ``key`` is set to `None` or no
        ``key`` is given, a dictionary with all the above keys is returned.

    See Also
    --------

    imate.info

    Notes
    -----

    To configure the compile-time definitions, export either of these
    variables and set them to ``0`` or ``1`` as applicable:

        .. tab-set::

            .. tab-item:: UNIX
                :sync: unix

                .. prompt:: bash

                    export USE_OPENMP=1
                    export USE_CUDA=1
                    export CUDA_DYNAMIC_LOADING=1
                    export DEBUG_MODE=1
                    export CYTHON_BUILD_IN_SOURCE=1
                    export CYTHON_BUILD_FOR_DOC=1

            .. tab-item:: Windows (Powershell)
                :sync: win

                .. prompt:: powershell

                    $env:USE_OPENMP = "1"
                    $env:USE_CUDA = "1"
                    $env:CUDA_DYNAMIC_LOADING = "1"
                    $env:DEBUG_MODE = "1"
                    $env:CYTHON_BUILD_IN_SOURCE = "1"
                    $env:CYTHON_BUILD_FOR_DOC = "1"

    Examples
    --------

    .. code-block:: python

        >>> from glearn import get_config

        >>> # Using a config key
        >>> get_config('use_openmp')
        True

        >>> # Using no key, results in returning all config
        >>> get_config()
        {
            'use_openmp': True,
            'use_cuda': True,
            'cuda_dynamic_loading': True,
            'debug_mode': False,
            'cython_build_in_source': False,
            'cython_build_for_doc': False,
        }
    """

    if key is None:
        config = {
            'use_openmp': bool(is_use_openmp()),
            'use_cuda': bool(is_use_cuda()),
            'cuda_dynamic_loading': bool(is_cuda_dynamic_loading()),
            'debug_mode': bool(is_debug_mode()),
            'cython_build_in_source': bool(is_cython_build_in_source()),
            'cython_build_for_doc': bool(is_cython_build_for_doc()),
        }
        return config
    elif key == 'use_openmp':
        return bool(is_use_openmp())
    elif key == 'use_cuda':
        return bool(is_use_cuda())
    elif key == 'cuda_dynamic_loading':
        return bool(is_cuda_dynamic_loading())
    elif key == 'debug_mode':
        return bool(is_debug_mode())
    elif key == 'cython_build_in_source':
        return bool(is_cython_build_in_source())
    elif key == 'cython_build_for_doc':
        return bool(is_cython_build_for_doc())
    else:
        raise ValueError('Invalid "key".')
