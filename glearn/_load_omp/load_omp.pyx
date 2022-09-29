# SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the license found in the LICENSE.txt file in the root directory
# of this source tree.


# ===============
# Import packages
# ===============

# To avoid this error:
#   OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already
#   initialized.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from cython.parallel cimport parallel


# ========
# load omp
# ========

def load_omp():
    """
    A python interface to load omp.

    Why this is needed?

    On MacOS, a segmentation fault 11 error is returned when there are
    duplicate libomp libraries are loaded. In this package, there are two
    libomp libraries:

    1. In /glearn/_mean/linear_model.py, this module is being loaded
        
        from detkit import orthogonalize

    2. This package itself loads libomp when any of /glearn/_correlation/*.pyx
       is loaded.

    There are two ways to resolve this in /glearn/__init__.py.

    1. Import LinearModel after all glearn  modules, so that they choose the
       libomp that this package is compiled with.
    2. In /glearn/_mean/linear_model.py, do not import detkit at the top of the
       script, rather, import detkit whenever orthogonalize() is called.
    3. Create a dummy function to call libomp (the one that this package is
       compiled with), and call this function in the beginning of
       /glearn/__init__.py.

    The purpose of this function is the option (3) in the above.
    """

    c_load_omp()


# ==========
# c load omp
# ==========
    
cdef void c_load_omp() nogil:

    with parallel():
        pass
