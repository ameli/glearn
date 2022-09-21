#! /usr/bin/env python

# SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the license found in the LICENSE.txt file in the root
# directory of this source tree.


# =======
# Imports
# =======

import sys
import os
from glearn import kernels


# ===========
# remove file
# ===========

def remove_file(filename):
    """
    Remove file.
    """

    if os.path.exists(filename):
        os.remove(filename)


# ============
# check kernel
# ============

def check_kernel(kernel):
    """
    Checks ``glearn.kernels`` classes.
    """

    x = 0.5
    kernel(x)
    kernel(x, derivative=1)
    kernel(x, derivative=2)
    kernel.plot(compare_numerical=True, test=True)

    remove_file('kernel.svg')
    print('OK')


# ============
# test kernels
# ============

def test_kernels():
    """
    A test for :mod:`glearn.kernels` module.
    """

    check_kernel(kernels.Exponential())
    check_kernel(kernels.SquareExponential())
    check_kernel(kernels.Linear())
    check_kernel(kernels.RationalQuadratic(alpha=1.2))
    check_kernel(kernels.Matern(nu=1.2))


# ===========
# System Main
# ===========

if __name__ == "__main__":
    sys.exit(test_kernels())
