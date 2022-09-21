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
from glearn.sample_data import generate_points, generate_data


# ===========
# remove file
# ===========

def remove_file(filename):
    """
    Remove file.
    """

    if os.path.exists(filename):
        os.remove(filename)


# ================
# test sample data
# ================

def test_sample_data():
    """
    A test for :mod:`glearn.priors` module.
    """

    # 1D grid
    generate_points(10, grid=True, a=0.2, b=0.4, contrast=0.8)

    # 1D random
    x = generate_points(100, grid=False, a=0.2, b=0.4, contrast=0.8)

    # Generate sample data
    generate_data(x, noise_magnitude=0.1, seed=0, plot='save')
    remove_file('data.svg')

    # 2D grid
    generate_points(30, dimension=2, grid=True)

    # 2D random
    x = generate_points(100, dimension=2, grid=False, a=(0.2, 0.3),
                        b=(0.4, 0.5), contrast=0.7)

    # Generate sample data
    generate_data(x, noise_magnitude=0.1, seed=0, plot='save')
    remove_file('data.svg')


# ===========
# System Main
# ===========

if __name__ == "__main__":
    sys.exit(test_sample_data())
