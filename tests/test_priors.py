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
from glearn import priors


# ===========
# remove file
# ===========

def remove_file(filename):
    """
    Remove file.
    """

    if os.path.exists(filename):
        os.remove(filename)


# ===========
# check prior
# ===========

def check_prior(prior):
    """
    Checks ``glearn.priors`` classes.
    """

    t = [0, 0.5, 1]
    prior.pdf(t)
    prior.pdf_jacobian(t)
    prior.pdf_hessian(t)
    prior.log_pdf(t)
    prior.log_pdf_jacobian(t)
    prior.log_pdf_hessian(t)
    prior.plot(log_scale=True, compare_numerical=True, test=True)

    remove_file('prior.svg')
    print('OK')


# ===========
# test priors
# ===========

def test_priors():
    """
    A test for :mod:`glearn.priors` module.
    """

    check_prior(priors.Uniform(0.2, 0.9))
    check_prior(priors.Normal(1, 3))
    check_prior(priors.Cauchy(1, 2))
    check_prior(priors.StudentT(4))
    check_prior(priors.Erlang(2, 4))
    check_prior(priors.Gamma(2, 4))
    check_prior(priors.InverseGamma(4, 2))
    check_prior(priors.BetaPrime(2, 4))


# ===========
# System Main
# ===========

if __name__ == "__main__":
    sys.exit(test_priors())
