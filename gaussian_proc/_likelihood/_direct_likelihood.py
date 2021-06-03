# SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the license found in the LICENSE.txt file in the root directory
# of this source tree.


# =======
# Imports
# =======

import numpy
from scipy.optimize import minimize
from functools import partial


# =================
# Direct Likelihood
# =================

class DirectLikelihood(object):

    # ==============
    # log likelihood
    # ==============

    @staticmethod
    def log_likelihood(z, X, K_mixed, sign_switch, hyperparameters):
        """
        Here we use direct parameter, sigma and sigma0

        sign_switch chnages the sign of the output from lp to -lp. When True,
        this is used to minimizing (instad of maximizing) the negative of
        log-likelihood function.
        """

        # hyperparameters
        sigma = hyperparameters[0]
        sigma0 = hyperparameters[1]

        # S is the (sigma**2) * K + (sigma0**2) * I, but we don't construct it
        # Also, Kn is K + eta I, where eta = (sigma0 / sigma)**2
        eta = (sigma0 / sigma)**2
        logdet_Kn = K_mixed.logdet(eta)
        logdet_S = K_mixed.get_matrix_size() * sigma**2 + logdet_Kn

        # Compute log det (X.T*Sinv*X)
        Y = K_mixed.solve(eta, X) / sigma**2
        w = K_mixed.solve(eta, z) / sigma**2

        XtSinvX = numpy.matmul(X.T, Y)
        logdet_XtSinvX = numpy.log(numpy.linalg.det(XtSinvX))

        Binv = numpy.linalg.inv(XtSinvX)
        YBinvYt = numpy.matmul(Y, numpy.matmul(Binv, Y.T))

        # Log likelihood
        lp = -0.5*logdet_S - 0.5*logdet_XtSinvX - \
            0.5*numpy.dot(z, w-numpy.dot(YBinvYt, z))

        # If lp is used in scipy.optimize.minimize, change the sign to optain
        # the minimum of -lp
        if sign_switch:
            lp = -lp

        return lp

    # =======================
    # maximize log likelihood
    # =======================

    @staticmethod
    def maximize_log_likelihood(z, X, K_mixed):
        """
        Maximizing the log-likelihood function over the space of parameters
        sigma and eta.
        """

        print('Maximize log likelihood with sigma sigma0 ...')

        # Initial points # SETTING
        guess_sigma = 0.001
        guess_sigma0 = 0.001
        guess_parameters = [guess_sigma, guess_sigma0]

        # Partial function with minus to make maximization to a minimization
        sign_switch = True
        log_likelihood_partial_function = partial(
                DirectLikelihood.log_likelihood, z, X, K_mixed, sign_switch)

        # Minimize
        # method = 'BFGS'
        # method = 'CG'
        method = 'Nelder-Mead'
        tol = 1e-6  # SETTING
        res = minimize(log_likelihood_partial_function, guess_parameters,
                       method=method, tol=tol)

        print('Iter: %d, Eval: %d, Success: %s'
              % (res.nit, res.nfev, res.success))

        # Extract res
        sigma = res.x[0]
        sigma0 = res.x[1]
        eta = (sigma0/sigma)**2
        max_lp = -res.fun

        # Output distionary
        results = {
            'sigma': sigma,
            'sigma0': sigma0,
            'eta': eta,
            'max_lp': max_lp
        }

        return results
