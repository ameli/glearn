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

from ._full_likelihood import FullLikelihood
from ._profile_likelihood import ProfileLikelihood
from ._double_profile_likelihood import DoubleProfileLikelihood
import numpy


# ==========
# Likelihood
# ==========

class Likelihood(object):
    """
    """

    # ====
    # init
    # ====

    def __init__(self, mean, cov):
        """
        """

        self.mean = mean
        self.cov = cov

        self.X = self.mean.X
        self.mixed_cor = self.cov.mixed_cor

    # ==========
    # likelihood
    # ==========

    def likelihood(self, z, hyperparam):
        """
        """

        sign_switch = False
        return FullLikelihood.likelihood(z, self.X, self.cov, sign_switch,
                                         hyperparam)

    # ===================
    # maximize likelihood
    # ===================

    def maximize_likelihood(
            self,
            z,
            hyperparam_guess,
            profile_param='var',
            optimization_method='Newton-CG',
            tol=1e-3,
            verbose=False,
            plot=False):
        """
        """

        if profile_param == 'none':

            if optimization_method == 'chandrupatla':
                raise ValueError('"chandrupatla" method can only be used ' +
                                 'with "profiled" likelihood method.')

            likelihood = FullLikelihood(z, self.X, self.cov)

            # Find optimal hyperparam
            result = likelihood.maximize_likelihood(
                    tol=tol, hyperparam_guess=hyperparam_guess,
                    optimization_method=optimization_method, verbose=verbose)

            # Plot log likelihood
            if plot:

                # Plot likelihood for scale, fixed sigma and sigma0
                likelihood.plot_likelihood_versus_scale(
                        result, other_sigmas=numpy.logspace(-1, 1, 3))

                # Plot likelihood for sigma, fixed sigma0 and scale
                likelihood.plot_likelihood_versus_sigma(
                        result, other_scales=numpy.logspace(-1, 1, 3))

                # Plot likelihood for sigma0, fixed sigma and scale
                likelihood.plot_likelihood_versus_sigma0(
                        result, other_scales=numpy.logspace(-1, 1, 3))

                # 2d plot of likelihood versus sigma0 and sigma
                likelihood.plot_likelihood_versus_sigma0_sigma(result)

        elif profile_param == 'var':

            likelihood = ProfileLikelihood(z, self.X, self.cov)

            # Find optimal hyperparam
            result = likelihood.maximize_likelihood(
                    tol=tol, hyperparam_guess=hyperparam_guess,
                    optimization_method=optimization_method, verbose=verbose)

            if plot:
                # Plot log-lp versus eta
                likelihood.plot_likelihood_versus_eta(
                        result, numpy.logspace(-2, 2, 5))

                # Plot log-lp versus scale
                likelihood.plot_likelihood_versus_scale(
                        result, numpy.logspace(-2, 2, 5))

                # 3D Plot of log-lp function
                likelihood.plot_likelihood_versus_eta_scale(result)

                # Plot first derivative of log likelihood
                likelihood.plot_likelihood_der1_eta(result)

        elif profile_param == 'var_noise':

            likelihood = DoubleProfileLikelihood(z, self.X, self.cov)

            # Find optimal hyperparam
            result = likelihood.maximize_likelihood(
                    tol=tol, hyperparam_guess=hyperparam_guess,
                    optimization_method=optimization_method, verbose=verbose)

            if plot:
                # Plot log-lp
                likelihood.plot_likelihood_versus_scale(result)

        else:
            raise ValueError('"profile_param" can be one of "none", ' +
                             '"var", or "var_noise".')

        return result
