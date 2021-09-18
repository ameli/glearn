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

            # Find optimal hyperparam
            result = FullLikelihood.maximize_likelihood(
                    z, self.X, self.cov, tol=tol,
                    hyperparam_guess=hyperparam_guess,
                    optimization_method=optimization_method,
                    verbose=verbose)

            # Plot log likelihood
            if plot:
                FullLikelihood.plot_likelihood(z, self.X, self.cov, result)

        elif profile_param == 'var':

            # Find optimal hyperparam
            result = ProfileLikelihood.maximize_likelihood(
                    z, self.X, self.cov, tol=tol,
                    hyperparam_guess=hyperparam_guess,
                    optimization_method=optimization_method,
                    verbose=verbose)

            if plot:
                # Plot log-lp when distance_scale is fixed
                # ProfileLikelihood.plot_likelihood_for_fixed_distance_scale(
                #         z, self.X, self.mixed_cor, result,
                #         numpy.logspace(-2, 2, 5))

                # Plot log-lp when eta is fixed, for a selection of eta
                # ProfileLikelihood.plot_likelihood_for_fixed_eta(
                #         z, self.X, self.cov, result,
                #         numpy.logspace(-2, 2, 5))

                # 3D Plot of log-lp function
                ProfileLikelihood.plot_likelihood(z, self.X, self.mixed_cor,
                                                  result)

                # # Plot first derivative of log likelihood
                # ProfileLikelihood.plot_likelihood_der1_eta(
                #         z, self.X, self.mixed_cor, result)

        elif profile_param == 'var_noise':

            # Find optimal hyperparam
            result = DoubleProfileLikelihood.maximize_likelihood(
                    z, self.X, self.cov, tol=tol,
                    hyperparam_guess=hyperparam_guess,
                    optimization_method=optimization_method, verbose=verbose)

            if plot:
                # Plot log-lp
                DoubleProfileLikelihood.plot_likelihood(
                        z, self.X, self.cov, result)

        else:
            raise ValueError('"profile_param" can be one of "none", ' +
                             '"var", or "var_noise".')

        return result
