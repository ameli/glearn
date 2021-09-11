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

from ._direct_likelihood import DirectLikelihood
from ._profile_likelihood import ProfileLikelihood
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
        return DirectLikelihood.log_likelihood(z, self.X, self.cov,
                                               sign_switch, hyperparam)

    # =======================
    # maximize log likelihood
    # =======================

    def maximize_log_likelihood(
            self,
            z,
            hyperparam_guess,
            likelihood_method='direct',
            optimization_method='Newton-CG',
            profile_eta=False,
            plot=False):
        """
        """

        if likelihood_method == 'direct':

            if optimization_method == 'chandrupatla':
                raise ValueError('"chandrupatla" method can only be used ' +
                                 'with "profiled" likelihood method.')

            # Find optimal hyperparam
            result = DirectLikelihood.maximize_log_likelihood(
                    z, self.X, self.cov, tol=1e-3,
                    hyperparam_guess=hyperparam_guess,
                    optimization_method=optimization_method)

            # Plot log likelihood
            if plot:
                optimal_sigma = result['sigma']
                optimal_sigma0 = result['sigma0']
                optimal_hyperparam = [optimal_sigma, optimal_sigma0]
                DirectLikelihood.plot_log_likelihood(z, self.X, self.cov,
                                                     optimal_hyperparam)

            # return ProfileLikelihood.maximize_log_likelihood_with_sigma_eta(
            #         z, self.X, self.mixed_cor)

        elif likelihood_method == 'profiled':

            # Find optimal hyperparam
            result = ProfileLikelihood.maximize_log_likelihood(
                    z, self.X, self.cov, tol=1e-3,
                    hyperparam_guess=hyperparam_guess,
                    optimization_method=optimization_method,
                    profile_eta=profile_eta)

            if plot:
                # Plot log-lp when eta is fixed, for a selection of eta
                ProfileLikelihood.plot_log_likelihood_for_fixed_eta(
                        z, self.X, self.mixed_cor,
                        numpy.r_[result['eta'], numpy.logspace(-3, 3, 7)])

                # Plot log-lp when distance_scale is fixed, for a selection of
                # distance_scale
                ProfileLikelihood.plot_log_likelihood_for_fixed_distance_scale(
                        z, self.X, self.mixed_cor,
                        numpy.r_[result['distance_scale'],
                                 numpy.logspace(-3, 3, 7)])

                # 3D Plot of log-lp function
                ProfileLikelihood.plot_log_likelihood(z, self.X,
                                                      self.mixed_cor, result)

                # Plot first derivative of log likelihood
                optimal_eta = result['eta']
                ProfileLikelihood.plot_log_likelihood_der1_eta(
                        z, self.X, self.mixed_cor, optimal_eta)

        return result
