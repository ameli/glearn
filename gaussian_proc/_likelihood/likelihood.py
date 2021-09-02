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
            plot=False):
        """
        """

        if likelihood_method == 'direct':

            if optimization_method == 'chandrupatla':
                raise ValueError('"chandrupatla" method can only be used ' +
                                 'with "profiled" likelihood method.')

            # Find hyperparam
            results = DirectLikelihood.maximize_log_likelihood(
                    z, self.X, self.cov, tol=1e-3,
                    hyperparam_guess=hyperparam_guess,
                    optimization_method=optimization_method)

            # Plot log likelihood
            if plot:
                optimal_sigma = results['sigma']
                optimal_sigma0 = results['sigma0']
                optimal_hyperparam = [optimal_sigma, optimal_sigma0]
                DirectLikelihood.plot_log_likelihood(z, self.X, self.cov,
                                                     optimal_hyperparam)

            # return ProfileLikelihood.maximize_log_likelihood_with_sigma_eta(
            #         z, self.X, self.mixed_cor)

        elif likelihood_method == 'profiled':

            if optimization_method == 'chandrupatla' and \
                    len(hyperparam_guess) > 1:
                raise ValueError('Length of "hyperparam_guess" should be one' +
                                 'when "chandrupatla" optimization method ' +
                                 'used.')

            # Find hyperparam
            results = ProfileLikelihood.maximize_log_likelihood(
                    z, self.X, self.cov, tol=1e-3,
                    hyperparam_guess=hyperparam_guess,
                    optimization_method=optimization_method)

            # Plot first derivative of log likelihood
            if plot:
                optimal_eta = results['eta']
                ProfileLikelihood.plot_log_likelihood_der1_eta(
                        z, self.X, self.K, self.mixed_cor, optimal_eta)

        return results
