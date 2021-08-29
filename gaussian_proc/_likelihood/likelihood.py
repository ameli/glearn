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
        self.S_mixed = cov

        self.X = self.mean.X
        self.K_mixed = self.S_mixed.K_mixed

    # ==========
    # likelihood
    # ==========

    def likelihood(self, z, hyperparam):
        """
        """

        sign_switch = False
        return DirectLikelihood.log_likelihood(z, self.X, self.S_mixed,
                                               sign_switch, hyperparam)

    # =======================
    # maximize log likelihood
    # =======================

    def maximize_log_likelihood(
            self,
            z,
            likelihood_method='direct',
            optimization_method='Newton-CG',
            plot=False):
        """
        """

        if likelihood_method == 'direct':
            results = DirectLikelihood.maximize_log_likelihood(
                    z, self.X, self.S_mixed, tol=1e-3,
                    hyperparam_guess=[0.1, 0.1],
                    optimization_method=optimization_method)

            # Plot log likelihood
            if plot:
                optimal_sigma = results['sigma']
                optimal_sigma0 = results['sigma0']
                optimal_hyperparam = [optimal_sigma, optimal_sigma0]
                DirectLikelihood.plot_log_likelihood(z, self.X, self.S_mixed,
                                                     optimal_hyperparam)

            # return ProfileLikelihood.maximize_log_likelihood_with_sigma_eta(
            #         z, self.X, self.K_mixed)

        elif likelihood_method == 'profiled':

            # Note: When using interpolation, make sure the interval below is
            # exactly the end points of eta_i, not less or more.
            interval_eta = [1e-4, 1e+3]

            # Find hyperparam
            results = ProfileLikelihood.find_log_likelihood_der1_zeros(
                    z, self.X, self.K_mixed, interval_eta)

            # Finding the maxima. This isn't neccessary and affects run time
            # results['max_lp'] = ProfileLikelihood.log_likelihood(
            #         z, self.X, self.K_mixed, False,
            #         [results['sigma'], results['eta']])

            # Plot first derivative of log likelihood
            if plot:
                optimal_eta = results['eta']
                ProfileLikelihood.plot_log_likelihood_der1_eta(
                        z, self.X, self.K, self.K_mixed, optimal_eta)

        return results
