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

import time
import numpy
import scipy
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import scipy.optimize
from functools import partial
from ._base_likelihood import BaseLikelihood
from ._profile_likelihood import ProfileLikelihood


# =========================
# Double Profile Likelihood
# =========================

class DoubleProfileLikelihood(BaseLikelihood):
    """
    Likelihood function that is profiled with respect to :math:`\\sigma` and
    :math:`\\eta` variables.
    """

    # Import plot-related methods of this class implemented in a separate file
    from ._double_profile_likelihood_plots import plot

    # ====
    # init
    # ====

    def __init__(self, mean, cov, z, log_hyperparam=True):
        """
        Initialization
        """

        # Super class constructor sets self.z, self.X, self.cov, self.mixed_cor
        super().__init__(mean, cov, z)

        # Attributes
        self.profile_likelihood = ProfileLikelihood(mean, cov, z,
                                                    log_hyperparam)

        # The index in hyperparam array where scale starts. In this class,
        # hyperparam is of the form [scale], hence, scale starts at index 0.
        self.scale_index = 0

        # Configuration
        self.hyperparam_tol = 1e-8

        if log_hyperparam:
            self.use_log_scale = True
        else:
            self.use_log_scale = False

        # Store ell, its Jacobian and Hessian.
        self.optimal_eta = None
        self.ell = None
        self.ell_jacobian = None
        self.ell_hessian = None

        # Store hyperparam used to compute ell, its Jacobian and Hessian.
        self.optimal_eta_hyperparam = None
        self.ell_hyperparam = None
        self.ell_jacobian_hyperparam = None
        self.ell_hessian_hyperparam = None

        # Optimization method used to find optimal eta in profile_likelihood
        # self.optimization_method = 'chandrupatla'  # needs jac
        # self.optimization_method = 'Nelder-Mead'   # needs func
        # self.optimization_method = 'BFGS'          # needs func, jac
        # self.optimization_method = 'CG'            # needs func, jac
        self.optimization_method = 'Newton-CG'       # needs func, jac, hess
        # self.optimization_method = 'dogleg'        # needs func, jac, hess
        # self.optimization_method = 'trust-exact'   # needs func, jac, hess
        # self.optimization_method = 'trust-ncg'     # needs func, jac, hess

    # ===================
    # scale to hyperparam
    # ===================

    def _scale_to_hyperparam(self, scale):
        """
        Sets hyperparam from scale. scale is always given with no log-scale
        If self.use_log_eta is True, hyperparam is set as log10 of scale,
        otherwise, just as scale.
        """

        if numpy.isscalar(scale):
            scale = numpy.array([scale], dtype=float)
        if isinstance(scale, list):
            scale = numpy.array(scale, dtype=float)

        # If logscale is used, output hyperparam is log of scale.
        if self.use_log_scale:
            hyperparam = numpy.log10(numpy.abs(scale))
        else:
            hyperparam = numpy.abs(scale)

        return hyperparam

    # ===================
    # hyperparam to scale
    # ===================

    def _hyperparam_to_scale(self, hyperparam):
        """
        Sets scale from hyperparam. If self.use_log_eta is True, hyperparam is
        the log10 of scale, hence, 10**hyperparam is set to scale. If
        self.use_log_eta is False, hyperparam is directly set to scale.
        """

        if numpy.isscalar(hyperparam):
            hyperparam = numpy.array([hyperparam], dtype=float)
        elif isinstance(hyperparam, list):
            hyperparam = numpy.array(hyperparam, ftype=float)

        # If logscale is used, input hyperparam is log of the scale.
        if self.use_log_scale:
            scale = 10.0**hyperparam
        else:
            scale = numpy.abs(hyperparam)

        return scale

    # ================
    # find optimal eta
    # ================

    def _find_optimal_eta(
            self,
            scale,
            eta_guess=1.0,
            optimization_method=None):
        """
        Finds optimal eta to profile it out of the log-likelihood.
        """

        if optimization_method is None:
            optimization_method = self.optimization_method

        # Convert scale to hyperparam inorder to compare with the current
        # hyperparam of this class.
        hyperparam = self._scale_to_hyperparam(scale)

        # Check if likelihood is already computed for an identical hyperparam
        if (self.optimal_eta is not None) and \
                (self.optimal_eta_hyperparam is not None) and \
                (hyperparam.size == self.optimal_eta_hyperparam.size) and \
                numpy.allclose(hyperparam, self.optimal_eta_hyperparam,
                               atol=self.hyperparam_tol):
            return self.optimal_eta

        self.cov.set_scale(scale)

        # min_eta_guess = numpy.min([1e-4, eta_guess * 1e-2])
        # max_eta_guess = numpy.max([1e+3, eta_guess * 1e+2])
        # interval_eta = [min_eta_guess, max_eta_guess]
        #
        # # Using root finding method on the first derivative w.r.t eta
        # result = self.profile_likelihood.find_likelihood_der1_zeros(
        #         interval_eta)
        # eta = result['hyperparam']['eta']

        # Reset attributes of profile_likelihood object since scale has been
        # changed, however, scale is not in the hyperparam of this object,
        # hence, when it tries to request likelihood, or its Jacobian or
        # Hessian, it could returns its last computed value withput computing
        # them for the new scale.
        self.profile_likelihood.reset_attributes()

        result = self.profile_likelihood.maximize_likelihood(
                tol=1e-3, hyperparam_guess=[eta_guess],
                optimization_method=optimization_method)

        eta = result['hyperparam']['eta']

        # Store optimal eta to member data
        self.optimal_eta = eta
        self.optimal_eta_hyperparam = hyperparam

        return eta

    # ==========
    # likelihood
    # ==========

    def likelihood(
            self,
            sign_switch,
            eta_guess,
            hyperparam):
        """
        Variable eta is profiled out, meaning that optimal value of eta is
        used in log-likelihood function.
        """

        # Check if likelihood is already computed for an identical hyperparam
        if (self.ell_hyperparam is not None) and \
                (self.ell is not None) and \
                (hyperparam.size == self.ell_hyperparam.size) and \
                numpy.allclose(hyperparam, self.ell_hyperparam,
                               atol=self.hyperparam_tol):
            if sign_switch:
                return -self.ell
            else:
                return self.ell

        # Here, hyperparam consists of only scale, but not eta.
        scale = self._hyperparam_to_scale(hyperparam)
        self.cov.set_scale(scale)

        # Find optimal eta
        eta = self._find_optimal_eta(scale, eta_guess)

        # Convert eta to log of eta (if necessary). That is, hyperparam_eta
        # can be either equal to eta, or log10 of eta.
        hyperparam_eta = self.profile_likelihood._eta_to_hyperparam(eta)
        hyperparam_scale = self.profile_likelihood._scale_to_hyperparam(scale)

        # Construct new hyperparam that consists of both eta and scale
        hyperparam_full = numpy.r_[hyperparam_eta, hyperparam_scale]

        # Finding the maxima
        ell = self.profile_likelihood.likelihood(sign_switch, hyperparam_full)

        # Store ell to member data (without sign-switch).
        self.ell = ell
        self.ell_hyperparam = hyperparam

        return ell

    # ===================
    # likelihood jacobian
    # ===================

    def likelihood_jacobian(
            self,
            sign_switch,
            eta_guess,
            hyperparam):
        """
        Computes Jacobian w.r.t eta, and if given, scale.
        """

        # Check if Jacobian is already computed for an identical hyperparam
        if (self.ell_jacobian_hyperparam is not None) and \
                (self.ell_jacobian is not None) and \
                (hyperparam.size == self.ell_jacobian_hyperparam.size) and \
                numpy.allclose(hyperparam, self.ell_jacobian_hyperparam,
                               atol=self.hyperparam_tol):
            if sign_switch:
                return -self.ell_jacobian
            else:
                return self.ell_jacobian

        # When profiling eta is enabled, derivative w.r.t eta is not needed.
        # Compute only Jacobian w.r.t scale. Also, here, the input hyperparam
        # consists of only scale (and not eta).
        scale = self._hyperparam_to_scale(hyperparam)
        self.cov.set_scale(scale)

        # Find optimal eta
        eta = self._find_optimal_eta(scale, eta_guess)

        # Convert eta to log of eta (if necessary). That is, hyperparam_eta
        # can be either equal to eta, or log10 of eta.
        hyperparam_eta = self.profile_likelihood._eta_to_hyperparam(eta)
        hyperparam_scale = self.profile_likelihood._scale_to_hyperparam(scale)

        # Construct new hyperparam that consists of both eta and scale
        hyperparam_full = numpy.r_[hyperparam_eta, hyperparam_scale]

        # Compute first derivative w.r.t scale
        dell_dscale = self.profile_likelihood._likelihood_der1_scale(
                hyperparam_full)

        # Convert derivative w.r.t log of scale
        if self.use_log_scale:
            for p in range(scale.size):
                dell_dscale[p] = dell_dscale[p] * scale[p] * \
                    numpy.log(10.0)

        # Jacobian only consists of the derivative w.r.t scale
        jacobian = dell_dscale

        # Store jacobian to member data (without sign-switch).
        self.ell_jacobian = jacobian
        self.ell_jacobian_hyperparam = hyperparam

        if sign_switch:
            jacobian = -jacobian

        return jacobian

    # ==================
    # likelihood hessian
    # ==================

    def likelihood_hessian(self, sign_switch, eta_guess, hyperparam):
        """
        Computes Hessian w.r.t eta, and if given, scale.
        """

        # Check if Hessian is already computed for an identical hyperparam
        if (self.ell_hessian_hyperparam is not None) and \
                (self.ell_hessian is not None) and \
                (hyperparam.size == self.ell_hessian_hyperparam.size) and \
                numpy.allclose(hyperparam, self.ell_hessian_hyperparam,
                               atol=self.hyperparam_tol):
            if sign_switch:
                return -self.ell_hessian
            else:
                return self.ell_hessian

        # When profiling eta is enabled, derivative w.r.t eta is not needed.
        # Compute only Jacobian w.r.t scale. Also, here, the input hyperparam
        # consists of only scale (and not eta).
        if isinstance(hyperparam, list):
            hyperparam = numpy.array(hyperparam)
        scale = self._hyperparam_to_scale(hyperparam)
        self.cov.set_scale(scale)

        # Find optimal eta
        eta = self._find_optimal_eta(scale, eta_guess)

        # Convert eta to log of eta (if necessary). That is, hyperparam_eta
        # can be either equal to eta, or log10 of eta.
        hyperparam_eta = self.profile_likelihood._eta_to_hyperparam(eta)
        hyperparam_scale = self.profile_likelihood._scale_to_hyperparam(scale)

        # Construct new hyperparam that consists of both eta and scale
        hyperparam_full = numpy.r_[hyperparam_eta, hyperparam_scale]

        # Compute second derivative w.r.t scale
        d2ell_dscale2 = self.profile_likelihood._likelihood_der2_scale(
                hyperparam_full)

        if self.use_log_scale:

            # To convert derivative to log scale, Jacobian is needed. Note: The
            # Jacobian itself is already converted to log scale.
            jacobian_ = self.likelihood_jacobian(False, eta_guess, hyperparam)
            dell_dscale = jacobian_

            for p in range(scale.size):
                for q in range(scale.size):
                    if p == q:

                        # dell_dscale is already converted to logscale
                        d2ell_dscale2[p, q] = d2ell_dscale2[p, q] * \
                            scale[p]**2 * (numpy.log(10.0)**2) + \
                            dell_dscale[p] * numpy.log(10.0)
                    else:
                        d2ell_dscale2[p, q] = d2ell_dscale2[p, q] * \
                            scale[p] * scale[q] * (numpy.log(10.0)**2)

        # Concatenate derivatives to form Hessian of all variables
        hessian = d2ell_dscale2

        # Store hessian to member data (without sign-switch).
        self.ell_hessian = hessian
        self.ell_hessian_hyperparam = hyperparam

        if sign_switch:
            hessian = -hessian

        return hessian

    # ===================
    # maximize likelihood
    # ===================

    def maximize_likelihood(
            self,
            tol=1e-3,
            hyperparam_guess=[0.1, 0.1],
            optimization_method='Nelder-Mead',
            verbose=False):
        """
        Maximizing the log-likelihood function over the space of parameters
        sigma and sigma0

        In this function, hyperparam = [sigma, sigma0].
        """

        # Keeping times
        initial_wall_time = time.time()
        initial_proc_time = time.process_time()

        # When profile eta is used, hyperparam only contains scale
        eta_guess = 1.0

        # Partial function of likelihood with profiled eta. The input
        # hyperparam is only scale, not eta.
        sign_switch = True
        likelihood_partial_func = partial(self.likelihood, sign_switch,
                                          eta_guess)

        # Partial function of Jacobian of likelihood (with minus sign)
        jacobian_partial_func = partial(self.likelihood_jacobian, sign_switch,
                                        eta_guess)

        # Partial function of Hessian of likelihood (with minus sign)
        # Note: In the double profile method, the Hessian is not computed
        # properly since the current implementation only computes the "local"
        # second derivative, not the total second derivative which takes into
        # account of variation of \hat{\eta}(\theta).
        # hessian_partial_func = partial(self.likelihood_hessian, sign_switch,
        #                                eta_guess)
        hessian_partial_func = None

        # Minimize
        res = scipy.optimize.minimize(
                likelihood_partial_func, hyperparam_guess,
                method=optimization_method, tol=tol, jac=jacobian_partial_func,
                hess=hessian_partial_func)

        # Get the optimal scale
        scale = self._hyperparam_to_scale(res.x)

        # Find optimal eta with the given scale
        eta = self._find_optimal_eta(scale, eta_guess)

        # Find optimal sigma and sigma0 with the optimal eta
        sigma, sigma0 = self.profile_likelihood._find_optimal_sigma_sigma0(eta)
        max_ell = -res.fun

        # Adding time to the results
        wall_time = time.time() - initial_wall_time
        proc_time = time.process_time() - initial_proc_time

        # Output dictionary
        result = {
            'hyperparam':
            {
                'sigma': sigma,
                'sigma0': sigma0,
                'eta': eta,
                'scale': scale,
            },
            'optimization':
            {
                'max_likelihood': max_ell,
                'iter': res.nit,
            },
            'time':
            {
                'wall_time': wall_time,
                'proc_time': proc_time
            }
        }

        return result
