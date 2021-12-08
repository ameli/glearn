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
import imate
from ._base_likelihood import BaseLikelihood


# ==================
# Profile Likelihood
# ==================

class ProfileLikelihood(BaseLikelihood):
    """
    Likelihood function that is profiled with respect to :math:`\\sigma`
    variable.
    """

    # Import plot-related methods of this class implemented in a separate file
    from ._profile_likelihood_plots import plot

    # ====
    # init
    # ====

    def __init__(self, mean, cov, z, log_hyperparam=True):
        """
        Initialization.
        """

        # Super class constructor sets self.z, self.X, self.cov, self.mixed_cor
        super().__init__(mean, cov, z)

        # The index in hyperparam array where scale starts. In this class,
        # hyperparam is of the form [eta, scale], hence, scale starts at index
        # 1.
        self.scale_index = 1

        # Configuration
        self.hyperparam_tol = 1e-8

        if log_hyperparam:
            self.use_log_eta = True
            self.use_log_scale = True
        else:
            self.use_log_eta = False
            self.use_log_scale = False

        # Determine to compute traceinv (only for some of inner computations of
        # derivatives w.r.t scale) using direct inversion of matrices or with
        # Hutchinson method (a stochastic method).
        if self.mixed_cor.imate_method in ['hutchinson', 'slq']:
            # Use Hutchinson method (note: SLQ method cannot be used).
            self.stochastic_traceinv = True
        else:
            # For the rest of methods (like eigenvalue, cholesky, etc),
            # compute traceinv directly by matrix inversion.
            self.stochastic_traceinv = False

        # Store ell, its Jacobian and Hessian.
        self.ell = None
        self.ell_jacobian = None
        self.ell_hessian = None

        # Store hyperparam used to compute ell, its Jacobian and Hessian.
        self.ell_hyperparam = None
        self.ell_jacobian_hyperparam = None
        self.ell_hessian_hyperparam = None

        # Internal settings
        self.max_eta = 1e+16
        self.min_eta = 1e-16

        # Interval variables that are shared between class methods
        self.Y = None
        self.Cinv = None
        self.C = None
        self.Mz = None
        self.MMz = None
        self.sigma2 = None
        self.sigma02 = None
        self.Kninv = None
        self.KnpKninv = None

        # Hyperparameter which the interval variables in the above were
        # computed based upon it.
        self.Y_C_Mz_hyperparam = None
        self.sigma_hyperparam = None
        self.MMz_hyperparam = None
        self.Kninv_KnpKninv_hyperparam = None

        # Asymptotic variables
        self.asym_poly = None
        self.asym_roots = None
        self.asym_C = None

    # ================
    # reset attributes
    # ================

    def reset_attributes(self):
        """
        Sets all internal attributes back to None. This is sometimes needed,
        for example when the profile_likelihood is called multiple times for
        different scale parameters, but the scale parameter is not a part of
        hyperparam. This happens in double_profile_likelihood (see
        _find_optimal_eta function in ``DoubleProfileLikelihood``). By
        resetting all the internal attributes, these variables will be computed
        with the new scale parameter instead of reading them from the stored
        attributes.
        """

        self.ell = None
        self.ell_jacobian = None
        self.ell_hessian = None

        self.ell_hyperparam = None
        self.ell_jacobian_hyperparam = None
        self.ell_hessian_hyperparam = None

        self.Y = None
        self.Cinv = None
        self.C = None
        self.Mz = None
        self.MMz = None
        self.sigma2 = None
        self.sigma02 = None
        self.Kninv = None
        self.KnpKninv = None

        self.Y_C_Mz_hyperparam = None
        self.sigma_hyperparam = None
        self.MMz_hyperparam = None
        self.Kninv_KnpKninv_hyperparam = None

    # =================
    # eta to hyperparam
    # =================

    def _eta_to_hyperparam(self, eta):
        """
        Sets hyperparam from eta. eta is always given as natural scale (i.e.,
        not as in log scale). If self.use_log_eta is True, hyperparam is set
        as log10 of eta, otherwise, just as eta.
        """

        # If logscale is used, output hyperparam is log of eta.
        if self.use_log_eta:
            hyperparam = numpy.log10(numpy.abs(eta))
        else:
            hyperparam = numpy.abs(eta)

        return hyperparam

    # =================
    # hyperparam to eta
    # =================

    def _hyperparam_to_eta(self, hyperparam):
        """
        Sets eta from hyperparam. If self.use_log_eta is True, hyperparam is
        the log10 of eta, hence, 10**hyperparam is set to eta. If
        self.use_log_eta is False, hyperparam is directly set to eta.
        """

        # Using only the first component (if an array is given)
        if numpy.isscalar(hyperparam):
            hyperparam_ = hyperparam
        else:
            hyperparam_ = hyperparam[0]

        # If logscale is used, input hyperparam is log of eta.
        if self.use_log_eta:
            eta = 10.0**hyperparam_
        else:
            eta = numpy.abs(hyperparam_)

        return eta

    # ===================
    # scale to hyperparam
    # ===================

    def _scale_to_hyperparam(self, scale):
        """
        Sets hyperparam from scale. scale is always given with no log-scale
        If self.use_log_scale is True, hyperparam is set as log10 of scale,
        otherwise, just as scale.
        """

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

        # If logscale is used, input hyperparam is log of the scale.
        if self.use_log_scale:
            scale = 10.0**hyperparam
        else:
            scale = numpy.abs(hyperparam)

        return scale

    # ============================
    # hyperparam to log hyperparam
    # ============================

    def hyperparam_to_log_hyperparam(self, hyperparam):
        """
        Converts the input hyperparameters to their log10, if this is enabled
        by ``self.use_log_eta`` and ``self.use_scale``.

        If is assumed that the input hyperparam is not in log scale, and it
        contains either of the following form:

        * [eta]
        * [eta, scale1, scale2, ...]
        """

        if numpy.isscalar(hyperparam):
            hyperparam_ = numpy.array([hyperparam], dtype=float)
        elif isinstance(hyperparam, list):
            hyperparam_ = numpy.array(hyperparam, dtype=float)
        else:
            # Copy to avoid overwriting input
            hyperparam_ = hyperparam.copy()

        # Convert eta to log10 of eta
        if self.use_log_eta:
            eta = hyperparam_[0]
            hyperparam_[0] = self._eta_to_hyperparam(eta)

        # Convert scale to log10 of scale
        if hyperparam_.size > self.scale_index:
            if self.use_log_scale:
                scale = hyperparam_[self.scale_index:]
                hyperparam_[self.scale_index:] = \
                    self._scale_to_hyperparam(scale)

        return hyperparam_

    # ==================
    # extract hyperparam
    # ==================

    def extract_hyperparam(self, hyperparam):
        """
        It is assumed the input hyperparam might be in the log10 scale, and
        may or may not contain scales. The output will be converted to non-log
        format and will include scale, regardless if the input has scale or
        not.
        """

        if numpy.isscalar(hyperparam):
            hyperparam_ = numpy.array([hyperparam], dtype=float)
        else:
            hyperparam_ = hyperparam

        eta = self._hyperparam_to_eta(hyperparam_[0])

        sigma, sigma0 = self._find_optimal_sigma_sigma0(hyperparam_)

        if hyperparam_.size > self.scale_index:
            scale = self._hyperparam_to_scale(hyperparam_[self.scale_index:])
        else:
            scale = self.mixed_cor.get_scale()

        return sigma, sigma0, eta, scale

    # =====
    # M dot
    # =====

    def M_dot(self, C, Y, eta, z):
        """
        Multiplies the matrix :math:`\\mathbf{M}_{1,\\eta}` by a given vector
        :math:`\\boldsymbol{z}`. The matrix :math:`\\mathbf{M}` is defined by

        .. math::

            \\mathbf{M}_{1, \\eta} = \\boldsymbol{\\K}_{\\eta}^{-1} \\mathbf{P}

        where the covariance matrix :math:`\\boldsymbol{\\Sigmna}` is defined
        by

        .. math::

            \\boldsymbol{\\Sigma} = \\sigma^2 \\mathbf{K} +
            \\sigma_0^2 \\mathbf{I},

        and the projection matrix :math:`\\mathbf{P}` is defined by

        .. math::

            \\mathbf{P} = \\mathbf{I} - \\mathbf{X} (\\mathbf{X}^{\\intercal}
            \\boldsymbol{\\Sigma}^{-1}) \\mathbf{X})^{-1}
            \\mathbf{X}^{\\intercal} \\boldsymbol{\\Sigma}^{-1}.

        :param cov: An object of class :class:`Covariance` which represents
            the operator :math:`\\sigma^2 \\mathbf{K} +
            \\sigma_0^2 \\mathbf{I}`.
        :type cov: glearn.Covariance

        :param Binv: The inverse of matrix
            :math:`\\mathbf{B} = \\mathbf{X}^{\\intercal} \\mathbf{Y}`.
        :type Binv: numpy.ndarray

        :param Y: The matrix
            :math:`\\mathbf{Y} = \\boldsymbol{\\Sigma}^{-1} \\mathbf{X}`.
        :type Y: numpy.ndarray

        :param sigma: The parameter :math:`\\sigma`.
        :type sigma: float

        :param sigma0: The parameter :math:`\\sigma_0`.
        :type sigma0: float

        :param z: The data column vector.
        :type z: numpy.ndarray
        """

        # Computing w = Sinv*z, where S is sigma**2 * K + sigma0**2 * I
        w = self.mixed_cor.solve(z, eta=eta)

        # Computing Mz
        Ytz = numpy.matmul(Y.T, z)
        CYtz = numpy.matmul(C, Ytz)
        YCYtz = numpy.matmul(Y, CYtz)
        Mz = w - YCYtz

        return Mz

    # =============
    # update Y C Mz
    # =============

    def _update_Y_C_Mz(self, hyperparam):
        """
        Computes Y, C, Cinv, and Mz. These variables are shared among many of
        the functions, hence their values are stored as the class attribute to
        avoid re-computation when the hyperparam is the same.
        """

        if numpy.isscalar(hyperparam):
            hyperparam_ = numpy.array([hyperparam], dtype=float)
        else:
            hyperparam_ = hyperparam

        # Check if likelihood is already computed for an identical hyperparam
        if (self.Y is None) or \
                (self.Cinv is None) or \
                (self.C is None) or \
                (self.Mz is None) or \
                (self.Y_C_Mz_hyperparam is None) or \
                (hyperparam_.size != self.Y_C_Mz_hyperparam.size) or \
                (not numpy.allclose(hyperparam_, self.Y_C_Mz_hyperparam,
                                    atol=self.hyperparam_tol)):

            # hyperparameters
            eta = self._hyperparam_to_eta(hyperparam_)

            # Include derivative w.r.t scale
            if (not numpy.isscalar(hyperparam_)) and \
                    (hyperparam_.size > self.scale_index):

                # Set scale of the covariance object
                scale = self._hyperparam_to_scale(
                        hyperparam_[self.scale_index:])
                self.mixed_cor.set_scale(scale)

            # Variables to compute/update. Note that in all variables, sigma
            # is factored out. For example, self.B is indeed B1 = B/(sigma**2),
            # and self.C here is C1 = C/(sigma**2). That is, B and C denoted in
            # this code are B1 and C1 of the notations used in the paper.
            self.Y = self.mixed_cor.solve(self.X, eta=eta)
            self.Cinv = numpy.matmul(self.X.T, self.Y) + self.Binv
            self.C = numpy.linalg.inv(self.Cinv)
            self.Mz = self.M_dot(self.C, self.Y, eta, self.z)

            # Update the current hyperparam
            self.Y_C_Mz_hyperparam = hyperparam_

    # =========================
    # find optimal sigma sigma0
    # =========================

    def _find_optimal_sigma_sigma0(self, hyperparam):
        """
        Based on a given eta, finds optimal sigma and sigma0.
        """

        # Get eta
        eta = self._hyperparam_to_eta(hyperparam)

        if numpy.abs(eta) > self.max_eta:

            # eta is very large. Use Asymptotic relation
            sigma02 = self._find_optimal_sigma02()

            if numpy.isinf(eta):
                sigma2 = 0.
            else:
                sigma2 = sigma02 / eta

        else:

            # Find sigma2
            sigma2 = self._find_optimal_sigma2(hyperparam)

            # Find sigma02
            if numpy.abs(eta) < self.min_eta:
                sigma02 = 0.0
            else:
                sigma02 = eta * sigma2

        sigma = numpy.sqrt(sigma2)
        sigma0 = numpy.sqrt(sigma02)

        return sigma, sigma0

    # ===================
    # find optimal sigma2
    # ===================

    def _find_optimal_sigma2(self, hyperparam):
        """
        Finds optimal sigma2 either if eta is large or not. As a product, this
        function also computes Y, C, and Mz and stores them into the class
        attribute.
        """

        # Get eta
        eta = self._hyperparam_to_eta(hyperparam)

        if numpy.isinf(eta):
            self.sigma2 = 0.0

        elif numpy.abs(eta) > self.max_eta:

            # eta is very large. Use Asymptotic relation
            sigma02 = self._find_optimal_sigma02()
            self.sigma2 = sigma02 / eta

        else:

            if numpy.isscalar(hyperparam):
                hyperparam_ = numpy.array([hyperparam], dtype=float)
            else:
                hyperparam_ = hyperparam

            # Check if sigma2 is already computed for an identical hyperparam
            if (self.sigma2 is None) or \
                    (self.sigma2_hyperparam is None) or \
                    (hyperparam_.size != self.sigma2_hyperparam.size) or \
                    (not numpy.allclose(hyperparam_, self.sigma2_hyperparam,
                                        atol=self.hyperparam_tol)):

                # Make sure Y, C, Mz are updated for the given hyperparam
                self._update_Y_C_Mz(hyperparam_)

                self.sigma2 = numpy.dot(self.z, self.Mz) / self.rdof

                # Update hyperparam
                self.sigma2_hyperparam = hyperparam_

        return self.sigma2

    # ====================
    # find optimal sigma02
    # ====================

    def _find_optimal_sigma02(self):
        """
        When eta is very large, we assume sigma is zero. Thus, sigma0 is
        computed by this function. This is the Ordinary Least Square (OLS)
        solution of the regression problem where we assume there is no
        correlation between points, hence sigma is assumed to be zero.

        This function does not require update of self.mixed_cor with
        hyperparameters.
        """

        # Note: this sigma0 is only when eta is at infinity. Hence, computing
        # it does not require eta, update of self.mixed_cor, or update of Y, C,
        # Mz. Hence, once it is computed, it can be reused even if other
        # variables like eta changed. Here, it suffice to only check of
        # self.sigma0 is None to compute it for the first time. On next calls,
        # it does not have to be recomputed.
        if self.sigma02 is None:

            if self.B is None:
                Cinv = numpy.matmul(self.X.T, self.X)
                C = numpy.linalg.inv(Cinv)
                Xtz = numpy.matmul(self.X.T, self.z)
                XCXtz = numpy.matmul(self.X, numpy.matmul(C, Xtz))
                self.sigma02 = numpy.dot(self.z, self.z-XCXtz) / self.rdof

            else:
                self.sigma02 = numpy.dot(self.z, self.z) / self.rdof

        return self.sigma02

    # ==========
    # update MMz
    # ==========

    def _update_MMz(self, hyperparam):
        """
        Computes MMz.
        """

        if numpy.isscalar(hyperparam):
            hyperparam_ = numpy.array([hyperparam], dtype=float)
        else:
            hyperparam_ = hyperparam

        # Check if likelihood is already computed for an identical hyperparam
        if (self.MMz is None) or \
                (hyperparam_.size != self.MMz_hyperparam.size) or \
                (not numpy.allclose(hyperparam_, self.MMz_hyperparam,
                                    atol=self.hyperparam_tol)):

            # Get eta
            eta = self._hyperparam_to_eta(hyperparam_)

            # Include derivative w.r.t scale
            if (not numpy.isscalar(hyperparam_)) and \
                    (hyperparam_.size > self.scale_index):

                # Set scale of the covariance object
                scale = self._hyperparam_to_scale(
                        hyperparam[self.scale_index:])
                self.mixed_cor.set_scale(scale)

            # Update Y, C, Mz
            self._update_Y_C_Mz(hyperparam_)

            # Compute M*M*z
            self.MMz = self.M_dot(self.C, self.Y, eta, self.Mz)

            # Update the current hyperparam
            self.MMz_hyperparam = hyperparam_

    # =====================
    # update Kninv KnpKninv
    # =====================

    def _update_Kninv_KnpKninv(self, hyperparam):
        """
        Compute Kninv, KnpKninv.
        """

        if numpy.isscalar(hyperparam):
            hyperparam_ = numpy.array([hyperparam], dtype=float)
        else:
            hyperparam_ = hyperparam

        # Check if likelihood is already computed for an identical hyperparam
        if (self.Kninv is None) or \
                (self.KnpKninv is None) or \
                (self.Kninv_KnpKninv_hyperparam is None) or \
                (hyperparam_.size != self.Kninv_KnpKninv_hyperparam.size) or \
                (not numpy.allclose(hyperparam_,
                                    self.Kninv_KnpKninv_hyperparam,
                                    atol=self.hyperparam_tol)):

            # Get eta
            eta = self._hyperparam_to_eta(hyperparam_)

            # Set scale of the covariance object
            scale = self._hyperparam_to_scale(hyperparam[self.scale_index:])
            self.mixed_cor.set_scale(scale)

            # Update Kninv
            Kn = self.mixed_cor.get_matrix(eta)
            self.Kninv = numpy.linalg.inv(Kn)

            # Initialize KnpKninv as list of size of scale.size
            self.KnpKninv = [None] * scale.size

            for p in range(scale.size):
                Knp = self.mixed_cor.get_matrix(eta, derivative=[p])
                self.KnpKninv[p] = Knp @ self.Kninv

            # Update the current hyperparam
            self.Kninv_KnpKninv_hyperparam = hyperparam_

    # ==========
    # Likelihood
    # ==========

    def likelihood(self, sign_switch, hyperparam):
        """
        Log likelihood function

            L = -(1/2) log det(S) - (1/2) log det(X.T*Sinv*X) -
                (1/2) sigma^(-2) * z.T * M1 * z

        where
            S = sigma^2 Kn is the covariance
            Sinv is the inverse of S
            M1 = Sinv = Sinv*X*(X.T*Sinv*X)^(-1)*X.T*Sinv

        hyperparam = [eta, scale[0], scale[1], ...]

        sign_switch changes the sign of the output from ell to -ell. When True,
        this is used to minimizing (instead of maximizing) the negative of
        log-likelihood function.
        """

        self.timer.tic()

        if numpy.isscalar(hyperparam):
            hyperparam_ = numpy.array([hyperparam], dtype=float)
        else:
            hyperparam_ = hyperparam

        # Check if likelihood is already computed for an identical hyperparam
        if (self.ell is not None) and \
                (self.ell_hyperparam is not None) and \
                (hyperparam_.size == self.ell_hyperparam.size) and \
                numpy.allclose(hyperparam_, self.ell_hyperparam,
                               atol=self.hyperparam_tol):

            if sign_switch:
                return -self.ell
            else:
                return self.ell

        # Get eta
        eta = self._hyperparam_to_eta(hyperparam_)

        # Extract scale from hyperparam
        if (not numpy.isscalar(hyperparam_)) and \
                (hyperparam_.size > self.scale_index):

            # Set scale of the covariance object
            scale = self._hyperparam_to_scale(hyperparam_[self.scale_index:])
            self.mixed_cor.set_scale(scale)

        if numpy.abs(eta) >= self.max_eta:

            # Optimal sigma02 when eta is very large
            sigma02 = self._find_optimal_sigma02()

            # Log likelihood
            ell = -0.5*self.rdof * (numpy.log(2.0*numpy.pi) + 1.0 +
                                    numpy.log(sigma02))

            if self.B is None:
                Cinv = numpy.matmul(self.X.T, self.X)
                logdet_Cinv = numpy.log(numpy.linalg.det(Cinv))
                ell += - 0.5*logdet_Cinv
            # else:
            #     logdet_B = numpy.log(numpy.linalg.det(self.B))
            #     ell += 0.5*logdet_B

        else:

            # Update Y, C, Mz (all needed for computing optimal sigma2)
            self._update_Y_C_Mz(hyperparam_)

            # Find (or update) optimal sigma2
            sigma2 = self._find_optimal_sigma2(hyperparam_)

            logdet_Kn = self.mixed_cor.logdet(eta)
            logdet_Cinv = numpy.log(numpy.linalg.det(self.Cinv))

            if numpy.isnan(logdet_Kn):
                raise RuntimeError('Logdet of "Kn" is nan at eta: %0.3e.'
                                   % eta)

            # Log likelihood
            ell = -0.5*self.rdof * \
                (numpy.log(2.0*numpy.pi) + 1.0 + numpy.log(sigma2)) \
                - 0.5*logdet_Kn - 0.5*logdet_Cinv

            if self.B is not None:
                # Note that self.B is indeed B1, that is the matrix B without
                # sigma**2.
                logdet_B = numpy.log(numpy.linalg.det(self.B))
                ell += -0.5*logdet_B

        # Store ell to member data (without sign-switch).
        self.ell = ell
        self.ell_hyperparam = hyperparam_

        # If ell is used in scipy.optimize.minimize, change the sign to obtain
        # the minimum of -ell
        if sign_switch:
            ell = -ell

        self.timer.toc()

        return ell

    # ===================
    # likelihood der1 eta
    # ===================

    def _likelihood_der1_eta(self, hyperparam):
        """
        ell is the log likelihood probability. dell_deta is d(ell)/d(eta),
        which is the derivative of ell with respect to eta when the optimal
        value of sigma is substituted in the likelihood function per given eta.
        """

        # Get eta
        eta = self._hyperparam_to_eta(hyperparam)

        # Include derivative w.r.t scale
        if (not numpy.isscalar(hyperparam)) and \
                (hyperparam.size > self.scale_index):

            # Set scale of the covariance object
            scale = self._hyperparam_to_scale(hyperparam[self.scale_index:])
            self.mixed_cor.set_scale(scale)

        # Update Y, C, Mz
        self._update_Y_C_Mz(hyperparam)

        # Find optimal sigma2
        sigma2 = self._find_optimal_sigma2(hyperparam)

        # Traces
        trace_Kninv = self.mixed_cor.traceinv(eta)
        YtY = numpy.matmul(self.Y.T, self.Y)
        trace_CYtY = numpy.trace(numpy.matmul(self.C, YtY))
        trace_M = trace_Kninv - trace_CYtY

        # Derivative of log likelihood
        zM2z = numpy.dot(self.Mz, self.Mz)
        dell_deta = -0.5*(trace_M - zM2z/sigma2)

        # Return as scalar or array of length one
        if numpy.isscalar(hyperparam):
            return dell_deta
        else:
            return numpy.array([dell_deta], dtype=float)

    # ===================
    # likelihood der2 eta
    # ===================

    def _likelihood_der2_eta(self, hyperparam):
        """
        The second derivative of ell is computed as a function of only eta.
        Here, we substituted optimal value of sigma, which is self is a
        function of eta.
        """

        # Get eta
        eta = self._hyperparam_to_eta(hyperparam)

        # Include derivative w.r.t scale
        if (not numpy.isscalar(hyperparam)) and \
                (hyperparam.size > self.scale_index):

            # Set scale of the covariance object
            scale = self._hyperparam_to_scale(hyperparam[self.scale_index:])
            self.mixed_cor.set_scale(scale)

        # Update Y, C, Mz
        self._update_Y_C_Mz(hyperparam)

        # Find optimal sigma2
        sigma2 = self._find_optimal_sigma2(hyperparam)

        V = self.mixed_cor.solve(self.Y, eta=eta)

        # Trace of M**2
        trace_Kn2inv = self.mixed_cor.traceinv(eta, exponent=2)
        YtY = numpy.matmul(self.Y.T, self.Y)
        YtV = numpy.matmul(self.Y.T, V)
        CYtV = numpy.matmul(self.C, YtV)
        trace_CYtV = numpy.trace(CYtV)
        A = numpy.matmul(self.C, YtY)
        AA = numpy.matmul(A, A)
        trace_AA = numpy.trace(AA)
        trace_M2 = trace_Kn2inv - 2.0*trace_CYtV + trace_AA

        # Compute (or update) MMz
        self._update_MMz(hyperparam)

        # Second derivative (only at the location of zero first derivative)
        zM2z = numpy.dot(self.Mz, self.Mz)
        zM3z = numpy.dot(self.Mz, self.MMz)
        # d2ell_deta2 = 0.5*(trace_M2 * zM2z - 2.0*trace_M * zM3z)

        # Warning: this relation is the second derivative only at optimal eta,
        # where the first derivative vanishes. It does not require the
        # computation of zM2z. However, for plotting, or using Hessian in
        # scipy.optimize.minimize, this formula must not be used, because it is
        # not the actual second derivative everywhere else other than optimal
        # point of eta.
        # d2ell_deta2 = (0.5/sigma2) * \
        #     ((trace_M2/self.rdof + (trace_M/(self.rdof)**2) * zMz - 2.0*zM3z)

        # This relation is the actual second derivative. Use this relation for
        # the Hessian in scipy.optimize.minimize.
        d2ell_deta2 = 0.5 * \
            (trace_M2 - 2.0*zM3z/sigma2 + zM2z**2/(self.rdof*sigma2**2))

        # Return as scalar or array of length one
        if numpy.isscalar(hyperparam):
            return d2ell_deta2
        else:
            return numpy.array([[d2ell_deta2]], dtype=float)

    # =====================
    # likelihood der1 scale
    # =====================

    def _likelihood_der1_scale(self, hyperparam):
        """
        ell is the log likelihood probability. ell_dscale is d(ell)/d(theta),
        is the derivative of ell with respect to the distance scale (theta).
        """

        # Get eta
        eta = self._hyperparam_to_eta(hyperparam)

        # Set scale of the covariance object
        scale = self._hyperparam_to_scale(hyperparam[self.scale_index:])
        self.mixed_cor.set_scale(scale)

        # Initialize jacobian
        dell_dscale = numpy.zeros((scale.size, ), dtype=float)

        # Update Y, C, Mz
        self._update_Y_C_Mz(hyperparam)

        # Find optimal sigma2
        sigma2 = self._find_optimal_sigma2(hyperparam)

        # Compute (or update) Kninv and KnpKninv
        if not self.stochastic_traceinv:
            self._update_Kninv_KnpKninv(hyperparam)

        # Knp is the derivative of mixed_cor (Kn) w.r.t p-th element of scale.
        for p in range(scale.size):

            if self.stochastic_traceinv:
                # Compute traceinv using stochastic estimation method. Note
                # that since Knp is not positive-definite, we cannot use
                # Cholesky method in imate. The only viable option is
                # Hutchinson's method.
                Knp = self.mixed_cor.get_matrix(eta, derivative=[p])
                trace_KnpKninv = self.mixed_cor.traceinv(
                        eta, B=Knp, imate_method='hutchinson')
            else:
                trace_KnpKninv, _ = imate.trace(self.KnpKninv[p],
                                                method='exact')

            # Compute the second component of trace of Knp * M
            KnpY = self.mixed_cor.dot(self.Y, eta=eta, derivative=[p])
            YtKnpY = numpy.matmul(self.Y.T, KnpY)
            CYtKnpY = numpy.matmul(self.C, YtKnpY)
            trace_CYtKnpY = numpy.trace(CYtKnpY)

            # Compute trace of Knp * M
            trace_KnpM = trace_KnpKninv - trace_CYtKnpY

            # Compute zMKnpMz
            KnpMz = self.mixed_cor.dot(self.Mz, eta=eta, derivative=[p])
            zMKnpMz = numpy.dot(self.Mz, KnpMz)

            # Derivative of ell w.r.t p-th element of distance scale
            dell_dscale[p] = -0.5*trace_KnpM + 0.5*zMKnpMz / sigma2

        return dell_dscale

    # =====================
    # likelihood der2 scale
    # =====================

    def _likelihood_der2_scale(self, hyperparam):
        """
        ell is the log likelihood probability. der2_scale is d2(ell)/d(theta2),
        is the second derivative of ell with respect to the distance scale
        (theta). The output is a 2D array of the size of scale.
        """

        # Get eta
        eta = self._hyperparam_to_eta(hyperparam)

        # Set scale of the covariance object
        scale = self._hyperparam_to_scale(hyperparam[self.scale_index:])
        self.mixed_cor.set_scale(scale)

        # Initialize Hessian
        d2ell_dscale2 = numpy.zeros((scale.size, scale.size), dtype=float)

        # Update Y, C, Mz
        self._update_Y_C_Mz(hyperparam)

        # Find optimal sigma2
        sigma2 = self._find_optimal_sigma2(hyperparam)

        # Compute (or update) Kninv and KnpKninv
        if not self.stochastic_traceinv:
            self._update_Kninv_KnpKninv(hyperparam)

        # Knp is the derivative of mixed_cor (Kn) w.r.t p-th element of scale.
        for p in range(scale.size):

            KnpMz = self.mixed_cor.dot(self.Mz, eta=eta, derivative=[p])
            MKnpMz = self.M_dot(self.C, self.Y, eta, KnpMz)

            for q in range(scale.size):

                # 1. Compute zMKnqMKnpMz
                if p == q:
                    KnqMz = KnpMz
                else:
                    KnqMz = self.mixed_cor.dot(self.Mz, eta=eta,
                                               derivative=[q])
                zMKnqMKnpMz = numpy.dot(KnqMz, MKnpMz)

                # 2. Compute zMKnpqMz
                KnpqMz = self.mixed_cor.dot(self.Mz, eta=eta,
                                            derivative=[p, q])
                zMKnpqMz = numpy.dot(self.Mz, KnpqMz)

                # 3. Computing trace of Knpq * M in three steps

                # Compute the first component of trace of Knpq * Kninv
                Knpq = self.mixed_cor.get_matrix(eta, derivative=[p, q])
                if self.stochastic_traceinv:
                    trace_KnpqKninv = self.mixed_cor.traceinv(
                            eta, B=Knpq, imate_method='hutchinson')
                else:
                    KnpqKninv = Knpq @ self.Kninv
                    trace_KnpqKninv, _ = imate.trace(KnpqKninv, method='exact')

                # Compute the second component of trace of Knpq * M
                KnpqY = self.mixed_cor.dot(self.Y, eta=eta, derivative=[p, q])
                YtKnpqY = numpy.matmul(self.Y.T, KnpqY)
                CYtKnpqY = numpy.matmul(self.C, YtKnpqY)
                trace_CYtKnpqY = numpy.trace(CYtKnpqY)

                # Compute trace of Knpq * M
                trace_KnpqM = trace_KnpqKninv - trace_CYtKnpqY

                # 4. Compute trace of Knp * M * Knq * M

                # Compute first part of trace of Knp * M * Knq * M
                Knp = self.mixed_cor.get_matrix(eta, derivative=[p])
                Knq = self.mixed_cor.get_matrix(eta, derivative=[q])
                if self.stochastic_traceinv:
                    trace_KnpMKnqM_1 = self.mixed_cor.traceinv(
                            eta, B=Knq, C=Knp, imate_method='hutchinson')
                else:
                    KnpKninvKnqKninv = numpy.matmul(self.KnpKninv[p],
                                                    self.KnpKninv[q])
                    trace_KnpMKnqM_1, _ = imate.trace(KnpKninvKnqKninv,
                                                      method='exact')

                # Compute the second part of trace of Knp * M * Knq * M
                KnpY = Knp @ self.Y
                if p == q:
                    KnqY = KnpY
                else:
                    KnqY = Knq @ self.Y
                KninvKnqY = self.mixed_cor.solve(KnqY, eta=eta)
                YtKnpKninvKnqY = numpy.matmul(KnpY.T, KninvKnqY)
                F21 = numpy.matmul(self.C, YtKnpKninvKnqY)
                F22 = numpy.matmul(self.C, YtKnpKninvKnqY.T)
                trace_KnpMKnqM_21 = numpy.trace(F21)
                trace_KnpMKnqM_22 = numpy.trace(F22)

                # Compute the third part of trace of Knp * M * Knq * M
                YtKnpY = numpy.matmul(self.Y.T, KnpY)
                if p == q:
                    YtKnqY = YtKnpY
                else:
                    YtKnqY = numpy.matmul(self.Y.T, KnqY)
                Dp = numpy.matmul(self.C, YtKnpY)
                if p == q:
                    Dq = Dp
                else:
                    Dq = numpy.matmul(self.C, YtKnqY)
                D = numpy.matmul(Dp, Dq)
                trace_KnpMKnqM_3 = numpy.trace(D)

                # Compute trace of Knp * M * Knq * M
                trace_KnpMKnqM = trace_KnpMKnqM_1 - trace_KnpMKnqM_21 - \
                    trace_KnpMKnqM_22 + trace_KnpMKnqM_3

                # 5. Second "local" derivatives w.r.t scale
                local_d2ell_dscale2 = -0.5*trace_KnpqM + 0.5*trace_KnpMKnqM + \
                    (0.5*zMKnpqMz - zMKnqMKnpMz) / sigma2

                # Computing total second derivative
                dp_log_sigma2 = -numpy.dot(self.Mz, KnpMz) / \
                    (self.rdof*sigma2)
                if p == q:
                    dq_log_sigma2 = dp_log_sigma2
                else:
                    dq_log_sigma2 = -numpy.dot(self.Mz, KnqMz) / \
                        (self.rdof*sigma2)
                d2ell_dscale2[p, q] = local_d2ell_dscale2 + \
                    0.5 * self.rdof * dp_log_sigma2 * dq_log_sigma2

                if p != q:
                    d2ell_dscale2[q, p] = d2ell_dscale2[p, q]

        return d2ell_dscale2

    # =====================
    # likelihood der2 mixed
    # =====================

    def _likelihood_der2_mixed(self, hyperparam):
        """
        ell is the log likelihood probability. d2ell_deta_dscale is the mixed
        second derivative w.r.t eta and scale. The output is a 1D vector of the
        size of scale.
        """

        # Get eta
        eta = self._hyperparam_to_eta(hyperparam)

        # Set scale of the covariance object
        scale = self._hyperparam_to_scale(hyperparam[self.scale_index:])
        self.mixed_cor.set_scale(scale)

        # Initialize mixed derivative as 2D array with one row.
        d2ell_deta_dscale = numpy.zeros((1, scale.size), dtype=float)

        # Update Y, C, Mz
        self._update_Y_C_Mz(hyperparam)

        # Find optimal sigma2
        sigma2 = self._find_optimal_sigma2(hyperparam)

        # Computing Y=Kninv*X.
        YtY = numpy.matmul(self.Y.T, self.Y)
        V = self.mixed_cor.solve(self.Y, eta=eta)

        # Compute (or update) MMz
        self._update_MMz(hyperparam)

        # Compute Mz and MMz
        zMMz = numpy.dot(self.Mz, self.Mz)

        # Compute (or update) Kninv and KnpKninv
        if not self.stochastic_traceinv:
            self._update_Kninv_KnpKninv(hyperparam)
            Kninv2 = self.Kninv @ self.Kninv

        # Common variables in the for loop below
        D2 = numpy.matmul(self.C, YtY)

        # Knp is the derivative of mixed_cor (Kn) w.r.t p-th element of scale.
        for p in range(scale.size):

            # Compute zMKnpMMz
            KnpMz = self.mixed_cor.dot(self.Mz, eta=eta, derivative=[p])
            zMKnpMz = numpy.dot(self.Mz, KnpMz)
            zMKnpMMz = numpy.dot(KnpMz, self.MMz)

            # Compute trace of KnpKninv2
            Knp = self.mixed_cor.get_matrix(eta, derivative=[p])
            if self.stochastic_traceinv:
                trace_KnpKninv2 = self.mixed_cor.traceinv(
                        eta, B=Knp, exponent=2, imate_method='hutchinson')
            else:
                KnpKninv2 = Knp @ Kninv2
                trace_KnpKninv2, _ = imate.trace(KnpKninv2, method='exact')

            # Compute traces
            KnpY = self.mixed_cor.dot(self.Y, eta=eta, derivative=[p])
            YtKnpY = numpy.matmul(self.Y.T, KnpY)
            VtKnpY = numpy.matmul(V.T, KnpY)
            F1 = numpy.matmul(self.C, VtKnpY)
            F2 = numpy.matmul(self.C, VtKnpY.T)
            D1 = numpy.matmul(self.C, YtKnpY)
            D = numpy.matmul(D1, D2)

            trace_F1 = numpy.trace(F1)
            trace_F2 = numpy.trace(F2)
            trace_D = numpy.trace(D)

            # Compute trace of M * Knp * M
            trace_MKnpM = trace_KnpKninv2 - trace_F1 - trace_F2 + trace_D

            # Compute mixed derivative
            local_d2ell_deta_dscale = 0.5*trace_MKnpM - zMKnpMMz / sigma2
            d2ell_deta_dscale[0, p] = local_d2ell_deta_dscale + \
                (0.5/(self.rdof*sigma2**2)) * zMMz * zMKnpMz

        return d2ell_deta_dscale

    # ===================
    # likelihood jacobian
    # ===================

    def likelihood_jacobian(self, sign_switch, hyperparam):
        """
        Computes Jacobian w.r.t eta, and if given, scale.
        """

        self.timer.tic()

        if numpy.isscalar(hyperparam):
            hyperparam_ = numpy.array([hyperparam], dtype=float)
        else:
            hyperparam_ = hyperparam

        # Check if Jacobian is already computed for an identical hyperparam
        if (self.ell_jacobian_hyperparam is not None) and \
                (self.ell_jacobian is not None) and \
                (hyperparam_.size == self.ell_jacobian_hyperparam.size) and \
                numpy.allclose(hyperparam_, self.ell_jacobian_hyperparam,
                               atol=self.hyperparam_tol):
            if sign_switch:
                return -self.ell_jacobian
            else:
                return self.ell_jacobian

        # Compute first derivative w.r.t eta
        dell_deta = self._likelihood_der1_eta(hyperparam)

        # Since we use xi = log_eta instead of eta as the variable, the
        # derivative of ell w.r.t log_eta should be taken into account.
        if self.use_log_eta:
            eta = self._hyperparam_to_eta(hyperparam)
            dell_deta = dell_deta * eta * numpy.log(10.0)

        jacobian = dell_deta

        # Compute Jacobian w.r.t scale
        if hyperparam_.size > self.scale_index:

            # Compute first derivative w.r.t scale
            dell_dscale = self._likelihood_der1_scale(hyperparam)

            # Convert derivative w.r.t log of scale
            if self.use_log_scale:
                scale = self._hyperparam_to_scale(
                        hyperparam_[self.scale_index:])
                for p in range(scale.size):
                    dell_dscale[p] = dell_dscale[p] * scale[p] * \
                        numpy.log(10.0)

            # Concatenate derivatives of eta and scale if needed
            jacobian = numpy.r_[dell_deta, dell_dscale]

        # Store Jacobian to member data (without sign-switch).
        self.ell_jacobian = jacobian
        self.ell_jacobian_hyperparam = hyperparam_

        if sign_switch:
            jacobian = -jacobian

        self.timer.toc()

        return jacobian

    # ==================
    # likelihood hessian
    # ==================

    def likelihood_hessian(self, sign_switch, hyperparam):
        """
        Computes Hessian w.r.t eta, and if given, scale.
        """

        self.timer.tic()

        if numpy.isscalar(hyperparam):
            hyperparam_ = numpy.array([hyperparam], dtype=float)
        else:
            hyperparam_ = hyperparam

        # Check if Hessian is already computed for an identical hyperparam
        if (self.ell_hessian_hyperparam is not None) and \
                (self.ell_hessian is not None) and \
                (hyperparam_.size == self.ell_hessian_hyperparam.size) and \
                numpy.allclose(hyperparam_, self.ell_hessian_hyperparam,
                               atol=self.hyperparam_tol):
            if sign_switch:
                return -self.ell_hessian
            else:
                return self.ell_hessian

        # Compute second derivative w.r.t eta
        d2ell_deta2 = self._likelihood_der2_eta(hyperparam)

        # To convert derivative to log scale, Jacobian is needed. Note: The
        # Jacobian itself is already converted to log scale.
        if self.use_log_eta or self.use_log_scale:
            jacobian_ = self.likelihood_jacobian(False, hyperparam)

        # Since we use xi = log_eta instead of eta as the variable, the
        # derivative of ell w.r.t log_eta should be taken into account.
        if self.use_log_eta:
            eta = self._hyperparam_to_eta(hyperparam)
            if numpy.isscalar(jacobian_):
                dell_deta = jacobian_
            else:
                dell_deta = jacobian_[0]

            # Convert second derivative to log scale (Note: dell_deta is
            # already in log scale)
            d2ell_deta2 = d2ell_deta2 * eta**2 * numpy.log(10.0)**2 + \
                dell_deta * numpy.log(10.0)

        # Hessian here is a 2D array of size 1.
        hessian = d2ell_deta2

        # Compute Hessian w.r.t scale
        if hyperparam_.size > self.scale_index:

            # Compute second derivative w.r.t scale
            d2ell_dscale2 = self._likelihood_der2_scale(hyperparam)

            # Convert derivative w.r.t log of scale (if needed)
            if self.use_log_scale:
                scale = self._hyperparam_to_scale(
                        hyperparam_[self.scale_index:])
                dell_dscale = jacobian_[self.scale_index:]

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

            # Compute second mixed derivative w.r.t scale and eta
            d2ell_deta_dscale = self._likelihood_der2_mixed(hyperparam)

            if self.use_log_eta:
                eta = self._hyperparam_to_eta(hyperparam)
                for p in range(scale.size):
                    d2ell_deta_dscale[0, p] = d2ell_deta_dscale[0, p] * \
                        eta * numpy.log(10.0)

            if self.use_log_scale:
                scale = self._hyperparam_to_scale(
                        hyperparam_[self.scale_index:])
                for p in range(scale.size):
                    d2ell_deta_dscale[0, p] = d2ell_deta_dscale[0, p] * \
                        scale[p] * numpy.log(10.0)

            # Concatenate derivatives to form Hessian of all variables
            hessian = numpy.block(
                    [[d2ell_deta2, d2ell_deta_dscale],
                     [d2ell_deta_dscale.T, d2ell_dscale2]])

        # Store hessian to member data (without sign-switch).
        self.ell_hessian = hessian
        self.ell_hessian_hyperparam = hyperparam_

        if sign_switch:
            hessian = -hessian

        self.timer.toc()

        return hessian

    # ===============
    # bounds der1 eta
    # ===============

    def bounds_der1_eta(self, eta):
        """
        Upper and lower bound of the first derivative of likelihood w.r.t eta.
        """

        # Get the smallest and largest eigenvalues of K
        eig_smallest, eig_largest = \
            self.mixed_cor.cor.get_extreme_eigenvalues()

        # upper bound
        dell_deta_upper_bound = 0.5*self.rdof * \
            (1.0/(eta+eig_smallest) - 1.0/(eta+eig_largest))

        # Lower bound
        dell_deta_lower_bound = -dell_deta_upper_bound

        return dell_deta_upper_bound, dell_deta_lower_bound

    # =====
    # Q dot
    # =====

    def Q_dot(self, z):
        """
        Matrix-vector multiplication Q*z where Q is defined by

            Q = I - X * (X.T * X)^{-1} * X.T

        where X is (n, m) matrix. If n is large, direct computation of Q is
        inefficient. Hence, an implicit matrix-vector operation is preferred.
        """

        Xtz = numpy.dot(self.X.T, z)
        CXtz = self.asym_C @ Xtz
        XCXtz = numpy.dot(self.X, CXtz)

        Qz = z - XCXtz

        return Qz

    # =====
    # N dot
    # =====

    def N_dot(self, z):
        """
        Matrix-vector multiplication N*z where N is defined by:

            N = K * Q
        """

        K = self.mixed_cor.get_matrix(0.0)
        Qz = self.Q_dot(z)
        Nz = K @ Qz

        return Nz

    # ===========================
    # asymptotic polynomial coeff
    # ===========================

    def _asymptotic_polynomial_coeff(self, degree=2):
        """
        Returns the asymptotic polynomial coefficients for the first derivative
        of likelihood w.r.t eta.

        If degree is 1 but the stored polynomial is of the second order (degree
        2), then it returns the first two entries of the polynomial coeffs.
        """

        if degree != 1 and degree != 2:
            raise ValueError('Asymptotic polynomial degree should be either ' +
                             '"1" or "2".')

        # Check if polynomial coeffs need to be computed
        if self.asym_poly is None or \
                (degree == 2 and self.asym_poly.size != 4):

            # asym_C is X.T*X
            if self.asym_C is None:
                asym_Cinv = self.X.T @ self.X
                self.asym_C = numpy.linalg.inv(asym_Cinv)

            Rz = self.Q_dot(self.z)
            zRz = numpy.dot(self.z, Rz)
            z_Rnorm = numpy.sqrt(zRz)
            zc = self.z / z_Rnorm

            # Powers of N
            Nzc = self.N_dot(zc)
            N2zc = self.N_dot(Nzc)
            if degree == 2:
                N3zc = self.N_dot(N2zc)
                N4zc = self.N_dot(N3zc)

            # Traces of N and N2
            K = self.mixed_cor.get_matrix(0.0)
            KX = K @ self.X
            XtKX = self.X.T @ KX
            XtK2X = KX.T @ KX
            D1 = self.asym_C @ XtKX
            D2 = self.asym_C @ XtK2X
            trace_K = self.mixed_cor.trace(eta=0, exponent=1)
            trace_K2 = self.mixed_cor.trace(eta=0, exponent=2)
            trace_N = trace_K - numpy.trace(self.asym_C @ XtKX)
            trace_N2 = trace_K2 - 2.0*numpy.trace(D2) + numpy.trace(D1 @ D1)

            # Normalized traces of N and N2
            mtrN = trace_N/self.rdof
            mtrN2 = trace_N2/self.rdof

            # Compute A0, A1, A2, A3
            A0zc = -self.Q_dot(mtrN*zc - Nzc)
            A1zc = self.Q_dot(mtrN*Nzc + mtrN2*zc - 2.0*N2zc)
            if degree == 2:
                A2zc = -self.Q_dot(mtrN*N2zc + mtrN2*Nzc - 2.0*N3zc)
                A3zc = self.Q_dot(mtrN2*N2zc - N4zc)

            # Coefficients
            a0 = numpy.dot(zc, A0zc)
            a1 = numpy.dot(zc, A1zc)
            if degree == 2:
                a2 = numpy.dot(zc, A2zc)
                a3 = numpy.dot(zc, A3zc)

            # Coefficients as array
            if degree == 1:
                self.asym_poly = numpy.array([a0, a1], dtype=float)
            else:
                self.asym_poly = numpy.array([a0, a1, a2, a3], dtype=float)

        if degree == 1:
            return self.asym_poly[:2]
        elif degree == 1:
            return self.asym_poly

    # =================
    # asymptotic maxima
    # =================

    def asymptotic_maxima(self, degree=2):
        """
        Approximates the maxima of the likelihood based on the zeros of the
        asymptotic relation of the first derivative of likelihood w.r.t eta.
        If the second derivative at the root is negative, the root is maxima.
        """

        if degree != 1 and degree != 2:
            raise ValueError('Asymptotic polynomial degree should be either ' +
                             '"1" or "2".')

        # Ensure asymptotes are calculated
        self._asymptotic_polynomial_coeff(degree=degree)

        # All roots
        if degree == 1:
            roots = numpy.roots(self.asym_poly[:2])
        else:
            roots = numpy.roots(self.asym_poly)

        # Remove complex roots
        roots = numpy.sort(numpy.real(
            roots[numpy.abs(numpy.imag(roots)) < 1e-10]))

        # Remove positive roots
        roots = roots[roots >= 0.0]

        # Output
        asym_maxima = []

        # Check sign of the second derivative
        for i in range(roots.size):
            asym_d2ell_deta2 = self._asymptotic_likelihood_der2_eta(
                    roots[i], degree=degree)
            if asym_d2ell_deta2 <= 0.0:
                asym_maxima.append(roots[i])

        return asym_maxima

    # ==============================
    # asymptotic likelihood der1 eta
    # ==============================

    def _asymptotic_likelihood_der1_eta(self, eta, degree=2):
        """
        Computes the asymptote of the likelihood first derivative w.r.t eta.
        """

        if degree != 1 and degree != 2:
            raise ValueError('Asymptotic polynomial degree should be either ' +
                             '"1" or "2".')

        # Ensure asymptotes are calculated
        self._asymptotic_polynomial_coeff(degree=degree)

        # Ensure array
        if numpy.isscalar(eta):
            eta_ = numpy.asarray([eta])
        else:
            eta_ = eta

        # Initialize output
        asym_dell_deta = numpy.empty(numpy.asarray(eta).size)

        for i in range(eta_.size):

            if degree == 1:
                asym_dell_deta[i] = (-0.5*self.rdof) * \
                        (self.asym_poly[0] +
                         self.asym_poly[1]/eta_[i]) / \
                        eta_[i]**2

            elif degree == 2:
                asym_dell_deta[i] = (-0.5*self.rdof) * \
                    (self.asym_poly[0] +
                     self.asym_poly[1]/eta_[i] +
                     self.asym_poly[2]/eta_[i]**2 +
                     self.asym_poly[3]/eta_[i]**3) / \
                    eta_[i]**2

        if numpy.isscalar(eta):
            return asym_dell_deta[0]
        else:
            return asym_dell_deta

    # ==============================
    # asymptotic likelihood der2 eta
    # ==============================

    def _asymptotic_likelihood_der2_eta(self, eta, degree=2):
        """
        Computes the asymptote of the likelihood second derivative w.r.t eta.
        """

        if degree != 1 and degree != 2:
            raise ValueError('Asymptotic polynomial degree should be either ' +
                             '"1" or "2".')

        # Ensure asymptotes are calculated
        self._asymptotic_polynomial_coeff(degree=degree)

        # Ensure array
        if numpy.isscalar(eta):
            eta_ = numpy.asarray([eta])
        else:
            eta_ = eta

        # Initialize output
        asym_d2ell_deta2 = numpy.empty(numpy.asarray(eta).size)

        for i in range(eta_.size):

            if degree == 1:
                asym_d2ell_deta2[i] = (0.5*self.rdof) * \
                        (2.0*self.asym_poly[0] +
                         3.0*self.asym_poly[1]/eta_[i]) / \
                        eta_[i]**3

            elif degree == 2:
                asym_d2ell_deta2[i] = (0.5*self.rdof) * \
                    (2.0*self.asym_poly[0] +
                     3.0*self.asym_poly[1]/eta_[i] +
                     4.0*self.asym_poly[2]/eta_[i]**2 +
                     5.0*self.asym_poly[3]/eta_[i]**3) / \
                    eta_[i]**3

        if numpy.isscalar(eta):
            return asym_d2ell_deta2[0]
        else:
            return asym_d2ell_deta2
