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
import scipy.optimize
from functools import partial
from .._utilities.plot_utilities import *                    # noqa: F401, F403
from .._utilities.plot_utilities import load_plot_settings, plt, matplotlib, \
        make_axes_locatable
import imate


# ===============
# Full Likelihood
# ===============

class FullLikelihood(object):

    # ====
    # init
    # ====

    def __init__(self, z, X, cov):
        """
        Initialization.
        """

        # Attributes
        self.z = z
        self.X = X
        self.cov = cov

        # Configuration
        self.hyperparam_tol = 1e-8
        self.use_log_scale = True

        # Store ell, its Jacobian and Hessian.
        self.ell = None
        self.ell_jacobian = None
        self.ell_hessian = None

        # Store hyperparam used to compute ell, its Jacobian and Hessian.
        self.ell_hyperparam = None
        self.ell_jacobian_hyperparam = None
        self.ell_hessian_hyperparam = None

    # ===================
    # scale to hyperparam
    # ===================

    def _scale_to_hyperparam(self, scale):
        """
        Sets hyperparam from scale. scale is always given with no log-scale
        If self.use_log_eta is True, hyperparam is set as log10 of scale,
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

    # =====
    # M dot
    # =====

    def M_dot(self, Binv, Y, sigma, sigma0, z):
        """
        Multiplies the matrix :math:`\\mathbf{M}` by a given vector
        :math:`\\boldsymbol{z}`. The matrix :math:`\\mathbf{M}` is defined by

        .. math::

            \\mathbf{M} = \\boldsymbol{\\Sigma}^{-1} \\mathbf{P},

        where the covarance matrix :math:`\\boldsymbol{\\Sigmna}` is defined by

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
        :type cov: gaussian_proc.Covariance

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
        w = self.cov.solve(sigma, sigma0, z)

        # Computing Mz
        Ytz = numpy.matmul(Y.T, z)
        Binv_Ytz = numpy.matmul(Binv, Ytz)
        Y_Binv_Ytz = numpy.matmul(Y, Binv_Ytz)
        Mz = w - Y_Binv_Ytz

        return Mz

    # ==========
    # likelihood
    # ==========

    def likelihood(self, sign_switch, hyperparam):
        """
        Here we use direct parameter, sigma and sigma0

        sign_switch change s the sign of the output from ell to -ell. When
        True, this is used to minimizing (instead of maximizing) the negative
        of log-likelihood function.
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

        # hyperparameters
        sigma = hyperparam[0]
        sigma0 = hyperparam[1]

        # Include derivative w.r.t scale
        if hyperparam.size > 2:
            scale = self._hyperparam_to_scale(hyperparam[2:])
            self.cov.set_scale(scale)

        n, m = self.X.shape

        # cov is the (sigma**2) * K + (sigma0**2) * I
        logdet_S = self.cov.logdet(sigma, sigma0)
        Y = self.cov.solve(sigma, sigma0, self.X)

        # Compute zMz
        B = numpy.matmul(self.X.T, Y)
        Binv = numpy.linalg.inv(B)
        Mz = self.M_dot(Binv, Y, sigma, sigma0, self.z)
        zMz = numpy.dot(self.z, Mz)

        # Compute log det (X.T*Sinv*X)
        logdet_B = numpy.log(numpy.linalg.det(B))

        # Log likelihood
        ell = -0.5*(n-m)*numpy.log(2.0*numpy.pi) - 0.5*logdet_S \
            - 0.5*logdet_B - 0.5*zMz

        # Store ell to member data (without sign-switch).
        self.ell = ell
        self.ell_hyperparam = hyperparam

        # If ell is used in scipy.optimize.minimize, change the sign to obtain
        # the minimum of -ell
        if sign_switch:
            ell = -ell

        return ell

    # ===================
    # likelihood jacobian
    # ===================

    def likelihood_jacobian(self, sign_switch, hyperparam):
        """
        When both :math:`\\sigma` and :math:`\\sigma_0` are zero, jacobian is
        undefined.
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

        # hyperparameters
        sigma = hyperparam[0]
        sigma0 = hyperparam[1]

        # Include derivative w.r.t scale
        if hyperparam.size > 2:
            scale = self._hyperparam_to_scale(hyperparam[2:])
            self.cov.set_scale(scale)

        n, m = self.X.shape

        # Computing Y=Sinv*X and w=Sinv*z.
        Y = self.cov.solve(sigma, sigma0, self.X)

        # B is Xt * Y
        B = numpy.matmul(self.X.T, Y)
        Binv = numpy.linalg.inv(B)

        # Compute Mz
        Mz = self.M_dot(Binv, Y, sigma, sigma0, self.z)

        # Compute KMz (Setting sigma=1 and sigma0=0 to have cov = K)
        KMz = self.cov.dot(1.0, 0.0, Mz)

        # Compute zMMz and zMKMz
        zMMz = numpy.dot(Mz, Mz)
        zMKMz = numpy.dot(Mz, KMz)

        # Compute trace of M
        if numpy.abs(sigma) < self.cov.tol:
            trace_M = (n - m) / sigma0**2
        else:
            trace_Sinv = self.cov.traceinv(sigma, sigma0)
            YtY = numpy.matmul(Y.T, Y)
            trace_BinvYtY = numpy.trace(numpy.matmul(Binv, YtY))
            trace_M = trace_Sinv - trace_BinvYtY

        # Compute trace of KM which is (n-m)/sigma**2 - eta* trace(M)
        if numpy.abs(sigma) < self.cov.tol:
            YtKY = numpy.matmul(Y.T, self.cov.dot(1.0, 0.0, Y))
            BinvYtKY = numpy.matmul(Binv, YtKY)
            trace_BinvYtKY = numpy.trace(BinvYtKY)
            trace_KM = n/sigma0**2 - trace_BinvYtKY
        else:
            eta = (sigma0 / sigma)**2
            trace_KM = (n - m)/sigma**2 - eta*trace_M

        # Derivative of ell wrt to sigma
        dell_dsigma = -0.5*trace_KM + 0.5*zMKMz
        dell_dsigma0 = -0.5*trace_M + 0.5*zMMz

        jacobian = numpy.array([dell_dsigma, dell_dsigma0], dtype=float)

        # Compute Jacobian w.r.t scale
        if hyperparam.size > 2:

            dell_dscale = numpy.zeros((scale.size, ), dtype=float)

            # Needed to compute trace (TODO)
            S = self.cov.get_matrix(sigma, sigma0)
            Sinv = numpy.linalg.inv(S)

            # Sp is the derivative of cov w.r.t the p-th element of scale.
            for p in range(scale.size):

                # Compute zMSpMz
                SpMz = self.cov.dot(sigma, sigma0, Mz, derivative=[p])
                zMSpMz = numpy.dot(Mz, SpMz)

                # Compute the first component of trace of Sp * Sinv (TODO)
                Sp = self.cov.get_matrix(sigma, sigma0, derivative=[p])
                SpSinv = Sp @ Sinv
                trace_SpSinv, _ = imate.trace(SpSinv, method='exact')

                # Compute the second component of trace of Sp * M
                SpY = self.cov.dot(sigma, sigma0, Y, derivative=[p])
                YtSpY = numpy.matmul(Y.T, SpY)
                BinvYtSpY = numpy.matmul(Binv, YtSpY)
                trace_BinvYtSpY = numpy.trace(BinvYtSpY)

                # Compute trace of Sp * M
                trace_SpM = trace_SpSinv - trace_BinvYtSpY

                # Derivative of ell w.r.t p-th element of distance scale
                dell_dscale[p] = -0.5*trace_SpM + 0.5*zMSpMz

            # Convert derivative w.r.t log of scale
            if self.use_log_scale:
                scale = self._hyperparam_to_scale(hyperparam[2:])
                dell_dscale = numpy.multiply(dell_dscale, scale) * \
                    numpy.log(10.0)

            # Concatenate jacobian
            jacobian = numpy.r_[jacobian, dell_dscale]

        # Store jacobian to member data (without sign-switch).
        self.ell_jacobian = jacobian
        self.ell_jacobian_hyperparam = hyperparam

        if sign_switch:
            jacobian = -jacobian

        return jacobian

    # ==================
    # likelihood hessian
    # ==================

    def likelihood_hessian(self, sign_switch, hyperparam):
        """
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

        # hyperparameters
        sigma = hyperparam[0]
        sigma0 = hyperparam[1]
        eta = (sigma0 / sigma)**2

        # Include derivatove w.r.t scale
        if hyperparam.size > 2:
            scale = self._hyperparam_to_scale(hyperparam[2:])
            self.cov.set_scale(scale)

        n, m = self.X.shape

        # -----------------------------------------
        # Second derivatives w.r.t sigma and sigma0
        # -----------------------------------------

        # Computing Y=Sinv*X, V = Sinv*Y, and w=Sinv*z
        Y = self.cov.solve(sigma, sigma0, self.X)
        V = self.cov.solve(sigma, sigma0, Y)

        # B is Xt * Y
        B = numpy.matmul(self.X.T, Y)
        Binv = numpy.linalg.inv(B)
        YtY = numpy.matmul(Y.T, Y)
        A = numpy.matmul(Binv, YtY)

        # Compute Mz, MMz
        Mz = self.M_dot(Binv, Y, sigma, sigma0, self.z)
        MMz = self.M_dot(Binv, Y, sigma, sigma0, Mz)

        # Compute KMz, zMMMz (Setting sigma=1 and sigma0=0 to have cov=K)
        KMz = self.cov.dot(1.0, 0.0, Mz)
        zMMMz = numpy.dot(Mz, MMz)

        # Compute MKMz
        MKMz = self.M_dot(Binv, Y, sigma, sigma0, KMz)

        # Compute zMKMKMz
        zMMKMz = numpy.dot(MMz, KMz)
        zMKMKMz = numpy.dot(KMz, MKMz)

        # Trace of M
        if numpy.abs(sigma) < self.cov.tol:
            trace_M = (n - m) / sigma0**2
        else:
            trace_Sinv = self.cov.traceinv(sigma, sigma0)
            trace_A = numpy.trace(A)
            trace_M = trace_Sinv - trace_A

        # Trace of Sinv**2
        trace_S2inv = self.cov.traceinv(sigma, sigma0, exponent=2)

        # Trace of M**2
        YtV = numpy.matmul(Y.T, V)
        C = numpy.matmul(Binv, YtV)
        trace_C = numpy.trace(C)
        AA = numpy.matmul(A, A)
        trace_AA = numpy.trace(AA)
        trace_M2 = trace_S2inv - 2.0*trace_C + trace_AA

        # Trace of (KM)**2
        if numpy.abs(sigma) < self.cov.tol:
            trace_K2 = self.cov.trace(1.0, 0.0, exponent=2)
            D = numpy.matmul(self.X.T, self.X)
            Dinv = numpy.linalg.inv(D)
            KX = self.cov.dot(1.0, 0.0, self.X, exponent=1)
            XKX = numpy.matmul(self.X.T, KX)
            XK2X = numpy.matmul(KX.T, KX)
            E = numpy.matmul(Dinv, XKX)
            E2 = numpy.matmul(E, E)
            F = numpy.matmul(Dinv, XK2X)
            trace_KMKM = (trace_K2 - 2.0*numpy.trace(F) + numpy.trace(E2)) / \
                sigma0**4
        else:
            trace_KMKM = (n-m)/sigma**4 - (2*eta/sigma**2)*trace_M + \
                (eta**2)*trace_M2

        # Trace of K*(M**2)
        if numpy.abs(sigma) < self.cov.tol:
            trace_KM = (n - numpy.trace(E))/sigma0**2
            trace_KMM = trace_KM / sigma0**2
        else:
            trace_KMM = trace_M/sigma**2 - eta*trace_M2

        # Compute second derivatives
        d2ell_dsigma0_dsigma0 = 0.5*trace_M2 - zMMMz
        d2ell_dsigma_dsigma = 0.5*trace_KMKM - zMKMKMz
        d2ell_dsigma_dsigma0 = 0.5*trace_KMM - zMMKMz

        # Hessian
        hessian = numpy.array(
                [[d2ell_dsigma_dsigma, d2ell_dsigma_dsigma0],
                 [d2ell_dsigma_dsigma0, d2ell_dsigma0_dsigma0]], dtype=float)

        # Compute Hessian w.r.t scale
        if hyperparam.size > 2:

            # Initialize arrays
            d2ell_dscale2 = numpy.zeros((scale.size, scale.size), dtype=float)
            d2ell_dsigma_dscale = numpy.zeros((scale.size), dtype=float)
            d2ell_dsigma0_dscale = numpy.zeros((scale.size), dtype=float)
            d2ell_mixed = numpy.zeros((2, scale.size), dtype=float)

            # Needed to compute trace (TODO)
            S = self.cov.get_matrix(sigma, sigma0)
            Sinv = numpy.linalg.inv(S)

            # Sp is the derivative of cov w.r.t the p-th element of scale. Spq
            # is the second mixed derivative of S w.r.t p-th and q-th elements
            # of scale.
            for p in range(scale.size):

                # -----------------------------------------------
                # 1. Compute mixed derivatives of scale and sigma
                # -----------------------------------------------

                # 1.1. Compute zMSpMKMz
                SpMz = self.cov.dot(sigma, sigma0, Mz, derivative=[p])
                MSpMz = self.M_dot(Binv, Y, sigma, sigma0, SpMz)
                zMSpMKMz = numpy.dot(SpMz, MKMz)

                # 1.2. Compute zMKpMz
                KpMz = self.cov.dot(1.0, 0.0, Mz, derivative=[p])
                zMKpMz = numpy.dot(Mz, KpMz)

                # 1.3. Compute trace of Kp * M

                # Compute the first component of trace of Kp * Sinv (TODO)
                Kp = self.cov.get_matrix(1.0, 0.0, derivative=[p])
                KpSinv = numpy.matmul(Kp, Sinv)
                trace_KpSinv, _ = imate.trace(KpSinv, method='exact')

                # Compute the second component of trace of Kp * M
                KpY = self.cov.dot(1.0, 0.0, Y, derivative=[p])
                YtKpY = numpy.matmul(Y.T, KpY)
                BinvYtKpY = numpy.matmul(Binv, YtKpY)
                trace_BinvYtKpY = numpy.trace(BinvYtKpY)

                # Compute trace of Kp * M
                trace_KpM = trace_KpSinv - trace_BinvYtKpY

                # 1.4. Compute trace of K * M * Sp * M

                # Compute first part of trace of K * M * Sp * M
                K = self.cov.get_matrix(1.0, 0.0, derivative=[])
                KSinv = numpy.matmul(K, Sinv)
                Sp = self.cov.get_matrix(sigma, sigma0, derivative=[p])
                SpSinv = numpy.matmul(Sp, Sinv)
                KSinvSpSinv = numpy.matmul(KSinv, SpSinv)
                trace_KMSpM_1, _ = imate.trace(KSinvSpSinv, method='exact')

                # Compute the second part of trace of K * M * Sp * M
                KY = numpy.matmul(K, Y)
                SpY = numpy.matmul(Sp, Y)
                SinvSpY = self.cov.solve(sigma, sigma0, SpY)
                YtKSinvSpY = numpy.matmul(KY.T, SinvSpY)
                C21 = numpy.matmul(Binv, YtKSinvSpY)
                C22 = numpy.matmul(Binv, YtKSinvSpY.T)
                trace_KMSpM_21 = numpy.trace(C21)
                trace_KMSpM_22 = numpy.trace(C22)

                # Compute the third part of trace of K * M * Sp * M
                YtKY = numpy.matmul(Y.T, KY)
                YtSpY = numpy.matmul(Y.T, SpY)
                Dk = numpy.matmul(Binv, YtKY)
                Dp = numpy.matmul(Binv, YtSpY)
                D = numpy.matmul(Dk, Dp)
                trace_KMSpM_3 = numpy.trace(D)

                # Compute trace of K * M * Sp * M
                trace_KMSpM = trace_KMSpM_1 - trace_KMSpM_21 - \
                    trace_KMSpM_22 + trace_KMSpM_3

                # 1.5. Second derivatives w.r.t scale
                d2ell_dsigma_dscale[p] = -0.5*trace_KpM + 0.5*trace_KMSpM - \
                    zMSpMKMz + 0.5*zMKpMz

                # ------------------------------------------------
                # 2. Compute mixed derivatives of scale and sigma0
                # ------------------------------------------------

                # 2.1. Compute zMSpMMz
                zMSpMMz = numpy.dot(SpMz, MMz)

                # 2.4. Compute trace of M * Sp * M

                # Compute first part of trace of M * Sp * M
                SinvSpSinv = numpy.matmul(Sinv, SpSinv)
                trace_MSpM_1, _ = imate.trace(SinvSpSinv, method='exact')

                # Compute the second part of trace of M * Sp * M
                YtSinvSpY = numpy.matmul(Y.T, SinvSpY)
                C21 = numpy.matmul(Binv, YtSinvSpY)
                C22 = numpy.matmul(Binv, YtSinvSpY.T)
                trace_MSpM_21 = numpy.trace(C21)
                trace_MSpM_22 = numpy.trace(C22)

                # Compute the third part of trace of M * Sp * M
                D = numpy.matmul(Dp, A)
                trace_MSpM_3 = numpy.trace(D)

                # Compute trace of M * Sp * M
                trace_MSpM = trace_MSpM_1 - trace_MSpM_21 - trace_MSpM_22 + \
                    trace_MSpM_3

                # 2.5. Second derivatives w.r.t scale
                d2ell_dsigma0_dscale[p] = 0.5*trace_MSpM - zMSpMMz

                # Concatenate mixed derivatives of scale and sigmas
                d2ell_mixed[0, :] = d2ell_dsigma_dscale
                d2ell_mixed[1, :] = d2ell_dsigma0_dscale

                # -----------------------------------
                # Compute second derivatives of scale
                # -----------------------------------

                for q in range(p, scale.size):

                    # 1. Compute zMSqMSpMz
                    if p == q:
                        SqMz = SpMz
                    else:
                        SqMz = self.cov.dot(sigma, sigma0, Mz, derivative=[q])
                    zMSqMSpMz = numpy.dot(SqMz, MSpMz)

                    # 2. Compute zMSpqMz
                    SpqMz = self.cov.dot(sigma, sigma0, Mz, derivative=[p, q])
                    zMSpqMz = numpy.dot(Mz, SpqMz)

                    # 3. Computing trace of Spq * M in three steps

                    # Compute the first component of trace of Spq*Sinv (TODO)
                    Spq = self.cov.get_matrix(sigma, sigma0, derivative=[p, q])
                    SpqSinv = numpy.matmul(Spq, Sinv)
                    trace_SpqSinv, _ = imate.trace(SpqSinv, method='exact')

                    # Compute the second component of trace of Spq * M
                    SpqY = self.cov.dot(sigma, sigma0, Y, derivative=[p, q])
                    YtSpqY = numpy.matmul(Y.T, SpqY)
                    BinvYtSpqY = numpy.matmul(Binv, YtSpqY)
                    trace_BinvYtSpqY = numpy.trace(BinvYtSpqY)

                    # Compute trace of Spq * M
                    trace_SpqM = trace_SpqSinv - trace_BinvYtSpqY

                    # 4. Compute trace of Sp * M * Sq * M

                    # Compute first part of trace of Sp * M * Sq * M
                    Sq = self.cov.get_matrix(sigma, sigma0, derivative=[q])
                    if p == q:
                        SqSinv = SpSinv
                    else:
                        SqSinv = numpy.matmul(Sq, Sinv)
                    SpSinvSqSinv = numpy.matmul(SpSinv, SqSinv)
                    trace_SpMSqM_1, _ = imate.trace(SpSinvSqSinv,
                                                    method='exact')

                    # Compute the second part of trace of Sp * M * Sq * M
                    SpY = numpy.matmul(Sp, Y)
                    if p == q:
                        SqY = SpY
                    else:
                        SqY = numpy.matmul(Sq, Y)
                    SinvSqY = self.cov.solve(sigma, sigma0, SqY)
                    YtSpSinvSqY = numpy.matmul(SpY.T, SinvSqY)
                    C21 = numpy.matmul(Binv, YtSpSinvSqY)
                    C22 = numpy.matmul(Binv, YtSpSinvSqY.T)
                    trace_SpMSqM_21 = numpy.trace(C21)
                    trace_SpMSqM_22 = numpy.trace(C22)

                    # Compute the third part of trace of Sp * M * Sq * M
                    YtSpY = numpy.matmul(Y.T, SpY)
                    if p == q:
                        YtSqY = YtSpY
                    else:
                        YtSqY = numpy.matmul(Y.T, SqY)
                    Dp = numpy.matmul(Binv, YtSpY)
                    if p == q:
                        Dq = Dp
                    else:
                        Dq = numpy.matmul(Binv, YtSqY)
                    D = numpy.matmul(Dp, Dq)
                    trace_SpMSqM_3 = numpy.trace(D)

                    # Compute trace of Sp * M * Sq * M
                    trace_SpMSqM = trace_SpMSqM_1 - trace_SpMSqM_21 - \
                        trace_SpMSqM_22 + trace_SpMSqM_3

                    # 5. Second derivatives w.r.t scale
                    d2ell_dscale2[p, q] = -0.5*trace_SpqM + \
                        0.5*trace_SpMSqM - zMSqMSpMz + 0.5*zMSpqMz

                    if p != q:
                        d2ell_dscale2[q, p] = d2ell_dscale2[p, q]

            # -------------------------------------
            # Convert derivative w.r.t log of scale (if needed)
            # -------------------------------------

            if self.use_log_scale:
                scale = self._hyperparam_to_scale(hyperparam[2:])

                # Mixed derivative of eta and sigma
                for p in range(scale.size):
                    d2ell_mixed[0, p] = d2ell_mixed[0, p] * \
                        scale[p] * numpy.log(10.0)

                # Mixed derivative of eta and sigma0
                for p in range(scale.size):
                    d2ell_mixed[1, p] = d2ell_mixed[1, p] * \
                        scale[p] * numpy.log(10.0)

                # To convert derivative to log scale, Jacobian is needed. Note:
                # The Jacobian itself is already converted to log scale.
                jacobian_ = self.likelihood_jacobian(False, hyperparam)

                # Second derivative w.r.t eta
                dell_dscale = jacobian_[2:]

                for p in range(scale.size):
                    for q in range(scale.size):
                        if p == q:

                            # dell_dscale is already converted to logscale
                            d2ell_dscale2[p, q] = d2ell_dscale2[p, q] * \
                                scale[p]**2 * (numpy.log(10.0)**2) + \
                                dell_dscale * numpy.log(10.0)
                        else:
                            d2ell_dscale2[p, q] = d2ell_dscale2[p, q] * \
                                scale[p] * scale[q] * (numpy.log(10.0)**2)

            # ---------------------------------
            # Concatenate all mixed derivatives
            # ---------------------------------

            hessian = numpy.block([[hessian, d2ell_mixed],
                                   [d2ell_mixed.T, d2ell_dscale2]])

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

        # Partial function of likelihood (with minus to make maximization to a
        # minimization).
        sign_switch = True
        likelihood_partial_func = partial(self.likelihood, sign_switch)

        # Partial function of Jacobian of likelihood (with minus sign)
        jacobian_partial_func = partial(self.likelihood_jacobian, sign_switch)

        # Partial function of Hessian of likelihood (with minus sign)
        hessian_partial_func = partial(self.likelihood_hessian, sign_switch)

        # Minimize
        res = scipy.optimize.minimize(likelihood_partial_func,
                                      hyperparam_guess,
                                      method=optimization_method, tol=tol,
                                      jac=jacobian_partial_func,
                                      hess=hessian_partial_func)

        print(res)

        print('Iter: %d, Eval: %d, Success: %s'
              % (res.nit, res.nfev, res.success))

        # Extract res
        sigma = numpy.abs(res.x[0])
        sigma0 = numpy.abs(res.x[1])
        eta = (sigma0/sigma)**2
        max_ell = -res.fun

        # Distance scale
        if res.x.size > 2:
            scale = self._hyperparam_to_scale(res.x[2:])
        else:
            scale = self.cov.get_scale()

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

    # ============================
    # plot likelihood versus scale
    # ============================

    def plot_likelihood_versus_scale(
            self,
            result,
            other_sigmas=None):
        """
        Plots log likelihood versus scale hyperparameter. Other hyperparameters
        such as sigma and sigma0 are fixed. sigma is used by both its optimal
        value and user-defined values and plots are iterated by the multiple
        sigma values. On the other hand, sigma0 is only used from its optimal
        value.
        """

        # This function can only plot one dimensional data.
        dimension = self.cov.mixed_cor.cor.dimension
        if dimension != 1:
            raise ValueError('To plot likelihood w.r.t "eta" and "scale", ' +
                             'the dimension of the data points should be one.')

        load_plot_settings()

        # Optimal point
        optimal_sigma = result['hyperparam']['sigma']
        optimal_sigma0 = result['hyperparam']['sigma0']

        # Convert sigma to a numpy array
        if other_sigmas is not None:
            if numpy.isscalar(other_sigmas):
                other_sigmas = numpy.array([other_sigmas])
            elif isinstance(other_sigmas, list):
                other_sigmas = numpy.array(other_sigmas)
            elif not isinstance(other_sigmas, numpy.ndarray):
                raise TypeError('"other_sigmas" should be either a scalar, ' +
                                'list, or numpy.ndarray.')

        # Concatenate all given sigmas
        if other_sigmas is not None:
            sigmas = numpy.r_[optimal_sigma, other_sigmas]
        else:
            sigmas = numpy.r_[optimal_sigma]
        sigmas = numpy.sort(sigmas)

        # 2nd or 4th order finite difference coefficients for first derivative
        coeff = numpy.array([-1.0/2.0, 0.0, 1.0/2.0])
        # coeff = numpy.array([1.0/12.0, -2.0/3.0, 0.0, 2.0/3.0, -1.0/12.0])

        # The first axis of some arrays below is 3, used for varying eta for
        # numerical evaluating of mixed derivative w.r.t eta.
        stencil_size = coeff.size
        center_stencil = stencil_size//2  # Index of the center of stencil

        # Generate ell for various distance scales
        scale = numpy.logspace(-3, 2, 100)

        # The variable on the abscissa to take derivative with respect to it.
        if self.use_log_scale:
            scale_x = numpy.log10(scale)
        else:
            scale_x = scale

        d0_ell_perturb_sigma = numpy.zeros((stencil_size, sigmas.size,
                                           scale.size), dtype=float)
        d0_ell_perturb_sigma0 = numpy.zeros((stencil_size, sigmas.size,
                                            scale.size), dtype=float)
        d1_ell = numpy.zeros((sigmas.size, scale.size), dtype=float)
        d2_ell = numpy.zeros((sigmas.size, scale.size), dtype=float)
        d2_mixed_sigma_ell = numpy.zeros((sigmas.size, scale.size),
                                         dtype=float)
        d2_mixed_sigma0_ell = numpy.zeros((sigmas.size, scale.size),
                                          dtype=float)
        d1_ell_perturb_sigma_numerical = numpy.zeros(
                (stencil_size, sigmas.size, scale.size-2), dtype=float)
        d1_ell_perturb_sigma0_numerical = numpy.zeros(
                (stencil_size, sigmas.size, scale.size-2), dtype=float)
        d2_ell_numerical = numpy.zeros((sigmas.size, scale.size-4),
                                       dtype=float)
        d2_mixed_sigma_ell_numerical = numpy.zeros(
                (sigmas.size, scale.size-2), dtype=float)
        d2_mixed_sigma0_ell_numerical = numpy.zeros(
                (sigmas.size, scale.size-2), dtype=float)

        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))
        colors = matplotlib.cm.nipy_spectral(
                numpy.linspace(0, 0.9, sigmas.size))

        for i in range(sigmas.size):

            # Stencil to perturb sigma
            d_sigma = sigmas[i] * 1e-3
            sigma_stencil = sigmas[i] + d_sigma * \
                numpy.arange(-stencil_size//2+1, stencil_size//2+1)

            # Stencil to perturb sigma0
            d_sigma0 = optimal_sigma0 * 1e-3
            sigma0_stencil = optimal_sigma0 + d_sigma0 * \
                numpy.arange(-stencil_size//2+1, stencil_size//2+1)

            for j in range(scale.size):

                # Set the scale
                self.cov.set_scale(scale[j])

                # Likelihood (and its perturbation w.r.t sigma)
                for k in range(stencil_size):
                    hyperparam = numpy.r_[sigma_stencil[k], optimal_sigma0,
                                          self._scale_to_hyperparam(scale[j])]
                    sign_switch = False
                    d0_ell_perturb_sigma[k, i, j] = self.likelihood(
                            sign_switch, hyperparam)

                # Likelihood (and its perturbation w.r.t sigma0)
                for k in range(stencil_size):
                    hyperparam = numpy.r_[sigmas[i], sigma0_stencil[k],
                                          self._scale_to_hyperparam(scale[j])]
                    sign_switch = False
                    d0_ell_perturb_sigma0[k, i, j] = self.likelihood(
                            sign_switch, hyperparam)

                # First derivative of likelihood w.r.t distance scale
                hyperparam = numpy.r_[sigmas[i], optimal_sigma0,
                                      self._scale_to_hyperparam(scale[j])]
                jacobian_ = self.likelihood_jacobian(sign_switch, hyperparam)
                d1_ell[i, j] = jacobian_[2]

                # Second derivative of likelihood w.r.t distance scale
                hessian_ = self.likelihood_hessian(sign_switch, hyperparam)
                d2_mixed_sigma_ell[i, j] = hessian_[0, 2]
                d2_mixed_sigma0_ell[i, j] = hessian_[1, 2]
                d2_ell[i, j] = hessian_[2, 2]

            for k in range(stencil_size):
                # Compute first derivative numerically (perturb sigma)
                d1_ell_perturb_sigma_numerical[k, i, :] = \
                        (d0_ell_perturb_sigma[k, i, 2:] -
                         d0_ell_perturb_sigma[k, i, :-2]) / \
                        (scale_x[2:] - scale_x[:-2])

                # Compute first derivative numerically (perturb sigma0)
                d1_ell_perturb_sigma0_numerical[k, i, :] = \
                    (d0_ell_perturb_sigma0[k, i, 2:] -
                        d0_ell_perturb_sigma0[k, i, :-2]) / \
                    (scale_x[2:] - scale_x[:-2])

                # Compute second mixed derivative w.r.t sigma, numerically
                d2_mixed_sigma_ell_numerical[i, :] += \
                    coeff[k] * d1_ell_perturb_sigma_numerical[k, i, :] / \
                    d_sigma

                # Compute second mixed derivative w.r.t sigma0, numerically
                d2_mixed_sigma0_ell_numerical[i, :] += \
                    coeff[k] * d1_ell_perturb_sigma0_numerical[k, i, :] / \
                    d_sigma0

            # Note, the above mixed derivatives are w.r.t sigma and sigma0. To
            # compute the derivatives w.r.t to sigma**2 and sigma0**2 (squared
            # variables) divide them by 2*sigma and 2*sigma0 respectively.
            d2_mixed_sigma_ell_numerical[i, :] /= (2.0 * sigmas[i])
            d2_mixed_sigma0_ell_numerical[i, :] /= (2.0 * optimal_sigma0)

            # Compute second derivative numerically
            d2_ell_numerical[i, :] = \
                (d1_ell_perturb_sigma_numerical[center_stencil, i, 2:] -
                 d1_ell_perturb_sigma_numerical[center_stencil, i, :-2]) / \
                (scale_x[3:-1] - scale_x[1:-3])

            # Find maximum of ell
            max_index = numpy.argmax(
                    d0_ell_perturb_sigma[center_stencil, i, :])
            optimal_scale = scale[max_index]
            optimal_ell = d0_ell_perturb_sigma[center_stencil, i, max_index]

            # Plot
            if sigmas[i] == optimal_sigma:
                label = r'$\hat{\sigma}=%0.2e$' % sigmas[i]
                marker = 'X'
            else:
                label = r'$\sigma=%0.2e$' % sigmas[i]
                marker = 'o'
            ax[0, 0].plot(scale, d0_ell_perturb_sigma[center_stencil, i, :],
                          color=colors[i], label=label)
            ax[0, 1].plot(scale, d1_ell[i, :], color=colors[i], label=label)
            ax[0, 2].plot(scale, d2_ell[i, :], color=colors[i], label=label)
            ax[1, 0].plot(scale, d2_mixed_sigma_ell[i, :], color=colors[i],
                          label=label)
            ax[1, 1].plot(scale, d2_mixed_sigma0_ell[i, :], color=colors[i],
                          label=label)
            ax[0, 1].plot(scale[1:-1],
                          d1_ell_perturb_sigma_numerical[center_stencil, i, :],
                          '--', color=colors[i])
            ax[0, 2].plot(scale[2:-2], d2_ell_numerical[i, :], '--',
                          color=colors[i])
            ax[1, 0].plot(scale[1:-1], d2_mixed_sigma_ell_numerical[i, :],
                          '--', color=colors[i])
            ax[1, 1].plot(scale[1:-1], d2_mixed_sigma0_ell_numerical[i, :],
                          '--', color=colors[i])
            p = ax[0, 0].plot(optimal_scale, optimal_ell, marker,
                              color=colors[i], markersize=3)
            ax[0, 1].plot(optimal_scale, 0.0,  marker,
                          color=colors[i], markersize=3)

        ax[0, 0].legend(p, [r'optimal $\theta$'])
        ax[0, 0].legend(loc='lower right')
        ax[0, 1].legend(loc='lower right')
        ax[0, 2].legend(loc='lower right')
        ax[1, 0].legend(loc='lower right')
        ax[1, 1].legend(loc='lower right')
        ax[0, 0].set_xscale('log')
        ax[0, 1].set_xscale('log')
        ax[0, 2].set_xscale('log')
        ax[1, 0].set_xscale('log')
        ax[1, 1].set_xscale('log')
        ax[0, 0].set_yscale('linear')
        ax[0, 1].set_yscale('linear')
        ax[0, 2].set_yscale('linear')
        ax[1, 0].set_yscale('linear')
        ax[1, 1].set_yscale('linear')

        # Plot annotations
        ax[0, 0].set_xlim([scale[0], scale[-1]])
        ax[0, 1].set_xlim([scale[0], scale[-1]])
        ax[0, 2].set_xlim([scale[0], scale[-1]])
        ax[1, 0].set_xlim([scale[0], scale[-1]])
        ax[1, 1].set_xlim([scale[0], scale[-1]])
        ax[0, 0].set_xlabel(r'$\theta$')
        ax[0, 1].set_xlabel(r'$\theta$')
        ax[0, 2].set_xlabel(r'$\theta$')
        ax[1, 0].set_xlabel(r'$\theta$')
        ax[1, 1].set_xlabel(r'$\theta$')
        ax[0, 0].set_ylabel(r'$\ell(\theta | \sigma^2, \sigma_0^2)$')

        if self.use_log_scale:
            ax[0, 1].set_ylabel(
                r'$\frac{\mathrm{d} \ell(\theta | \sigma^2, \sigma_0^2)} ' +
                r'{\mathrm{d} (\ln \theta)}$')
        else:
            ax[0, 1].set_ylabel(
                r'$\frac{\mathrm{d} \ell(\theta | \sigma^2, \sigma_0^2)} ' +
                r'{\mathrm{d} \theta}$')

        if self.use_log_scale:
            ax[0, 2].set_ylabel(
                r'$\frac{\mathrm{d}^2 \ell(\theta | \sigma^2, \sigma_0^2)} ' +
                r'{\mathrm{d} (\ln \theta)^2}$')
        else:
            ax[0, 2].set_ylabel(
                r'$\frac{\mathrm{d}^2 \ell(\theta | \sigma^2, \sigma_0^2)} ' +
                r'{\mathrm{d} \theta^2}$')

        if self.use_log_scale:
            ax[1, 0].set_ylabel(
                r'$\frac{\mathrm{d}^2 \ell(\theta | \sigma^2, \sigma_0^2)} ' +
                r'{\mathrm{d} (\ln \theta) \mathrm{d} \sigma^2}$')
        else:
            ax[1, 0].set_ylabel(
                r'$\frac{\mathrm{d}^2 \ell(\theta | \sigma^2, \sigma_0^2)} ' +
                r'{\mathrm{d} (\ln \theta) \mathrm{d} \sigma^2}$')

        if self.use_log_scale:
            ax[1, 1].set_ylabel(
                r'$\frac{\mathrm{d}^2 \ell(\theta | \sigma^2, \sigma_0^2)} ' +
                r'{\mathrm{d} (\ln \theta) \mathrm{d} {\sigma_0}^2}$')
        else:
            ax[1, 1].set_ylabel(
                r'$\frac{\mathrm{d}^2 \ell(\theta | \sigma^2, \sigma_0^2)} ' +
                r'{\mathrm{d} \theta \mathrm{d} {\sigma_0}^2}$')

        ax[0, 0].set_title(r'Log likelihood function, given ' +
                           r'$(\sigma^2, \sigma_0^2)$ ')
        ax[0, 1].set_title(r'First derivative of log likelihood, given ' +
                           r'$(\sigma^2, \sigma_0^2)$')
        ax[0, 2].set_title(r'Second derivative of log likelihood, given' +
                           r'$(\sigma^2, \sigma_0^2)$')
        ax[1, 0].set_title(r'Second mixed derivative of log likelihood, ' +
                           r'given $(\sigma^2, \sigma_0^2)$')
        ax[1, 1].set_title(r'Second mixed derivative of log likelihood, ' +
                           r'given $(\sigma^2, \sigma_0^2)$')
        ax[0, 0].grid(True, which='both')
        ax[0, 1].grid(True, which='both')
        ax[0, 2].grid(True, which='both')
        ax[1, 0].grid(True, which='both')
        ax[1, 1].grid(True, which='both')

        ax[1, 2].set_axis_off()

        plt.tight_layout()
        plt.show()

    # ============================
    # plot likelihood versus sigma
    # ============================

    def plot_likelihood_versus_sigma(
            self,
            result,
            other_scales=None):
        """
        Plots log likelihood versus sigma. Other hyperparameters are fixed.
        Also, scale is used from both its optimal value and user-defined
        values. Plots are iterated over multiple values of scale. On the other
        hand, sigma0 is fixed to its optimal value.
        """

        # This function can only plot one dimensional data.
        dimension = self.cov.mixed_cor.cor.dimension
        if dimension != 1:
            raise ValueError('To plot likelihood w.r.t "eta" and "scale", ' +
                             'the dimension of the data points should be one.')

        load_plot_settings()

        # Optimal point
        optimal_sigma = result['hyperparam']['sigma']
        optimal_sigma0 = result['hyperparam']['sigma0']
        optimal_scale = result['hyperparam']['scale']

        # Convert scale to a numpy array
        if other_scales is not None:
            if numpy.isscalar(other_scales):
                other_scales = numpy.array([other_scales])
            elif isinstance(other_scales, list):
                other_scales = numpy.array(other_scales)
            elif not isinstance(other_scales, numpy.ndarray):
                raise TypeError('"other_scales" should be either a scalar, ' +
                                'list, or numpy.ndarray.')

        # Concatenate all given scales
        if other_scales is not None:
            scales = numpy.r_[optimal_scale, other_scales]
        else:
            scales = numpy.r_[optimal_scale]
        scales = numpy.sort(scales)

        # 2nd or 4th order finite difference coefficients for first derivative
        coeff = numpy.array([-1.0/2.0, 0.0, 1.0/2.0])
        # coeff = numpy.array([1.0/12.0, -2.0/3.0, 0.0, 2.0/3.0, -1.0/12.0])

        # The first axis of some arrays below is 3, used for varying eta for
        # numerical evaluating of mixed derivative w.r.t eta.
        stencil_size = coeff.size
        center_stencil = stencil_size//2  # Index of the center of stencil

        # Generate ell for various sigma
        sigma = numpy.logspace(-2, 2, 100)

        d0_ell_perturb_scale = numpy.zeros(
                (stencil_size, scales.size, sigma.size), dtype=float)
        d0_ell_perturb_sigma0 = numpy.zeros(
                (stencil_size, scales.size, sigma.size), dtype=float)
        d1_ell = numpy.zeros((scales.size, sigma.size), dtype=float)
        d2_ell = numpy.zeros((scales.size, sigma.size), dtype=float)
        d2_mixed_scale_ell = numpy.zeros((scales.size, sigma.size),
                                         dtype=float)
        d2_mixed_sigma0_ell = numpy.zeros((scales.size, sigma.size),
                                          dtype=float)
        d1_ell_perturb_scale_numerical = numpy.zeros(
                (stencil_size, scales.size, sigma.size-2), dtype=float)
        d1_ell_perturb_sigma0_numerical = numpy.zeros(
                (stencil_size, scales.size, sigma.size-2), dtype=float)
        d2_ell_numerical = numpy.zeros((scales.size, sigma.size-4),
                                       dtype=float)
        d2_mixed_scale_ell_numerical = numpy.zeros(
                (scales.size, sigma.size-2), dtype=float)
        d2_mixed_sigma0_ell_numerical = numpy.zeros(
                (scales.size, sigma.size-2), dtype=float)

        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))
        colors = matplotlib.cm.nipy_spectral(
                numpy.linspace(0, 0.9, scales.size))

        for i in range(scales.size):

            # Stencil to perturb scale
            if self.use_log_scale:
                log_scale = numpy.log10(scales[i])
                d_scale = log_scale * 1e-3
                scale_stencil = 10.0**(
                        log_scale + d_scale *
                        numpy.arange(-stencil_size//2+1, stencil_size//2+1))
            else:
                d_scale = scales[i] * 1e-3
                scale_stencil = scales[i] + d_scale * \
                    numpy.arange(-stencil_size//2+1, stencil_size//2+1)

            # Stencil to perturb sigma0
            d_sigma0 = optimal_sigma0 * 1e-3
            sigma0_stencil = optimal_sigma0 + d_sigma0 * \
                numpy.arange(-stencil_size//2+1, stencil_size//2+1)

            # Likelihood (and its perturbation w.r.t sigma)
            for k in range(stencil_size):

                # Set the scale
                self.cov.set_scale(scale_stencil[k])

                for j in range(sigma.size):
                    hyperparam = numpy.r_[
                            sigma[j], optimal_sigma0,
                            self._scale_to_hyperparam(scale_stencil[k])]
                    sign_switch = False
                    d0_ell_perturb_scale[k, i, j] = self.likelihood(
                        sign_switch, hyperparam)

            # Likelihood (and its perturbation w.r.t sigma0)
            self.cov.set_scale(scales[i])
            for k in range(stencil_size):
                for j in range(sigma.size):
                    hyperparam = numpy.r_[sigma[j], sigma0_stencil[k],
                                          self._scale_to_hyperparam(scales[i])]
                    sign_switch = False
                    d0_ell_perturb_sigma0[k, i, j] = self.likelihood(
                            sign_switch, hyperparam)

            # First derivative of likelihood w.r.t distance scale
            for j in range(sigma.size):
                hyperparam = numpy.r_[sigma[j], optimal_sigma0,
                                      self._scale_to_hyperparam(scales[i])]
                jacobian_ = self.likelihood_jacobian(sign_switch, hyperparam)
                d1_ell[i, j] = jacobian_[0]

                # Second derivative of likelihood w.r.t distance scale
                hessian_ = self.likelihood_hessian(sign_switch, hyperparam)
                d2_mixed_scale_ell[i, j] = hessian_[0, 2]
                d2_mixed_sigma0_ell[i, j] = hessian_[0, 1]
                d2_ell[i, j] = hessian_[0, 0]

            for k in range(stencil_size):
                # First derivative numerically (perturb scale)
                d1_ell_perturb_scale_numerical[k, i, :] = \
                        (d0_ell_perturb_scale[k, i, 2:] -
                         d0_ell_perturb_scale[k, i, :-2]) / \
                        (sigma[2:] - sigma[:-2])

                # To take derivative w.r.t sigma**2, divide by 2*sigma.
                for j in range(sigma.size-2):
                    d1_ell_perturb_scale_numerical[k, i, j] /= \
                            (2.0 * sigma[j+1])

                # Compute first derivative numerically (perturb sigma0)
                d1_ell_perturb_sigma0_numerical[k, i, :] = \
                    (d0_ell_perturb_sigma0[k, i, 2:] -
                        d0_ell_perturb_sigma0[k, i, :-2]) / \
                    (sigma[2:] - sigma[:-2])

                # To take derivative w.r.t sigma**2, divide by 2*sigma.
                for j in range(sigma.size-2):
                    d1_ell_perturb_sigma0_numerical[k, i, j] /= \
                            (2.0 * sigma[j+1])

                # Second mixed derivative w.r.t scale, numerically
                d2_mixed_scale_ell_numerical[i, :] += coeff[k] * \
                    d1_ell_perturb_scale_numerical[k, i, :] / d_scale

                # Compute second mixed derivative w.r.t sigma0, numerically
                d2_mixed_sigma0_ell_numerical[i, :] += \
                    coeff[k] * d1_ell_perturb_sigma0_numerical[k, i, :] / \
                    d_sigma0

            # To take derivative w.r.t sigma0**2, divide by 2*sigma0.
            d2_mixed_sigma0_ell_numerical[i, :] /= (2.0 * optimal_sigma0)

            # Compute second derivative numerically
            d2_ell_numerical[i, :] = \
                (d1_ell_perturb_scale_numerical[center_stencil, i, 2:] -
                 d1_ell_perturb_scale_numerical[center_stencil, i, :-2]) / \
                (sigma[3:-1] - sigma[1:-3])

            # To take derivative w.r.t sigma0**2, divide by 2*sigma0.
            for j in range(sigma.size-4):
                d2_ell_numerical[i, j] /= (2.0 * sigma[j+2])

            # Find maximum of ell
            max_index = numpy.argmax(
                    d0_ell_perturb_scale[center_stencil, i, :])
            optimal_sigma = sigma[max_index]
            optimal_ell = d0_ell_perturb_scale[center_stencil, i, max_index]

            # Plot
            if any(scales[i] == optimal_scale):
                label = r'$\hat{\theta}=%0.2e$' % scales[i]
                marker = 'X'
            else:
                label = r'$\theta=%0.2e$' % scales[i]
                marker = 'o'
            ax[0, 0].plot(sigma, d0_ell_perturb_scale[center_stencil, i, :],
                          color=colors[i], label=label)
            ax[0, 1].plot(sigma, d1_ell[i, :], color=colors[i], label=label)
            ax[0, 2].plot(sigma, d2_ell[i, :], color=colors[i], label=label)
            ax[1, 0].plot(sigma, d2_mixed_scale_ell[i, :], color=colors[i],
                          label=label)
            ax[1, 1].plot(sigma, d2_mixed_sigma0_ell[i, :], color=colors[i],
                          label=label)
            ax[0, 1].plot(sigma[1:-1], d1_ell_perturb_scale_numerical[
                          center_stencil, i, :], '--', color=colors[i])
            ax[0, 2].plot(sigma[2:-2], d2_ell_numerical[i, :], '--',
                          color=colors[i])
            ax[1, 0].plot(sigma[1:-1], d2_mixed_scale_ell_numerical[i, :],
                          '--', color=colors[i])
            ax[1, 1].plot(sigma[1:-1], d2_mixed_sigma0_ell_numerical[i, :],
                          '--', color=colors[i])
            p = ax[0, 0].plot(optimal_sigma, optimal_ell, marker,
                              color=colors[i], markersize=3)
            ax[0, 1].plot(optimal_sigma, 0.0,  marker, color=colors[i],
                          markersize=3)

        ax[0, 0].legend(p, [r'optimal $\sigma$'])
        ax[0, 0].legend(loc='lower right')
        ax[0, 1].legend(loc='lower right')
        ax[0, 2].legend(loc='lower right')
        ax[1, 0].legend(loc='lower right')
        ax[1, 1].legend(loc='lower right')
        ax[0, 0].set_xscale('log')
        ax[0, 1].set_xscale('log')
        ax[0, 2].set_xscale('log')
        ax[1, 0].set_xscale('log')
        ax[1, 1].set_xscale('log')
        ax[0, 0].set_yscale('linear')
        ax[0, 1].set_yscale('linear')
        ax[0, 2].set_yscale('linear')
        ax[1, 0].set_yscale('linear')
        ax[1, 1].set_yscale('linear')

        # Plot annotations
        ax[0, 0].set_xlim([sigma[0], sigma[-1]])
        ax[0, 1].set_xlim([sigma[0], sigma[-1]])
        ax[0, 2].set_xlim([sigma[0], sigma[-1]])
        ax[1, 0].set_xlim([sigma[0], sigma[-1]])
        ax[1, 1].set_xlim([sigma[0], sigma[-1]])
        ax[0, 0].set_xlabel(r'$\sigma$')
        ax[0, 1].set_xlabel(r'$\sigma$')
        ax[0, 2].set_xlabel(r'$\sigma$')
        ax[1, 0].set_xlabel(r'$\sigma$')
        ax[1, 1].set_xlabel(r'$\sigma$')
        ax[0, 0].set_ylabel(r'$\ell(\sigma^2 | \sigma_0^2, \theta)$')
        ax[0, 1].set_ylabel(
            r'$\frac{\mathrm{d} \ell(\sigma^2 | \sigma_0^2, \theta)} ' +
            r'{\mathrm{d} \sigma^2}$')
        ax[0, 2].set_ylabel(
            r'$\frac{\mathrm{d}^2 \ell(\sigma^2 | \sigma_0^2, \theta)} ' +
            r'{\mathrm{d} (\sigma^2)^2}$')

        if self.use_log_scale:
            ax[1, 0].set_ylabel(
                r'$\frac{\mathrm{d}^2 \ell(\sigma^2 | \sigma_0^2, \theta)} ' +
                r'{\mathrm{d} \sigma^2 \mathrm{d} (\ln \theta)}$')
        else:
            ax[1, 0].set_ylabel(
                r'$\frac{\mathrm{d}^2 \ell(\sigma^2 | \sigma_0^2, \theta)} ' +
                r'{\mathrm{d} \sigma^2 \mathrm{d} \theta}$')

        ax[1, 1].set_ylabel(
            r'$\frac{\mathrm{d}^2 \ell(\sigma^2 | \sigma_0^2, \theta)} ' +
            r'{\mathrm{d} \sigma^2 \mathrm{d} {\sigma_0}^2}$')
        ax[0, 0].set_title(r'Log likelihood function, given ' +
                           r'$(\sigma_0^2, \theta)$')
        ax[0, 1].set_title(r'First derivative of log likelihood, given ' +
                           r'$(\sigma_0^2, \theta)$')
        ax[0, 2].set_title(r'Second derivative of log likelihood, given ' +
                           r'$(\sigma_0^2, \theta)$')
        ax[1, 0].set_title(r'Second mixed derivative of log likelihood, ' +
                           r'given $(\sigma_0^2, \theta)$')
        ax[1, 1].set_title(r'Second mixed derivative of log likelihood, ' +
                           r'given $(\sigma_0^2, \theta)$')
        ax[0, 0].grid(True, which='both')
        ax[0, 1].grid(True, which='both')
        ax[0, 2].grid(True, which='both')
        ax[1, 0].grid(True, which='both')
        ax[1, 1].grid(True, which='both')

        ax[1, 2].set_axis_off()

        plt.tight_layout()
        plt.show()

    # =============================
    # plot likelihood versus sigma0
    # =============================

    def plot_likelihood_versus_sigma0(
            self,
            result,
            other_scales=None):
        """
        Plots log likelihood versus sigma0. Other hyperparameters are fixed.
        Also, scale is used from both its optimal value and user-defined
        values. Plots are iterated over multiple values of scale. On the other
        hand, sigma is fixed to its optimal value.
        """

        # This function can only plot one dimensional data.
        dimension = self.cov.mixed_cor.cor.dimension
        if dimension != 1:
            raise ValueError('To plot likelihood w.r.t "eta" and "scale", ' +
                             'the dimension of the data points should be one.')

        load_plot_settings()

        # Optimal point
        optimal_sigma = result['hyperparam']['sigma']
        optimal_sigma0 = result['hyperparam']['sigma0']
        optimal_scale = result['hyperparam']['scale']

        # Convert scale to a numpy array
        if other_scales is not None:
            if numpy.isscalar(other_scales):
                other_scales = numpy.array([other_scales])
            elif isinstance(other_scales, list):
                other_scales = numpy.array(other_scales)
            elif not isinstance(other_scales, numpy.ndarray):
                raise TypeError('"other_scales" should be either a scalar, ' +
                                'list, or numpy.ndarray.')

        # Concatenate all given scales
        if other_scales is not None:
            scales = numpy.r_[optimal_scale, other_scales]
        else:
            scales = numpy.r_[optimal_scale]
        scales = numpy.sort(scales)

        # 2nd or 4th order finite difference coefficients for first derivative
        coeff = numpy.array([-1.0/2.0, 0.0, 1.0/2.0])
        # coeff = numpy.array([1.0/12.0, -2.0/3.0, 0.0, 2.0/3.0, -1.0/12.0])

        # The first axis of some arrays below is 3, used for varying eta for
        # numerical evaluating of mixed derivative w.r.t eta.
        stencil_size = coeff.size
        center_stencil = stencil_size//2  # Index of the center of stencil

        # Generate ell for various sigma0
        sigma0 = numpy.logspace(-2, 2, 100)

        d0_ell_perturb_scale = numpy.zeros(
                (stencil_size, scales.size, sigma0.size), dtype=float)
        d0_ell_perturb_sigma = numpy.zeros(
                (stencil_size, scales.size, sigma0.size), dtype=float)
        d1_ell = numpy.zeros((scales.size, sigma0.size), dtype=float)
        d2_ell = numpy.zeros((scales.size, sigma0.size), dtype=float)
        d2_mixed_scale_ell = numpy.zeros((scales.size, sigma0.size),
                                         dtype=float)
        d2_mixed_sigma_ell = numpy.zeros((scales.size, sigma0.size),
                                         dtype=float)
        d1_ell_perturb_scale_numerical = numpy.zeros(
                (stencil_size, scales.size, sigma0.size-2), dtype=float)
        d1_ell_perturb_sigma_numerical = numpy.zeros(
                (stencil_size, scales.size, sigma0.size-2), dtype=float)
        d2_ell_numerical = numpy.zeros((scales.size, sigma0.size-4),
                                       dtype=float)
        d2_mixed_scale_ell_numerical = numpy.zeros(
                (scales.size, sigma0.size-2), dtype=float)
        d2_mixed_sigma_ell_numerical = numpy.zeros(
                (scales.size, sigma0.size-2), dtype=float)

        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))
        colors = matplotlib.cm.nipy_spectral(
                numpy.linspace(0, 0.9, scales.size))

        for i in range(scales.size):

            # Stencil to perturb scale
            if self.use_log_scale:
                log_scale = numpy.log10(scales[i])
                d_scale = log_scale * 1e-3
                scale_stencil = 10.0**(
                        log_scale + d_scale *
                        numpy.arange(-stencil_size//2+1, stencil_size//2+1))
            else:
                d_scale = scales[i] * 1e-3
                scale_stencil = scales[i] + d_scale * \
                    numpy.arange(-stencil_size//2+1, stencil_size//2+1)

            # Stencil to perturb sigma
            d_sigma = optimal_sigma * 1e-3
            sigma_stencil = optimal_sigma + d_sigma * \
                numpy.arange(-stencil_size//2+1, stencil_size//2+1)

            # Likelihood (and its perturbation w.r.t sigma0)
            for k in range(stencil_size):

                # Set the scale
                self.cov.set_scale(scale_stencil[k])

                for j in range(sigma0.size):
                    hyperparam = numpy.r_[
                            optimal_sigma, sigma0[j],
                            self._scale_to_hyperparam(scale_stencil[k])]
                    sign_switch = False
                    d0_ell_perturb_scale[k, i, j] = self.likelihood(
                        sign_switch, hyperparam)

            # Likelihood (and its perturbation w.r.t sigma)
            self.cov.set_scale(scales[i])
            for k in range(stencil_size):
                for j in range(sigma0.size):
                    hyperparam = numpy.r_[sigma_stencil[k], sigma0[j],
                                          self._scale_to_hyperparam(scales[i])]
                    sign_switch = False
                    d0_ell_perturb_sigma[k, i, j] = self.likelihood(
                        sign_switch, hyperparam)

            # First derivative of likelihood w.r.t distance scale
            for j in range(sigma0.size):
                hyperparam = numpy.r_[sigma_stencil[k], sigma0[j],
                                      self._scale_to_hyperparam(scales[i])]
                jacobian_ = self.likelihood_jacobian(sign_switch, hyperparam)
                d1_ell[i, j] = jacobian_[1]

                # Second derivative of likelihood w.r.t distance scale
                hessian_ = self.likelihood_hessian(sign_switch, hyperparam)
                d2_mixed_scale_ell[i, j] = hessian_[1, 2]
                d2_mixed_sigma_ell[i, j] = hessian_[1, 0]
                d2_ell[i, j] = hessian_[1, 1]

            for k in range(stencil_size):
                # First derivative numerically (perturb scale)
                d1_ell_perturb_scale_numerical[k, i, :] = \
                        (d0_ell_perturb_scale[k, i, 2:] -
                         d0_ell_perturb_scale[k, i, :-2]) / \
                        (sigma0[2:] - sigma0[:-2])

                # To take derivative w.r.t sigma**2, divide by 2*sigma0.
                for j in range(sigma0.size-2):
                    d1_ell_perturb_scale_numerical[k, i, j] /= \
                            (2.0 * sigma0[j+1])

                # Compute first derivative numerically (perturb sigma)
                d1_ell_perturb_sigma_numerical[k, i, :] = \
                    (d0_ell_perturb_sigma[k, i, 2:] -
                        d0_ell_perturb_sigma[k, i, :-2]) / \
                    (sigma0[2:] - sigma0[:-2])

                # To take derivative w.r.t sigma0**2, divide by 2*sigma0.
                for j in range(sigma0.size-2):
                    d1_ell_perturb_sigma_numerical[k, i, j] /= \
                            (2.0 * sigma0[j+1])

                # Second mixed derivative w.r.t scale, numerically
                d2_mixed_scale_ell_numerical[i, :] += coeff[k] * \
                    d1_ell_perturb_scale_numerical[k, i, :] / d_scale

                # Compute second mixed derivative w.r.t sigma, numerically
                d2_mixed_sigma_ell_numerical[i, :] += \
                    coeff[k] * d1_ell_perturb_sigma_numerical[k, i, :] / \
                    d_sigma

            # To take derivative w.r.t sigma**2, divide by 2*sigma.
            d2_mixed_sigma_ell_numerical[i, :] /= (2.0 * optimal_sigma)

            # Compute second derivative numerically
            d2_ell_numerical[i, :] = \
                (d1_ell_perturb_scale_numerical[center_stencil, i, 2:] -
                 d1_ell_perturb_scale_numerical[center_stencil, i, :-2]) / \
                (sigma0[3:-1] - sigma0[1:-3])

            # To take derivative w.r.t sigma**2, divide by 2*sigma0.
            for j in range(sigma0.size-4):
                d2_ell_numerical[i, j] /= (2.0 * sigma0[j+2])

            # Find maximum of ell
            max_index = numpy.argmax(
                    d0_ell_perturb_scale[center_stencil, i, :])
            optimal_sigma0 = sigma0[max_index]
            optimal_ell = d0_ell_perturb_scale[center_stencil, i, max_index]

            # Plot
            if any(scales[i] == optimal_scale):
                label = r'$\hat{\theta}=%0.2e$' % scales[i]
                marker = 'X'
            else:
                label = r'$\theta=%0.2e$' % scales[i]
                marker = 'o'
            ax[0, 0].plot(sigma0, d0_ell_perturb_scale[center_stencil, i, :],
                          color=colors[i], label=label)
            ax[0, 1].plot(sigma0, d1_ell[i, :], color=colors[i], label=label)
            ax[0, 2].plot(sigma0, d2_ell[i, :], color=colors[i], label=label)
            ax[1, 0].plot(sigma0, d2_mixed_scale_ell[i, :], color=colors[i],
                          label=label)
            ax[1, 1].plot(sigma0, d2_mixed_sigma_ell[i, :], color=colors[i],
                          label=label)
            ax[0, 1].plot(sigma0[1:-1], d1_ell_perturb_scale_numerical[
                          center_stencil, i, :], '--', color=colors[i])
            ax[0, 2].plot(sigma0[2:-2], d2_ell_numerical[i, :], '--',
                          color=colors[i])
            ax[1, 0].plot(sigma0[1:-1], d2_mixed_scale_ell_numerical[i, :],
                          '--', color=colors[i])
            ax[1, 1].plot(sigma0[1:-1], d2_mixed_sigma_ell_numerical[i, :],
                          '--', color=colors[i])
            p = ax[0, 0].plot(optimal_sigma0, optimal_ell, marker,
                              color=colors[i], markersize=3)
            ax[0, 1].plot(optimal_sigma0, 0.0,  marker, color=colors[i],
                          markersize=3)

        ax[0, 0].legend(p, [r'optimal $\sigma0$'])
        ax[0, 0].legend(loc='lower right')
        ax[0, 1].legend(loc='lower right')
        ax[0, 2].legend(loc='lower right')
        ax[1, 0].legend(loc='lower right')
        ax[1, 1].legend(loc='lower right')
        ax[0, 0].set_xscale('log')
        ax[0, 1].set_xscale('log')
        ax[0, 2].set_xscale('log')
        ax[1, 0].set_xscale('log')
        ax[1, 1].set_xscale('log')

        # Plot annotations
        ax[0, 0].set_xlim([sigma0[0], sigma0[-1]])
        ax[0, 1].set_xlim([sigma0[0], sigma0[-1]])
        ax[0, 2].set_xlim([sigma0[0], sigma0[-1]])
        ax[1, 0].set_xlim([sigma0[0], sigma0[-1]])
        ax[1, 1].set_xlim([sigma0[0], sigma0[-1]])
        ax[0, 0].set_xlabel(r'$\sigma_0$')
        ax[0, 1].set_xlabel(r'$\sigma_0$')
        ax[0, 2].set_xlabel(r'$\sigma_0$')
        ax[1, 0].set_xlabel(r'$\sigma_0$')
        ax[1, 1].set_xlabel(r'$\sigma_0$')
        ax[0, 0].set_ylabel(r'$\ell({\sigma_0}^2 | \sigma^2, \theta)$')
        ax[0, 1].set_ylabel(
            r'$\frac{\mathrm{d} \ell({\sigma_0}^2 | \sigma^2, \theta)} ' +
            r'{\mathrm{d} {\sigma_0}^2}$')
        ax[0, 2].set_ylabel(
            r'$\frac{\mathrm{d}^2 \ell({\sigma_0}^2 | \sigma^2, \theta)} ' +
            r'{\mathrm{d} ({\sigma_0}^2)^2}$')

        if self.use_log_scale:
            ax[1, 0].set_ylabel(
                r'$\frac{\mathrm{d}^2 \ell({\sigma_0}^2 | \sigma^2, ' +
                r'\theta)} {\mathrm{d} {\sigma_0}^2 \mathrm{d} (\ln \theta)}$')
        else:
            ax[1, 0].set_ylabel(
                r'$\frac{\mathrm{d}^2 \ell({\sigma_0}^2 | \sigma^2, ' +
                r'\theta)} {\mathrm{d} {\sigma_0}^2 \mathrm{d} \theta}$')

        ax[1, 1].set_ylabel(
            r'$\frac{\mathrm{d}^2 \ell({\sigma_0}^2 | \sigma^2, \theta)} ' +
            r'{\mathrm{d} {\sigma_0}^2 \mathrm{d} \sigma^2}$')
        ax[0, 0].set_title(r'Log likelihood function, given ' +
                           r'$(\sigma^2, \theta)$')
        ax[0, 1].set_title(r'First derivative of log likelihood, given ' +
                           r'$(\sigma^2, \theta)$')
        ax[0, 2].set_title(r'Second derivative of log likelihood, given ' +
                           r'$(\sigma^2, \theta)$')
        ax[1, 0].set_title(r'Second mixed derivative of log likelihood, ' +
                           r'given $(\sigma^2, \theta)$')
        ax[1, 1].set_title(r'Second mixed derivative of log likelihood, ' +
                           r'given $(\sigma^2, \theta)$')
        ax[0, 0].grid(True, which='both')
        ax[0, 1].grid(True, which='both')
        ax[0, 2].grid(True, which='both')
        ax[1, 0].grid(True, which='both')
        ax[1, 1].grid(True, which='both')

        ax[1, 2].set_axis_off()

        plt.tight_layout()
        plt.show()

    # ===================================
    # plot likelihood versus sigma0 sigma
    # ===================================

    def plot_likelihood_versus_sigma0_sigma(self, result=None):
        """
        2D contour plot of log likelihood versus sigma0 and sigma.
        """

        load_plot_settings()

        # Optimal point
        optimal_sigma = result['hyperparam']['sigma']
        optimal_sigma0 = result['hyperparam']['sigma0']
        optimal_scale = result['hyperparam']['scale']
        optimal_ell = result['optimization']['max_likelihood']

        self.cov.set_scale(optimal_scale)

        # Intervals cannot contain origin point as ell is minus infinity.
        sigma0 = numpy.linspace(0.02, 0.25, 50)
        sigma = numpy.linspace(0.02, 0.25, 50)
        ell = numpy.zeros((sigma0.size, sigma.size))
        for i in range(sigma0.size):
            for j in range(sigma.size):
                ell[i, j] = self.likelihood(
                        False, numpy.array([sigma[j], sigma0[i]]))

        # Convert inf to nan
        ell = numpy.where(numpy.isinf(ell), numpy.nan, ell)

        # Smooth data for finer plot
        # sigma_ = [2, 2]  # in unit of data pixel size
        # ell = scipy.ndimage.filters.gaussian_filter(
        #         ell, sigma_, mode='nearest')

        # Increase resolution for better contour plot
        N = 300
        f = scipy.interpolate.interp2d(sigma, sigma0, ell, kind='cubic')
        sigma_fine = numpy.linspace(sigma[0], sigma[-1], N)
        sigma0_fine = numpy.linspace(sigma0[0], sigma0[-1], N)
        x, y = numpy.meshgrid(sigma_fine, sigma0_fine)
        ell_fine = f(sigma_fine, sigma0_fine)

        # We will plot the difference of max of ell to ell, called z
        # max_ell = numpy.abs(numpy.max(ell_fine))
        # z = max_ell - ell_fine
        z = ell_fine
        # x, y = numpy.meshgrid(sigma, sigma0)
        # z = ell

        # Cut data
        # cut_data = 0.92
        # clim = 0.87
        # z[z>CutData] = CutData   # Used for plotting data without prior

        # Min and max of data
        min_z = numpy.min(z)
        max_z = numpy.max(z)

        fig, ax = plt.subplots(ncols=3, figsize=(17, 5))

        # Adjust bounds of a colormap
        def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=2000):
            new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                'trunc({n},{a:.2f},{b:.2f})'.format(
                    n=cmap.name, a=minval, b=maxval),
                cmap(numpy.linspace(minval, maxval, n)))
            return new_cmap

        # cmap = plt.get_cmap('gist_stern_r')
        # cmap = plt.get_cmap('rainbow_r')
        # cmap = plt.get_cmap('nipy_spectral_r')
        # cmap = plt.get_cmap('RdYlGn')
        # cmap = plt.get_cmap('ocean')
        # cmap = plt.get_cmap('gist_stern_r')
        # cmap = plt.get_cmap('RdYlBu')
        # cmap = plt.get_cmap('gnuplot_r')
        # cmap = plt.get_cmap('Spectral')
        cmap = plt.get_cmap('gist_earth')
        colormap = truncate_colormap(cmap, 0, 1)
        # colormap = truncate_colormap(cmap, 0.2, 0.9)  # for ocean

        # Contour fill Plot
        levels = numpy.linspace(min_z, max_z, 2000)
        c = ax[0].contourf(x, y, z, levels, cmap=colormap, zorder=-9)
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(c, cax=cax, orientation='vertical')
        cbar.ax.set_ylabel(r'$\ell(\eta, \theta)$')
        # c.set_clim(0, clim)
        # cbar.set_ticks([0,0.3,0.6,0.9,1])

        # Find max for each fixed sigma
        opt_sigma01 = numpy.zeros((sigma_fine.size, ), dtype=float)
        opt_ell1 = numpy.zeros((sigma_fine.size, ), dtype=float)
        opt_ell1[:] = numpy.nan
        for j in range(sigma_fine.size):
            if numpy.all(numpy.isnan(ell_fine[:, j])):
                continue
            max_index = numpy.nanargmax(ell_fine[:, j])
            opt_sigma01[j] = sigma_fine[max_index]
            opt_ell1[j] = ell_fine[max_index, j]
        ax[0].plot(sigma_fine, opt_sigma01, color='red',
                   label=r'$\hat{\sigma}_0(\sigma)$')
        ax[1].plot(sigma_fine, opt_ell1, color='red')

        # Find max for each fixed sigma0
        opt_sigma2 = numpy.zeros((sigma0_fine.size, ), dtype=float)
        opt_ell2 = numpy.zeros((sigma0_fine.size, ), dtype=float)
        opt_ell2[:] = numpy.nan
        for i in range(sigma0_fine.size):
            if numpy.all(numpy.isnan(ell_fine[i, :])):
                continue
            max_index = numpy.nanargmax(ell_fine[i, :])
            opt_sigma2[i] = sigma0_fine[max_index]
            opt_ell2[i] = ell_fine[i, max_index]
        ax[0].plot(opt_sigma2, sigma0_fine, color='black',
                   label=r'$\hat{\sigma}(\sigma_0)$')
        ax[2].plot(sigma0_fine, opt_ell2, color='black')

        # Plot max of the whole 2D array
        max_indices = numpy.unravel_index(numpy.nanargmax(ell_fine),
                                          ell_fine.shape)
        opt_sigma0 = sigma0_fine[max_indices[0]]
        opt_sigma = sigma_fine[max_indices[1]]
        opt_ell = ell_fine[max_indices[0], max_indices[1]]
        ax[0].plot(opt_sigma, opt_sigma0, 'o', color='red', markersize=6,
                   label=r'$(\hat{\sigma}_0, \hat{\sigma})$ (by brute force ' +
                         r'on grid)')
        ax[1].plot(opt_sigma, opt_ell, 'o', color='red',
                   label=r'$\ell(\hat{\sigma}, \hat{\sigma}_0)$ by brute ' +
                         r'force on grid)')
        ax[2].plot(opt_sigma0, opt_ell, 'o', color='black',
                   label=r'$\ell(\hat{\sigma}_0, \hat{\sigma})$ (by brute ' +
                         r'force on grid)')

        # Plot optimal point as found by the profile likelihood method
        ax[0].plot(optimal_sigma, optimal_sigma0, 'X', color='black',
                   markersize=6,
                   label=r'$\max_{\sigma_0, \sigma} \ell$ (by optimization)')
        ax[1].plot(optimal_sigma, optimal_ell, 'X', color='red',
                   label=r'$\ell(\hat{\sigma}_0, \hat{\sigma})$ (by ' +
                         r'optimization)')
        ax[2].plot(optimal_sigma0, optimal_ell, 'X', color='black',
                   label=r'$\ell(\hat{\sigma}_0, \hat{\sigma})$ (by ' +
                         r'optimization)')

        # Plot annotations
        ax[0].legend()
        ax[1].legend()
        ax[2].legend()
        ax[0].set_xlim([sigma[0], sigma[-1]])
        ax[1].set_xlim([sigma[0], sigma[-1]])
        ax[2].set_xlim([sigma0[0], sigma0[-1]])
        ax[0].set_ylim([sigma0[0], sigma0[-1]])
        # ax[0].set_xscale('log')
        # ax[1].set_xscale('log')
        # ax[2].set_xscale('log')
        # ax[0].set_yscale('log')
        ax[0].set_xlabel(r'$\sigma$')
        ax[1].set_xlabel(r'$\sigma$')
        ax[2].set_xlabel(r'$\sigma_0$')
        ax[0].set_ylabel(r'$\sigma_0$')
        ax[1].set_ylabel(r'$\ell(\hat{\sigma}_0(\sigma), \sigma)$')
        ax[2].set_ylabel(r'$\ell(\sigma_0, \hat{\sigma}(\sigma_0))$')
        ax[0].set_title('Log likelihood function')
        ax[1].set_title(r'Log Likelihood profiled over $\sigma$ ')
        ax[2].set_title(r'Log likelihood profiled over $\sigma_0$')
        ax[1].grid(True)
        ax[2].grid(True)

        plt.tight_layout()
        plt.show()
