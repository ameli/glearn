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

__all__ = ['ProfileLikelihoodApprox']


# =========================
# Profile Likelihood Approx
# =========================

class ProfileLikelihoodApprox(BaseLikelihood):
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

        # Configuration
        self.hyperparam_tol = 1e-8

        # Storing variables to prevent redundant updates
        self.poly_coeffs = None
        self.C = None

        # Store hyperparam used to compute polynomial coefficients
        self.scale_hyperparam = None

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

    def _Q_dot(self, z):
        """
        Matrix-vector multiplication Q*z where Q is defined by

            Q = I - X * (X.T * X)^{-1} * X.T

        where X is (n, m) matrix. If n is large, direct computation of Q is
        inefficient. Hence, an implicit matrix-vector operation is preferred.
        """

        Xtz = numpy.dot(self.X.T, z)
        C = self._get_C()
        CXtz = C @ Xtz
        XCXtz = numpy.dot(self.X, CXtz)

        Qz = z - XCXtz

        return Qz

    # =====
    # N dot
    # =====

    def _N_dot(self, z):
        """
        Matrix-vector multiplication N*z where N is defined by:

            N = K * Q
        """

        K = self.mixed_cor.get_matrix(0.0)
        Qz = self._Q_dot(z)
        Nz = K @ Qz

        return Nz

    # =====
    # get C
    # =====

    def _get_C(self):
        """
        Computes C is it has bot been already, and returns it.
        """

        # C is the inv(X.T*X)
        if self.C is None:
            Cinv = self.X.T @ self.X
            self.C = numpy.linalg.inv(Cinv)

        return self.C

    # ===========
    # compute tau
    # ===========

    def _compute_tau(self, K, XtKiX, i):
        """
        Computes the trace of N^i where N = K @ Q, where Q is
        Q = I - X @(X.T @ X)^{-1} @ Xt

        Let define Li = XtKiX, and Ki = K^i, and C = inv(X.T @ X), P = XCXt.
        Thus,
            Q = I - P, and N = K - KP.

        Example: here we find the trace of N^3, which is
            trace(N^3) = (K - KP) (K - KP) (K - KP)
                       = KKK - KKP - KPK + KPP - PKK + PKP + PPK - PPP

        There are 8=2**3 components in the above. The sign of each component is
        determined by the products of (-P).

        Since trace has cyclic property on the product of matrices, trace of
        powers of K@Q is to find in which cyclic order the matrix K and P
        are arranged. For example

            trace(KKP) = trace(PKK) = trace(CL2)
            trace(KPP) = trace(PPK) = trace(PPL1)
            ...

        and so on. Overall, the sequence of 8 components

            KKK - KKP - KPK + KPP - PKK + PKP + PPK - PPP   (original sequence)

        can be coded to

            K3 - CL2 - CL2 + CCL1 - CL2 + CCL1 + CCL1 - CCC    (coded sequence)

        which in reality is

            trace(K^3) - trace(C@L2) - trace(C@L2) ...

        So, the problem is how to find the coded sequence from the original
        sequence in the above. For each P in the sequence, we assign the
        integer m, which denote the number of times K is followed P, in cyclic
        sense. Then write the sequence only with Ps and their m, with out any K
        such as in the examples below:

            PPP => P(m=0) P(m=0) P(m=0)
            PPK => P(m=0) P(m=1)
            KPP => PPK => P(m=0) P(m=1)
            KPK => PKK => P(m=2)

        To code the above from P/K to C/L matrices replace P with C followed
        by L(m+1). For the above examples:

            PPP => P(m=0) P(m=0) P(m=0)     => C L1 C L1 C L1
            PPK => P(m=0) P(m=1)            => C L1 C L2
            KPP => PPK => P(m=0) P(m=1)     => C L1 C L2
            KPK => PKK => P(m=2)            => C L3

        and the sign of each component is determined by the number of (-C) in
        the component.

        The first component of the sequence (KKK) where there is no P in there
        is an exception, since KKK is coded to K3 itself.

        To code the above, we use the binary system, where 0 is K and 1 is P.

            KKK => 000 => + K3
            KKP => 001 => - C @ L3
            KPK => 010 => - C @ L3
            KPP => 011 => + C @ L1 @ C @ L2
            PKK => 100 => - C @ L3
            PKP => 101 => + C @ L1 @ C @ L2
            PPK => 110 => + C @ L1 @ C @ L2
            PPP => 111 => - C @ L1 @ C @ L1 @ C @ L1
        """

        # At i = 0, we have two cases:
        # 1. If Binv is zero: tau = trace(Q @ N^0)/rdof = (n-m) / (n-m) = 1
        # 2. If Binv is not zero, tau = trace(K_tilde^0) / n = 1
        if i == 0:
            tau = 1.0
            return tau

        # C is the inverse of X.T @ X
        C = self._get_C()

        # Initialize the trace of all components. The first component of the
        # sequence is K..K = Ki which has no P in it.
        trace = imate.trace(K, p=i, method='exact')

        # For the rest of components where there is at least one P in the K/P
        # sequence, we add their trace as follows.
        for j in range(1, 2**i):

            # Initialize a matrix to be the product of C/L matrices for each
            # component. For example, the component KKP is coded to
            # C L1 C L2, and hence G = C @ L1 @ C @ L2.
            G = numpy.eye(self.X.shape[1], dtype=float)

            # Code i to binary. For example i=6 is 110. 1 correspond to P and
            # 0 correspond to K.
            code = numpy.base_repr(j).zfill(i)

            # Find the first P (or 1) in code
            P_index = [i for i in range(len(code)) if code.startswith('1', i)]
            P_index = numpy.asarray(P_index, dtype=int)

            # Find m for each P. m is the number of Ks followed after each P.
            # For example if we have KPPKKP, then the m for each of the three P
            # is 0, 2, 1. For the last P, we used the cyclic pattern.
            m = numpy.zeros_like(P_index)
            for k in range(P_index.size):
                # For the last P, use cyclic pattern to count not only all Ks
                # after P, but also Ks in the beginning.
                if k == P_index.size-1:
                    # Count Ks after P till end of code
                    m[k] = len(code) - (P_index[k]+1)

                    # Count Ks in the beginning
                    m[k] += P_index[0]
                else:
                    m[k] = P_index[k+1] - P_index[k] - 1

            # Form G by multiplications with C or Li. The rule is, for each P,
            # multiply G with C @ L(m+1) of the m corresponding to that P.
            # Recall that Li = X.T @ K^i @ X
            for k in range(P_index.size):
                G = G @ (-C) @ XtKiX[m[k]+1]

            # Add the trace of G to the trace
            trace += numpy.trace(G)

        # Normalize trace to to the residual degrees of freedom.
        tau = trace / self.rdof

        return tau

    # ========================
    # compute polynomial coeff
    # ========================

    def _compute_polynomial_coeff(self, degree=2):
        """
        Returns the asymptotic polynomial coefficients for the first derivative
        of likelihood w.r.t eta.

        If degree is 1 but the stored polynomial is of the second order (degree
        2), then it returns the first two entries of the polynomial coeffs.
        """

        if degree < 1:
            raise ValueError('Polynomial degree should be at least 1.')

        # Get the current scale hyperparam to compare with the previous scale
        # to check if scale has been changed. If not, the previous computation
        # of the polynomial coefficients could be used.
        scale = self.cov.get_scale()

        # Check if polynomial coeffs need to be computed
        if (self.poly_coeffs is None) or \
                (self.poly_coeffs.size != 2*degree) or \
                (self.scale_hyperparam is None) or \
                (not numpy.allclose(scale, self.scale_hyperparam,
                                    atol=self.hyperparam_tol)):

            Qz = self._Q_dot(self.z)
            zQz = numpy.dot(self.z, Qz)
            z_Qnorm = numpy.sqrt(zQz)
            zq = self.z / z_Qnorm

            # Compute N^i * zq. With i = 0, 1, ..., (degree+1)
            Nizq = [None] * (2*degree+1)
            Nizq[0] = zq
            for i in range(1, 2*degree+1):
                Nizq[i] = self._N_dot(Nizq[i-1])

            # Compute K^i * X. Only half of powers of i are needed.
            KiX = [None] * (degree+1)
            KiX[0] = self.X
            K = self.mixed_cor.get_matrix(0.0)
            for i in range(1, degree+1):
                KiX[i] = K @ KiX[i-1]

            # Computing X.T * N^i * X for i = 1, ..., degree+1. The first one,
            # which is XtKX[0] is None and will not be used, and only used to
            # keep the indices i start from 1 to degree+1.
            XtKiX = [None] * (2*degree+1)
            for i in range(1, 2*degree+1):
                if i % 2 == 0:
                    XtKiX[i] = KiX[i//2].T @ KiX[i//2]
                else:
                    XtKiX[i] = KiX[i//2].T @ KiX[i//2+1]

            # Traces of N^i, i = 1, ..., (degree+1)
            tau = numpy.zeros((degree+1, ), dtype=float)
            for i in range(degree+1):
                tau[i] = self._compute_tau(K, XtKiX, i)

            # Compute matrices Ai*zq for i = 0, ..., degree-1
            Aizq = [None] * (2*degree)
            for i in range(degree):
                Bzq = numpy.zeros_like(zq, dtype=float)
                for j in range(i+2):
                    Bzq += tau[i+1-j] * Nizq[j]
                Bzq -= (i+2)*Nizq[i+1]
                Aizq[i] = (-1.0)**(i+1) * self._Q_dot(Bzq)

            # Compute matrices Ai*zq for i = degree, ..., 2*degree
            for i in range(degree, 2*degree):
                Bzq = numpy.zeros_like(zq, dtype=float)
                for j in range(i+1-degree, degree+1):
                    Bzq += tau[i+1-j] * Nizq[j]
                Bzq -= (2*degree-i)*Nizq[i+1]
                Aizq[i] = (-1.0)**(i+1) * self._Q_dot(Bzq)

            # Coefficients of polynomial
            self.poly_coeffs = numpy.zeros((2*degree, ), dtype=float)
            for i in range(2*degree):
                self.poly_coeffs[i] = numpy.dot(zq, Aizq[i])

            # Store scale to avoid repetitive computation if this function is
            # called with the same scale.
            self.scale_hyperparam = scale

        return self.poly_coeffs

    # ===================
    # maximize likelihood
    # ===================

    def maximaize_likelihood(self, degree=2):
        """
        Approximates the maxima of the likelihood based on the zeros of the
        asymptotic relation of the first derivative of likelihood w.r.t eta.
        If the second derivative at the root is negative, the root is maxima.
        """

        if degree < 1:
            raise ValueError('Polynomial degree should be at least 1.')

        # Ensure polynomial coefficients are calculated
        self._compute_polynomial_coeff(degree=degree)

        # All roots
        poly_roots = numpy.roots(self.poly_coeffs)

        # Remove complex roots
        poly_roots = numpy.sort(numpy.real(
            poly_roots[numpy.abs(numpy.imag(poly_roots)) < 1e-10]))

        # Remove negative roots
        poly_roots = poly_roots[poly_roots >= 0.0]

        # Output
        maxima = []

        # Check sign of the second derivative
        for i in range(poly_roots.size):

            # Compute second derivative on each root of the first derivative
            d2ell_deta2 = self._likelihood_der2_eta(
                    poly_roots[i], degree=degree)

            if d2ell_deta2 <= 0.0:
                maxima.append(poly_roots[i])

        return maxima

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

    # ==============
    # likelihood inf
    # ==============

    def _likelihood_inf(self):
        """
        Returns the value of likelihood function at eta=infinity.
        """

        # Optimal sigma02 when eta is very large
        sigma02 = self._find_optimal_sigma02()

        # Log likelihood
        ell = -0.5*self.rdof * (numpy.log(2.0*numpy.pi) + 1.0 +
                                numpy.log(sigma02))

        if self.B is None:
            Cinv = numpy.matmul(self.X.T, self.X)
            logdet_Cinv = numpy.log(numpy.linalg.det(Cinv))
            ell += - 0.5*logdet_Cinv

        return ell

    # ==========
    # likelihood
    # ==========

    def likelihood(self, eta, degree=2):
        """
        Computes the likelihood first derivative w.r.t eta.
        """

        if degree < 1:
            raise ValueError('Polynomial degree should be at least 1.')

        # Ensure polynomial coefficients are calculated
        self._compute_polynomial_coeff(degree=degree)

        # Ensure array
        if numpy.isscalar(eta):
            eta_ = numpy.asarray([eta])
        else:
            eta_ = eta

        # Initialize output
        ell = numpy.zeros((eta_.size, ), dtype=float)

        # Initialize ell with constant of integration. This constant is the
        # value of ell at eta=infinity.
        ell[:] = self._likelihood_inf()

        # Add terms of polynomial which is the integral of polynomial derived
        # for its first derivative, dell_deta
        for i in range(eta_.size):
            for j in range(self.poly_coeffs.size):
                ell[i] += (1.0/(j+1.0)) * self.poly_coeffs[j] / (eta_[i]**j)
            ell[i] *= (0.5 * self.rdof / eta_[i])

        if numpy.isscalar(eta):
            return ell[0]
        else:
            return ell

    # ===================
    # likelihood der1 eta
    # ===================

    def _likelihood_der1_eta(self, eta, degree=2):
        """
        Computes the likelihood first derivative w.r.t eta.
        """

        if degree < 1:
            raise ValueError('Polynomial degree should be at least 1.')

        # Ensure polynomial coefficients are calculated
        self._compute_polynomial_coeff(degree=degree)

        # Ensure array
        if numpy.isscalar(eta):
            eta_ = numpy.asarray([eta])
        else:
            eta_ = eta

        # Initialize output
        dell_deta = numpy.zeros((eta_.size, ), dtype=float)

        for i in range(eta_.size):
            for j in range(self.poly_coeffs.size):
                dell_deta[i] += self.poly_coeffs[j] / (eta_[i]**j)
            dell_deta[i] *= (-0.5 * self.rdof / eta_[i]**2)

        if numpy.isscalar(eta):
            return dell_deta[0]
        else:
            return dell_deta

    # ===================
    # likelihood der2 eta
    # ===================

    def _likelihood_der2_eta(self, eta, degree=2):
        """
        Computes the likelihood second derivative w.r.t eta.
        """

        if degree < 1:
            raise ValueError('Polynomial degree should be at least 1.')

        # Ensure polynomial coefficients are calculated
        self._compute_polynomial_coeff(degree=degree)

        # Ensure array
        if numpy.isscalar(eta):
            eta_ = numpy.asarray([eta])
        else:
            eta_ = eta

        # Initialize output
        d2ell_deta2 = numpy.zeros((eta_.size, ), dtype=float)

        for i in range(eta_.size):
            for j in range(self.poly_coeffs.size):
                d2ell_deta2[i] += (j+2) * self.poly_coeffs[j] / (eta_[i]**j)
            d2ell_deta2[i] *= (0.5 * self.rdof / eta_[i]**3)

        if numpy.isscalar(eta):
            return d2ell_deta2[0]
        else:
            return d2ell_deta2
