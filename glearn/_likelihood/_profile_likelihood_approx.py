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

        # Storing variables to prevent redundant updates
        self.asym_poly = None
        self.asym_roots = None
        self.asym_C = None

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
        CXtz = self.asym_C @ Xtz
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

            Rz = self._Q_dot(self.z)
            zRz = numpy.dot(self.z, Rz)
            z_Rnorm = numpy.sqrt(zRz)
            zc = self.z / z_Rnorm

            # Powers of N
            Nzc = self._N_dot(zc)
            N2zc = self._N_dot(Nzc)
            if degree == 2:
                N3zc = self._N_dot(N2zc)
                N4zc = self._N_dot(N3zc)

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
            A0zc = -self._Q_dot(mtrN*zc - Nzc)
            A1zc = self._Q_dot(mtrN*Nzc + mtrN2*zc - 2.0*N2zc)
            if degree == 2:
                A2zc = -self._Q_dot(mtrN*N2zc + mtrN2*Nzc - 2.0*N3zc)
                A3zc = self._Q_dot(mtrN2*N2zc - N4zc)

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

    # ===================
    # maximize likelihood
    # ===================

    def maximaize_likelihood(self, degree=2):
        """
        Approximates the maxima of the likelihood based on the zeros of the
        asymptotic relation of the first derivative of likelihood w.r.t eta.
        If the second derivative at the root is negative, the root is maxima.
        """

        if degree != 1 and degree != 2:
            raise ValueError('Asymptotic polynomial degree should be either ' +
                             '"1" or "2".')

        # Ensure asymptotes are calculated
        self._compute_polynomial_coeff(degree=degree)

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
            asym_d2ell_deta2 = self._likelihood_der2_eta(
                    roots[i], degree=degree)
            if asym_d2ell_deta2 <= 0.0:
                asym_maxima.append(roots[i])

        return asym_maxima

    # ===================
    # likelihood der1 eta
    # ===================

    def _likelihood_der1_eta(self, eta, degree=2):
        """
        Computes the asymptote of the likelihood first derivative w.r.t eta.
        """

        if degree != 1 and degree != 2:
            raise ValueError('Asymptotic polynomial degree should be either ' +
                             '"1" or "2".')

        # Ensure asymptotes are calculated
        self._compute_polynomial_coeff(degree=degree)

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

    # ===================
    # likelihood der2 eta
    # ===================

    def _likelihood_der2_eta(self, eta, degree=2):
        """
        Computes the asymptote of the likelihood second derivative w.r.t eta.
        """

        if degree != 1 and degree != 2:
            raise ValueError('Asymptotic polynomial degree should be either ' +
                             '"1" or "2".')

        # Ensure asymptotes are calculated
        self._compute_polynomial_coeff(degree=degree)

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
