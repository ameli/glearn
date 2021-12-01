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
from scipy.special import beta as beta_function
from .prior import Prior


# ==========
# Beta Prime
# ==========

class BetaPrime(Prior):
    """
    Beta Prime distribution.
    """

    # ====
    # init
    # ====

    def __init__(self, alpha=1.0, beta=1.0):
        """
        Initialization.
        """

        # This distribution does not have half distribution
        half = False
        super().__init__(half)

        # Check arguments
        self.alpha, self.beta = self._check_arguments(alpha, beta)

    # ===============
    # check arguments
    # ===============

    def _check_arguments(self, alpha, beta):
        """
        Checks user input arguments. Also, converts input arguments to numpy
        array.
        """

        # Check type
        if numpy.isscalar(alpha) and not isinstance(alpha, (int, float)):
            raise TypeError('"alpha" should be a float number.')

        if numpy.isscalar(beta) and not isinstance(beta, (int, float)):
            raise TypeError('"beta" should be a float number.')

        # Convert scalar inputs to numpy array
        if numpy.isscalar(alpha):
            alpha = numpy.array([alpha], dtype=float)
        elif isinstance(alpha, list):
            alpha = numpy.array(alpha, dtype=float)
        elif not isinstance(alpha, numpy.ndarray):
            raise TypeError('"alpha" should be a scalar, list of numpy ' +
                            'array.')

        # Convert scalar inputs to numpy array
        if numpy.isscalar(beta):
            beta = numpy.array([beta], dtype=float)
        elif isinstance(beta, list):
            beta = numpy.array(beta, dtype=float)
        elif not isinstance(beta, numpy.ndarray):
            raise TypeError('"beta" should be a scalar, list of numpy array.')

        if any(alpha <= 0.0):
            raise ValueError('"alpha" should be positive.')
        if any(beta <= 0.0):
            raise ValueError('"beta" should be positive.')

        return alpha, beta

    # ========================
    # suggest hyperparam guess
    # ========================

    def suggest_hyperparam_guess(self):
        """
        Suggests a guess for the hyperparam based on the prior distribution.
        """

        hyperparam_guess = numpy.zeros_like(self.shape)

        for i in range(self.shape):
            # Mean or mode could be used for initial hyperparam guess
            if self.beta[i] > 1.0:
                mean = self.alpha[i] / (self.beta[i] - 1.0)
                hyperparam_guess[i] = mean
            elif self.alpha[i] >= 1.0:
                mode = (self.alpha[i] - 1.0) / (self.beta[i] + 1.0)
                hyperparam_guess[i] = mode
            else:
                # mean and mode are infinity. Just set any finite number
                hyperparam_guess[i] = 1.0

        return hyperparam_guess

    # ===========
    # check param
    # ===========

    def _check_param(self, x):
        """
        Checks the input x.
        """

        # Convert input to numpy array
        if numpy.isscalar(x):
            x_ = numpy.array([x], dtype=float)
        elif isinstance(x, list):
            x_ = numpy.array(x, dtype=float)
        elif isinstance(x, numpy.ndarray):
            x_ = x
        else:
            raise TypeError('"x" should be scalar, list, or numpy ' +
                            'array.')

        # Match the size of self.beta and self.alpha with size of input x
        if x_.size == self.beta.size and x_.size == self.alpha.size:
            beta_ = self.beta
            alpha_ = self.alpha
        elif self.beta.size == 1 and self.alpha.size == 1:
            beta_ = numpy.tile(self.beta, x_.size)
            alpha_ = numpy.tile(self.alpha, x_.size)
        else:
            raise ValueError('Size of "x" and "self.beta" or "self.alpha" ' +
                             'do not match.')

        return x_, alpha_, beta_

    # ===
    # pdf
    # ===

    def pdf(self, x):
        """
        Returns the log-prior function for an array of hyperparameter. It is
        assumed that priors for each of hyperparameters are independent. The
        overall log-prior is the sum of log-prior for each hyperparameter.
        """

        # Convert x or self.beta to arrays of the same size
        x, alpha, beta = self._check_param(x)

        pdf_ = numpy.zeros((x.size, ), dtype=float)
        for i in range(x.size):
            coeff = 1.0 / beta_function(alpha, beta)
            a = alpha[i] - 1.0
            b = -alpha[i] - beta[i]
            k = (1.0 + x[i])**b
            m = x[i]**a
            pdf_[i] = coeff * m * k

        return pdf_

    # ============
    # pdf jacobian
    # ============

    def pdf_jacobian(self, x):
        """
        Returns the first derivative of log-prior function for an array of
        hyperparameter input.
        """

        # Convert x or self.beta to arrays of the same size
        x, alpha, beta = self._check_param(x)

        pdf_jacobian_ = numpy.zeros((x.size, ), dtype=float)

        for i in range(x.size):
            coeff = 1.0 / beta_function(alpha, beta)
            a = alpha[i] - 1.0
            b = -alpha[i] - beta[i]
            k = (1.0 + x[i])**b
            m = x[i]**a
            pdf_jacobian_[i] = coeff * m * k * (a/x[i] + b/(x[i]+1.0))

        return pdf_jacobian_

    # ===========
    # pdf hessian
    # ===========

    def pdf_hessian(self, x):
        """
        Returns the second derivative of log-prior function for an array of
        hyperparameter input.
        """

        # Convert x or self.beta to arrays of the same size
        x, alpha, beta = self._check_param(x)

        pdf_hessian_ = numpy.zeros((x.size, x.size), dtype=float)

        for i in range(x.size):
            coeff = 1.0 / beta_function(alpha, beta)
            a = alpha[i] - 1.0
            b = -alpha[i] - beta[i]
            k = (1.0 + x[i])**b
            m = x[i]**a
            pdf_hessian_[i, i] = coeff * m * k * \
                ((a**2/x[i]**2) - a/x[i]**2 + 2.0*a*b/(x[i]*(x[i]+1.0)) +
                    b**2/((x[i]+1.0)**2) - b/((x[i]+1.0)**2))

        return pdf_hessian_
