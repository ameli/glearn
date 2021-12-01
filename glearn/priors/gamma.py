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
from scipy.special import gamma
from .prior import Prior


# =====
# Gamma
# =====

class Gamma(Prior):
    """
    Gamma distribution.
    """

    # ====
    # init
    # ====

    def __init__(self, shape=1.0, rate=1.0):
        """
        Initialization.
        """

        # This distribution does not have half distribution
        half = False
        super().__init__(half)

        # Check arguments
        self.shape, self.rate = self._check_arguments(shape, rate)

    # ===============
    # check arguments
    # ===============

    def _check_arguments(self, shape, rate):
        """
        Checks user input arguments. Also, converts input arguments to numpy
        array.
        """

        # Check type
        if numpy.isscalar(shape) and not isinstance(shape, (int, float)):
            raise TypeError('"shape" should be a float number.')

        if numpy.isscalar(rate) and not isinstance(rate, (int, float)):
            raise TypeError('"rate" should be a float number.')

        # Convert scalar inputs to numpy array
        if numpy.isscalar(shape):
            shape = numpy.array([shape], dtype=float)
        elif isinstance(shape, list):
            shape = numpy.array(shape, dtype=float)
        elif not isinstance(shape, numpy.ndarray):
            raise TypeError('"shape" should be a scalar, list of numpy ' +
                            'array.')

        # Convert scalar inputs to numpy array
        if numpy.isscalar(rate):
            rate = numpy.array([rate], dtype=float)
        elif isinstance(rate, list):
            rate = numpy.array(rate, dtype=float)
        elif not isinstance(rate, numpy.ndarray):
            raise TypeError('"rate" should be a scalar, list of numpy array.')

        if any(shape <= 0.0):
            raise ValueError('"shape" should be positive.')
        if any(rate <= 0.0):
            raise ValueError('"rate" should be positive.')

        return shape, rate

    # ========================
    # suggest hyperparam guess
    # ========================

    def suggest_hyperparam_guess(self):
        """
        Suggests a guess for the hyperparam based on the prior distribution.
        """

        hyperparam_guess = numpy.zeros_like(self.shape)

        for i in range(self.shape):
            # Mean of distribution (could be used for initial hyperparam guess)
            mean = self.shape[i] / self.rate[i]
            hyperparam_guess[i] = mean

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

        # Match the size of self.rate and self.shape with size of input x
        if x_.size == self.rate.size and x_.size == self.shape.size:
            rate_ = self.rate
            shape_ = self.shape
        elif self.rate.size == 1 and self.shape.size == 1:
            rate_ = numpy.tile(self.rate, x_.size)
            shape_ = numpy.tile(self.shape, x_.size)
        else:
            raise ValueError('Size of "x" and "self.rate" or "self.shape" ' +
                             'do not match.')

        return x_, shape_, rate_

    # ===
    # pdf
    # ===

    def pdf(self, x):
        """
        Returns the log-prior function for an array of hyperparameter. It is
        assumed that priors for each of hyperparameters are independent. The
        overall log-prior is the sum of log-prior for each hyperparameter.
        """

        # Convert x or self.rate to arrays of the same size
        x, shape, rate = self._check_param(x)

        pdf_ = numpy.zeros((x.size, ), dtype=float)
        for i in range(x.size):
            coeff = rate[i]**shape[i] / gamma(shape)
            a = shape[i] - 1.0
            b = rate[i] * x[i]
            k = numpy.exp(-b)
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

        # Convert x or self.rate to arrays of the same size
        x, shape, rate = self._check_param(x)

        pdf_jacobian_ = numpy.zeros((x.size, ), dtype=float)

        for i in range(x.size):
            coeff = rate[i]**shape[i] / gamma(shape)
            a = shape[i] - 1.0
            b = rate[i] * x[i]
            k = numpy.exp(-b)
            m = x[i]**a
            pdf_jacobian_[i] = coeff * m * k * (a - b) / x[i]

        return pdf_jacobian_

    # ===========
    # pdf hessian
    # ===========

    def pdf_hessian(self, x):
        """
        Returns the second derivative of log-prior function for an array of
        hyperparameter input.
        """

        # Convert x or self.rate to arrays of the same size
        x, shape, rate = self._check_param(x)

        pdf_hessian_ = numpy.zeros((x.size, x.size), dtype=float)

        for i in range(x.size):
            coeff = rate[i]**shape[i] / gamma(shape)
            a = shape[i] - 1.0
            b = rate[i] * x[i]
            k = numpy.exp(-b)
            m = x[i]**a
            pdf_hessian_[i, i] = coeff * m * k * ((a - b)**2 - a) / x[i]**2

        return pdf_hessian_
