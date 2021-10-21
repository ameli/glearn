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
from .prior import Prior


# =======
# Uniform
# =======

class Uniform(Prior):
    """
    Uniform distribution.
    """

    # ====
    # init
    # ====

    def __init__(self, a=0, b=numpy.inf):
        """
        Initialization.
        """

        # This distribution does not have half distribution
        half = False
        super().__init__(half)

        # Check arguments
        self.a, self.b = self._check_arguments(a, b)

        # Mean of distribution (could be used for initial hyperparam guess)
        self.mean = self._pdf_mean()

    # ===============
    # check arguments
    # ===============

    def _check_arguments(self, a, b):
        """
        Checks user input arguments. Also, converts input arguments to numpy
        array.
        """

        # Check type
        if numpy.isscalar(a) and not isinstance(a, (int, float)):
            raise TypeError('"a" should be a float number.')
        if numpy.isscalar(b) and not isinstance(b, (int, float)):
            raise TypeError('"b" should be a float number.')

        # Convert scalar inputs to numpy array
        if numpy.isscalar(a):
            a = numpy.array([a], dtype=float)
        elif isinstance(a, list):
            a = numpy.array(a, dtype=float)
        elif not isinstance(a, numpy.ndarray):
            raise TypeError('"a" should be a scalar, list of numpy array.')

        if numpy.isscalar(b):
            b = numpy.array([b], dtype=float)
        elif isinstance(b, list):
            b = numpy.array(b, dtype=float)
        elif not isinstance(b, numpy.ndarray):
            raise TypeError('"b" should be a scalar, list of numpy array.')

        # Check size of a and b
        if a.size != b.size:
            raise ValueError('Sizes of "a" and "b" do not match.')

        # Each element of "a" cannot be larger than the corresponding element
        # of "b"
        if any(a > b):
            raise ValueError('"a" cannot be larger than "b".')

        return a, b

    # ========================
    # suggest hyperparam guess
    # ========================

    def suggest_hyperparam_guess(self):
        """
        Suggests a guess for the hyperparam based on the prior distribution.
        """

        hyperparam_guess = numpy.zeros_like(self.a)

        for i in range(hyperparam_guess.size):

            if not numpy.isinf(numpy.abs(self.a[i])) and \
                    not numpy.isinf(self.b[i]):
                mean = 0.5 * (self.a[i] + self.b[i])
                hyperparam_guess[i] = mean

            elif numpy.isinf(numpy.abs(self.a[i])) and \
                    not numpy.isinf(self.b[i]):
                hyperparam_guess[i] = self.b[i] - 1.0

            elif not numpy.isinf(numpy.abs(self.a[i])) and \
                    numpy.isinf(self.b[i]):
                hyperparam_guess[i] = self.a[i] + 1.0

            else:
                # a and b are infinity. Just pick any finite number.
                hyperparam_guess[i]

        return hyperparam_guess

    # =======
    # check x
    # =======

    def _check_param(self, x):
        """
        If x is an array, the prior is the sum of log-priors for each
        x element.
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

        # Match the size of self.a and self.b with size of input x
        if x_.size == self.a.size:
            a_ = self.a
            b_ = self.b
        elif self.a.size == 1:
            a_ = numpy.tile(self.a, x_.size)
            b_ = numpy.tile(self.b, x_.size)
        else:
            raise ValueError('Size of "x" and "self.a" do not match.')

        return x_, a_, b_

    # ========
    # pdf mean
    # ========

    def _pdf_mean(self):
        """
        Returns the mean of pdf.
        """

        if numpy.isinf(numpy.abs(self.a)) or numpy.isinf(self.b):
            self.mean = numpy.nan
        else:
            self.mean = 0.5 * (self.b - self.a)

    # ===
    # pdf
    # ===

    def pdf(self, x):
        """
        Returns the log-prior function for an array hyperparameter. It is
        assumed that priors for each hyperparameters are independent. The
        overall log-prior is the sum of log-prior for each hyperparameter.
        """

        # Convert hyperparam or self.a, and self.b to arrays of the same size
        x, a, b = self._check_param(x)

        pdf_ = numpy.zeros((x.size, ), dtype=float)
        for i in range(x.size):

            if x[i] > b[i] or x[i] < a[i]:
                # Prior is zero (log prior is -inf) outside [a, b]
                pdf_[i] = 0.0

            elif numpy.isinf(b[i]) or numpy.isinf(numpy.abs(a[i])):
                # Improper prior is 1 (it log is 0) for semi-infinite intervals
                pdf_[i] = 1.0
            else:
                # Uniform proper prior between interval [a, b]
                pdf_[i] = 1.0 / (b[i] - a[i])

        return pdf_

    # ============
    # pdf jacobian
    # ============

    def pdf_jacobian(self, x):
        """
        Returns the first derivative of log-prior function for an array of
        hyperparameter input.
        """

        # Convert hyperparam or self.a, and self.b to arrays of the same size
        x, _, _ = self._check_param(x)

        pdf_jacobian_ = numpy.zeros((x.size, ), dtype=float)

        return pdf_jacobian_

    # ===========
    # pdf hessian
    # ===========

    def pdf_hessian(self, x):
        """
        Returns the second derivative of log-prior function for an array of
        hyperparameter input.
        """

        # Convert hyperparam or self.a, and self.b to arrays of the same size
        x, _, _ = self._check_param(x)

        pdf_hessian_ = numpy.zeros((x.size, x.size), dtype=float)

        return pdf_hessian_
