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


# ======
# Normal
# ======

class Normal(Prior):
    """
    Normal distribution.
    """

    # ====
    # init
    # ====

    def __init__(self, mean=0.0, std=1.0, half=False):
        """
        Initialization.
        """

        super().__init__(half)

        # Check arguments
        self.mean, self.std = self._check_arguments(mean, std)

    # ===============
    # check arguments
    # ===============

    def _check_arguments(self, mean, std):
        """
        Checks user input arguments. Also, converts input arguments to numpy
        array.
        """

        # Check type
        if numpy.isscalar(mean) and not isinstance(mean, (int, float)):
            raise TypeError('"mean" should be a float number.')

        if numpy.isscalar(std) and not isinstance(std, (int, float)):
            raise TypeError('"std" should be a float number.')

        # Convert scalar inputs to numpy array
        if numpy.isscalar(mean):
            mean = numpy.array([mean], dtype=float)
        elif isinstance(mean, list):
            mean = numpy.array(mean, dtype=float)
        elif not isinstance(mean, numpy.ndarray):
            raise TypeError('"mean" should be a scalar, list of numpy ' +
                            'array.')

        # Convert scalar inputs to numpy array
        if numpy.isscalar(std):
            std = numpy.array([std], dtype=float)
        elif isinstance(std, list):
            std = numpy.array(std, dtype=float)
        elif not isinstance(std, numpy.ndarray):
            raise TypeError('"std" should be a scalar, list of numpy array.')

        if any(std <= 0.0):
            raise ValueError('"std" should be positive.')

        return mean, std

    # ========================
    # suggest hyperparam guess
    # ========================

    def suggest_hyperparam_guess(self):
        """
        Suggests a guess for the hyperparam based on the prior distribution.
        """

        if self.half:
            # For half-normal distirubiton, use std as inital hypepram guess.
            hyperparam_guess = self.std
        else:
            # Otherwise, use its mean.
            hyperparam_guess = self.mean

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

        # Match the size of self.std and self.mean with size of input x
        if x_.size == self.std.size and x_.size == self.mean.size:
            std_ = self.std
            mean_ = self.mean
        elif self.std.size == 1 and self.mean.size == 1:
            std_ = numpy.tile(self.std, x_.size)
            mean_ = numpy.tile(self.mean, x_.size)
        else:
            raise ValueError('Size of "x" and "self.std" or "self.mean" ' +
                             'do not match.')

        return x_, mean_, std_

    # ===
    # pdf
    # ===

    def pdf(self, x):
        """
        Returns the log-prior function for an array of hyperparameter. It is
        assumed that priors for each of hyperparameters are independent. The
        overall log-prior is the sum of log-prior for each hyperparameter.
        """

        # Convert x or self.std to arrays of the same size
        x, mean, std = self._check_param(x)

        pdf_ = numpy.zeros((x.size, ), dtype=float)
        for i in range(x.size):
            coeff = 1.0 / (std[i] * numpy.sqrt(2.0*numpy.pi))
            m = (x[i] - mean[i]) / std[i]
            k = numpy.exp(-0.5*m**2)
            pdf_[i] = coeff * k

        return pdf_

    # ============
    # pdf jacobian
    # ============

    def pdf_jacobian(self, x):
        """
        Returns the first derivative of log-prior function for an array of
        hyperparameter input.
        """

        # Convert x or self.std to arrays of the same size
        x, mean, std = self._check_param(x)

        pdf_jacobian_ = numpy.zeros((x.size, ), dtype=float)

        for i in range(x.size):
            coeff = 1.0 / (std[i] * numpy.sqrt(2.0*numpy.pi))
            m = (x[i] - mean[i]) / std[i]
            k = numpy.exp(-0.5*m**2)
            pdf_jacobian_[i] = -coeff * m * k / std[i]

        return pdf_jacobian_

    # ===========
    # pdf hessian
    # ===========

    def pdf_hessian(self, x):
        """
        Returns the second derivative of log-prior function for an array of
        hyperparameter input.
        """

        # Convert x or self.std to arrays of the same size
        x, mean, std = self._check_param(x)

        pdf_hessian_ = numpy.zeros((x.size, x.size), dtype=float)

        for i in range(x.size):
            coeff = 1.0 / (std[i] * numpy.sqrt(2.0*numpy.pi))
            m = (x[i] - mean[i]) / std[i]
            k = numpy.exp(-0.5*m**2)
            pdf_hessian_[i, i] = coeff * k * (m**2 - 1.0) / std[i]**2

        return pdf_hessian_
