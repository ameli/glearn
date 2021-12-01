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
# Cauchy
# ======

class Cauchy(Prior):
    """
    Cauchy distribution.
    """

    # ====
    # init
    # ====

    def __init__(self, median=0.0, scale=1.0, half=True):
        """
        Initialization.
        """

        super().__init__(half)

        # Check arguments
        self.median, self.scale = self._check_arguments(median, scale)

    # ===============
    # check arguments
    # ===============

    def _check_arguments(self, median, scale):
        """
        Checks user input arguments. Also, converts input arguments to numpy
        array.
        """

        # Check type
        if numpy.isscalar(median) and not isinstance(median, (int, float)):
            raise TypeError('"median" should be a float number.')

        if numpy.isscalar(scale) and not isinstance(scale, (int, float)):
            raise TypeError('"scale" should be a float number.')

        # Convert scalar inputs to numpy array
        if numpy.isscalar(median):
            median = numpy.array([median], dtype=float)
        elif isinstance(median, list):
            median = numpy.array(median, dtype=float)
        elif not isinstance(median, numpy.ndarray):
            raise TypeError('"median" should be a scalar, list of numpy ' +
                            'array.')

        # Convert scalar inputs to numpy array
        if numpy.isscalar(scale):
            scale = numpy.array([scale], dtype=float)
        elif isinstance(scale, list):
            scale = numpy.array(scale, dtype=float)
        elif not isinstance(scale, numpy.ndarray):
            raise TypeError('"scale" should be a scalar, list of numpy array.')

        if any(scale <= 0.0):
            raise ValueError('"scale" should be positive.')

        return median, scale

    # ========================
    # suggest hyperparam guess
    # ========================

    def suggest_hyperparam_guess(self):
        """
        Suggests a guess for the hyperparam based on the prior distribution.
        """

        # Mediam of distribution (could be used for initial hyperparam guess)
        if self.half:
            hyperparam_guess = self.scale
        else:
            hyperparam_guess = self.median

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

        # Match the size of self.scale and self.median with size of input x
        if x_.size == self.scale.size and x_.size == self.median.size:
            scale_ = self.scale
            median_ = self.median
        elif self.scale.size == 1 and self.median.size == 1:
            scale_ = numpy.tile(self.scale, x_.size)
            median_ = numpy.tile(self.median, x_.size)
        else:
            raise ValueError('Size of "x" and "self.scale" or "self.median" ' +
                             'do not match.')

        return x_, median_, scale_

    # ===
    # pdf
    # ===

    def pdf(self, x):
        """
        Returns the log-prior function for an array of hyperparameter. It is
        assumed that priors for each of hyperparameters are independent. The
        overall log-prior is the sum of log-prior for each hyperparameter.
        """

        # Convert x or self.scale to arrays of the same size
        x, median, scale = self._check_param(x)

        pdf_ = numpy.zeros((x.size, ), dtype=float)
        for i in range(x.size):
            k = 1.0 + ((x[i]-median[i])/scale[i])**2
            pdf_[i] = 1.0 / (scale[i] * numpy.pi * k)

        return pdf_

    # ============
    # pdf jacobian
    # ============

    def pdf_jacobian(self, x):
        """
        Returns the first derivative of log-prior function for an array of
        hyperparameter input.
        """

        # Convert x or self.scale to arrays of the same size
        x, median, scale = self._check_param(x)

        pdf_jacobian_ = numpy.zeros((x.size, ), dtype=float)

        for i in range(x.size):
            k = 1.0 + ((x[i]-median[i])/scale[i])**2
            pdf_jacobian_[i] = -2.0*x[i] / (scale[i]**3 * numpy.pi * k**2)

        return pdf_jacobian_

    # ===========
    # pdf hessian
    # ===========

    def pdf_hessian(self, x):
        """
        Returns the second derivative of log-prior function for an array of
        hyperparameter input.
        """

        # Convert x or self.scale to arrays of the same size
        x, median, scale = self._check_param(x)

        pdf_hessian_ = numpy.zeros((x.size, x.size), dtype=float)

        for i in range(x.size):
            k = 1.0 + ((x[i]-median[i])/scale[i])**2
            pdf_hessian_[i, i] = 8.0*x[i]**2 / \
                (scale[i]**5 * numpy.pi * k**3) - \
                2.0 / (scale[i]**3 * numpy.pi * k**2)

        return pdf_hessian_
