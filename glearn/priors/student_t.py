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


# =========
# Student T
# =========

class StudentT(Prior):
    """
    Student-t distribution.

    :param dof: Degrees of freedom (parameter :math:`\\nu`).
    :type dof: float, or numpy.ndarray
    """

    # ====
    # init
    # ====

    def __init__(self, dof=1.0, half=False):
        """
        Initialization.
        """

        super().__init__(half)

        # Check arguments
        self.dof = self._check_arguments(dof)

    # ===============
    # check arguments
    # ===============

    def _check_arguments(self, dof):
        """
        Checks user input arguments. Also, converts input arguments to numpy
        array.
        """

        # Check type
        if numpy.isscalar(dof) and not isinstance(dof, (int, float)):
            raise TypeError('"dof" should be a float number.')

        # Convert scalar inputs to numpy array
        if numpy.isscalar(dof):
            dof = numpy.array([dof], dtype=float)
        elif isinstance(dof, list):
            dof = numpy.array(dof, dtype=float)
        elif not isinstance(dof, numpy.ndarray):
            raise TypeError('"dof" should be a scalar, list of numpy array.')

        if any(dof <= 0.0):
            raise ValueError('"dof" should be positive.')

        return dof

    # ========================
    # suggest hyperparam guess
    # ========================

    def suggest_hyperparam_guess(self):
        """
        Suggests a guess for the hyperparam based on the prior distribution.
        """

        hyperparam_guess = numpy.zeros_like(self.dof)

        for i in range(hyperparam_guess.size):

            # std of distribution (could be used for initial hyperparam guess)
            if self.dof[i] > 2.0:
                std = numpy.sqrt(self.dof[i] / (self.dof[i] - 2.0))

                if self.half:
                    std = std * numpy.sqrt(2.0)

                hyperparam_guess[i] = std

            else:
                # mean and std are infinity. Just pick any finite number
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

        # Match the size of self.a and self.b with size of input x
        if x_.size == self.dof.size:
            dof_ = self.dof
        elif self.dof.size == 1:
            dof_ = numpy.tile(self.dof, x_.size)
        else:
            raise ValueError('Size of "x" and "self.a" do not match.')

        return x_, dof_

    # ===
    # pdf
    # ===

    def pdf(self, x):
        """
        Returns the log-prior function for an array of hyperparameter. It is
        assumed that priors for each of hyperparameters are independent. The
        overall log-prior is the sum of log-prior for each hyperparameter.
        """

        # Convert x or self.dof to arrays of the same size
        x, dof = self._check_param(x)

        pdf_ = numpy.zeros((x.size, ), dtype=float)
        for i in range(x.size):
            ex = 0.5 * (dof[i] + 1.0)
            coeff = gamma(ex) / \
                (numpy.sqrt(dof[i] * numpy.pi) * gamma(0.5*dof[i]))
            k = 1.0 + x[i]**2 / self.dof
            pdf_[i] = coeff * k**(-ex)

        return pdf_

    # ============
    # pdf jacobian
    # ============

    def pdf_jacobian(self, x):
        """
        Returns the first derivative of log-prior function for an array of
        hyperparameter input.
        """

        # Convert x or self.dof to arrays of the same size
        x, dof = self._check_param(x)

        pdf_jacobian_ = numpy.zeros((x.size, ), dtype=float)

        for i in range(x.size):
            ex = 0.5 * (dof[i] + 1.0)
            coeff = gamma(ex) / \
                (numpy.sqrt(dof[i] * numpy.pi) * gamma(0.5*dof[i]))
            k = 1.0 + x[i]**2 / dof[i]
            pdf_jacobian_[i] = -coeff * ex * k**(-ex-1.0) * (2.0*x[i]/dof[i])

        return pdf_jacobian_

    # ===========
    # pdf hessian
    # ===========

    def pdf_hessian(self, x):
        """
        Returns the second derivative of log-prior function for an array of
        hyperparameter input.
        """

        # Convert x or self.dof to arrays of the same size
        x, dof = self._check_param(x)

        pdf_hessian_ = numpy.zeros((x.size, x.size), dtype=float)

        for i in range(x.size):
            ex = 0.5 * (dof[i] + 1.0)
            coeff = gamma(ex) / \
                (numpy.sqrt(dof[i] * numpy.pi) * gamma(0.5*dof[i]))
            k = 1.0 + x[i]**2 / dof[i]
            pdf_hessian_[i, i] = -(2.0 * coeff * ex / dof[i]) * \
                (k**(-ex-1.0) -
                    (ex+1.0) * x[i] * k**(-ex-2.0) * (2.0 * x[i] / dof[i]))

        return pdf_hessian_
