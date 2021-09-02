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


# =====
# M dot
# =====

def M_dot(cov, Binv, Y, sigma, sigma0, z):
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
    w = cov.solve(sigma, sigma0, z)

    # Computing Mz
    Ytz = numpy.matmul(Y.T, z)
    Binv_Ytz = numpy.matmul(Binv, Ytz)
    Y_Binv_Ytz = numpy.matmul(Y, Binv_Ytz)
    Mz = w - Y_Binv_Ytz

    return Mz
