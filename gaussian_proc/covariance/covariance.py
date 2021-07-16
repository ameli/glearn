# SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the license found in the LICENSE.txt file in the root
# directory of this source tree.


# =======
# Imports
# =======

from ..correlation import Correlation
import numpy
from scipy.sparse import isspmatrix


# ==========
# Covariance
# ==========

class Covariance(object):
    """
    """

    # ====
    # init
    # ====

    def __init__(self, K, sigma=None, sigma0=None):
        """
        """

        self._check_arguments(K, sigma, sigma0)

        # Set attributes
        self.K = K
        self.sigma = sigma
        self.sigma0 = sigma0

    # ===============
    # Check arguments
    # ===============

    def _check_arguments(self, K, sigma, sigma0):
        """
        """

        # Check K
        if K is None:
            raise ValueError('"K" cannot be None.')

        elif not isinstance(K, numpy.ndarray) and \
             not isspmatrix(K) and \
             not isinstance(K, Correlation):
            raise TypeError('"K" should be either a "numpy.ndarray" matrix ' +
                            'or an instance of "Correlation" class.')
        
        if isinstance(K, numpy.ndarray):
            if K.ndim != 2:
                raise ValueError('"K" should be a 2D matrix.')

            elif K.shape[0] != K.shape[1]:
                raise ValueError('"K" should be a square matrix.')

            not_correlation = False
            for i in range(K.shape[0]):
                if K[i, i] != 1.0:
                    not_correlation = True
                    break

            if not_correlation:
                raise ValueError('Diagonal elements of "K" should be "1".')

        # Check sigma
        if sigma is not None:
            if not isinstance(sigma, int) and isinstance(sigma, float):
                raise TypeError('"sigma" should be a float type.')
            elif sigma < 0.0:
                raise ValueError('"sigma" cannot be negative.')

        # Check sigma0
        if sigma0 is not None:
            if not isinstance(sigma0, int) and isinstance(sigma0, float):
                raise TypeError('"sigma0" should be a float type.')
            elif sigma0 < 0.0:
                raise ValueError('"sigma0" cannot be negative.')
