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

import time


# ==============
# BaseLikelihood
# ==============

class BaseLikelihood(object):
    """
    """

    # ====
    # init
    # ====

    def __init__(self, mean, cov, z):
        """
        """

        # Input attributes
        self.mean = mean
        self.cov = cov
        self.z = z

        # Member data
        self.X = self.mean.X
        self.b = self.mean.b
        self.B = self.mean.B
        self.Binv = self.mean.Binv
        self.mixed_cor = self.cov.mixed_cor

        if self.B is not None:

            # Translate data to the mean of prior of beta.
            self.z = self.z - self.X @ self.b

        # Degrees of freedom of linear model
        if self.B is None:
            m = self.X.shape[1]
            self.dof = m
        else:
            self.dof = 0

        # Residual degrees of freedom
        n = self.X.shape[0]
        self.rdof = n - self.dof

        # Counting elapsed wall time and cpu proc time
        self.wall_time = 0.0
        self.proc_time = 0.0
        self.init_wall_time = 0.0
        self.init_proc_time = 0.0

    # ===
    # tic
    # ===

    def _tic(self):
        """
        Sets the initial wall and proc times.
        """

        self.init_wall_time = time.time()
        self.init_proc_time = time.proc_time()

    # ===
    # toc
    # ===

    def _toc(self):
        """
        Measures the elapsed time from the last tic.
        """

        self.wall_time += time.time() - self.init_wall_time
        self.proc_time += time.process_time() - self.init_proc_time

    # ==========
    # reset time
    # ==========

    def reset_time(self):
        """
        Resets time counters. Used when an instance of this class should be
        reused again.
        """

        self.wall_time = 0.0
        self.proc_time = 0.0
