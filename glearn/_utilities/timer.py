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

import time

# Check python version
import sys
if sys.version_info[0] == 2:
    python2 = True
else:
    python2 = False


# =====
# Timer
# =====

class Timer(object):
    """
    A timer to record elapsed wall time and CPU process time.
    """

    # ====
    # init
    # ====

    def __init__(self):
        """
        Initialization.
        """

        # Internal variable used to store inital timestamps
        self.init_wall_time = 0.0
        self.init_proc_time = 0.0

        # Public attributes
        self.wall_time = 0.0
        self.proc_time = 0.0

    # ===
    # tic
    # ===

    def tic(self):
        """
        Sets the initial wall and proc times.
        """

        self.init_wall_time = time.time()

        if python2:
            self.init_proc_time = time.time()
        else:
            self.init_proc_time = time.process_time()

    # ===
    # toc
    # ===

    def toc(self):
        """
        Measures the elapsed time from the last tic.
        """

        self.wall_time += time.time() - self.init_wall_time

        if python2:
            self.proc_time += time.time() - self.init_proc_time
        else:
            self.proc_time += time.process_time() - self.init_proc_time

    # =====
    # reset
    # =====

    def reset(self):
        """
        Resets time counters. Used when an instance of this class should be
        reused again.
        """

        self.wall_time = 0.0
        self.proc_time = 0.0
