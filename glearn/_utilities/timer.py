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

    A counter counts a completed tic-toc call. If there is one tic but multiple
    toc calls later, this is counted as only once.

    When ``hold`` is True, the times between successive tic-toc calls are
    commulative.
    """

    # ====
    # init
    # ====

    def __init__(self, hold=True):
        """
        Initialization.
        """

        # Internal variable used to store initial timestamps
        self.init_wall_time = 0.0
        self.init_proc_time = 0.0
        self.tic_initiated = False
        self.hold = hold

        # Public attributes
        self.wall_time = 0.0
        self.proc_time = 0.0
        self.count = 0

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

        # This variable is used to count a complete tic-toc call.
        self.tic_initiated = True

    # ===
    # toc
    # ===

    def toc(self):
        """
        Measures the elapsed time from the last tic.
        """

        wall_time_ = time.time() - self.init_wall_time

        if python2:
            proc_time_ = time.time() - self.init_proc_time
        else:
            proc_time_ = time.process_time() - self.init_proc_time

        if self.hold:
            # Commulative time between successive tic-toc
            self.wall_time += wall_time_
            self.proc_time += proc_time_
        else:
            # Only measures the elapsed time for the current tic-toc
            self.wall_time = wall_time_
            self.proc_time = proc_time_

        # Prevents counting multiple toc calls which were initiated with one
        # tic call.
        if self.tic_initiated:
            self.tic_initiated = False
            self.count += 1

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
        self.tic_initiated = False
        self.count = 0
