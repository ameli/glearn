#! /usr/bin/env python

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

import sys
import glearn
from glearn import Memory
from glearn import Timer


# ===========
# test device
# ===========

def test_device():
    """
    A test for glearn.device.
    """

    # Device inquiry
    glearn.info()
    glearn.device.locate_cuda()
    glearn.device.get_nvidia_driver_version()
    glearn.device.get_processor_name()
    glearn.device.get_gpu_name()
    glearn.device.get_num_cpu_threads()
    glearn.device.get_num_gpu_devices()
    glearn.device.restrict_to_single_processor()

    # Memory
    mem = Memory()
    mem.start()
    mem.stop()
    mem.get_mem(human_readable=True)
    Memory.get_resident_memory()
    Memory.get_resident_memory(human_readable=True)

    # Timer
    timer = Timer(hold=True)
    timer.tic()
    timer.toc()
    timer.wall_time
    timer.proc_time


# ===========
# System Main
# ===========

if __name__ == "__main__":
    sys.exit(test_device())
