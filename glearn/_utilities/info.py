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

from .device import get_processor_name, get_num_cpu_threads, get_gpu_name, \
        get_num_gpu_devices
from .memory import Memory
from .cuda import locate_cuda, get_cuda_version
from ..__version__ import __version__
import imate

__all__ = ['info']


# ====
# info
# ====

def info():
    """
    Prints info about device, package version and memory usage.
    """

    glearn_version = __version__
    imate_version = imate.__version__
    processor_name = get_processor_name()
    num_cpu_threads = get_num_cpu_threads()
    gpu_name = get_gpu_name()
    num_gpu_devices = get_num_gpu_devices()
    mem_used, mem_unit = Memory.get_memory_usage(human_readable=True)

    # Get cuda version
    cuda = locate_cuda()
    if cuda != {}:
        cuda_version = get_cuda_version(cuda['home'])
        cuda_version_ = '%d.%d.%d' \
            % (cuda_version['major'], cuda_version['minor'],
               cuda_version['patch'])
    else:
        cuda_version_ = 'not found'

    # Print
    print('')
    print('glearn version  : %s' % glearn_version)
    print('imate version   : %s' % imate_version)
    print('processor       : %s' % processor_name)
    print('num threads     : %d' % num_cpu_threads)
    print('gpu device      : %s' % gpu_name)
    print('num gpu devices : %d' % num_gpu_devices)
    print('cuda version    : %s' % cuda_version_)
    print('process memory  : %0.1f (%s)' % (mem_used, mem_unit))
    print('')
