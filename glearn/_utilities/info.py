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

def info(print_only=True):
    """
    Prints info about device, package version and memory usage.
    """

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

    info_ = {
        'glearn_version': __version__,
        'imate_version': imate.__version__,
        'processor_name': get_processor_name(),
        'num_cpu_threads': get_num_cpu_threads(),
        'gpu_name': get_gpu_name(),
        'num_gpu_devices': get_num_gpu_devices(),
        'mem_used': mem_used,
        'mem_unit': mem_unit,
        'cuda_version': cuda_version_
    }

    # Print
    if print_only:
        print('')
        print('glearn version  : %s' % info_['glearn_version'])
        print('imate version   : %s' % info_['imate_version'])
        print('processor       : %s' % info_['processor_name'])
        print('num threads     : %d' % info_['num_cpu_threads'])
        print('gpu device      : %s' % info_['gpu_name'])
        print('num gpu devices : %d' % info_['num_gpu_devices'])
        print('cuda version    : %s' % info_['cuda_version'])
        print('process memory  : %0.1f (%s)'
              % (info_['mem_used'], info_['mem_unit']))
        print('')
    else:
        return info_
