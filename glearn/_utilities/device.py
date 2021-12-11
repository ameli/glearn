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

import os
import re
import sys
import platform
import subprocess

# resource is not available in windows
if os.name == 'posix':
    import resource


# ==================
# get processor name
# ==================

def get_processor_name():
    """
    Gets the name of CPU.

    For windows operating system, this function still does not get the full
    brand name of the cpu.
    """

    if platform.system() == "Windows":
        return platform.processor()

    elif platform.system() == "Darwin":
        os.environ['PATH'] = os.environ['PATH'] + os.pathsep + '/usr/sbin'
        command = "sysctl -n machdep.cpu.brand_string"
        return subprocess.getoutput(command).strip()

    elif platform.system() == "Linux":
        command = "cat /proc/cpuinfo"
        all_info = subprocess.getoutput(command).strip()
        for line in all_info.split("\n"):
            if "model name" in line:
                return re.sub(".*model name.*:", "", line, 1)[1:]

    return ""


# ===================
# get num cpu threads
# ===================

def get_num_cpu_threads():
    """
    Returns the number of cpu threads.
    """

    num_cpu_threads = len(os.sched_getaffinity(0))
    return num_cpu_threads


# ============
# get gpu name
# ============

def get_gpu_name():
    """
    Gets the name of gpu device.
    """

    command = ['nvidia-smi', '-a', '|', 'grep', '-i', '"Product Name"', '-m',
               '1', '|', 'grep', '-o', '":.*"', '|', 'cut', '-c', '3-']

    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        stdout, _ = process.communicate()
        error_code = process.poll()

        # Error code 127 means nvidia-smi is not a recognized command. Error
        # code 9 means nvidia-smi could not find any device.
        if error_code != 0:
            gpu_name = 'none'
        else:
            gpu_name = stdout.strip()

    except FileNotFoundError:
        gpu_name = 'none'

    return gpu_name


# ===================
# get num gpu devices
# ===================

def get_num_gpu_devices():
    """
    Get number of all gpu devices
    """

    command = ['nvidia-smi', '--list-gpus', '|', 'wc', '-l']

    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        stdout, _ = process.communicate()
        error_code = process.poll()

        # Error code 127 means nvidia-smi is not a recognized command. Error
        # code 9 means nvidia-smi could not find any device.
        if error_code != 0:
            num_gpu_devices = 0
        else:
            num_gpu_devices = int(stdout)

    except FileNotFoundError:
        num_gpu_devices = 0

    return num_gpu_devices


# ================
# get memory usage
# ================

def get_memory_usage(human_readable=False):
    """
    Returns the memory usage of the current python process in bytes.
    """

    # Convert Kb to bytes
    k = 2**10

    if os.name == 'posix':
        # In Linux and MaxOS
        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        # In Linux, the output of the command is in Kb. Convert to Bytes.
        if sys.platform == 'linux':
            mem *= k

    else:
        # In windows
        pid = os.getpid()
        command = ['tasklist', '/fi', '"pid eq %d"' % pid]

        try:
            pid = os.getpid()
            command = ['tasklist', '/fi', 'pid eq %d' % pid]
            process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            error_code = process.poll()
            if error_code != 0:
                mem = 'n/a'
                return mem

            # Parse output
            last_line = stdout.strip().decode().split("\n")[-1]

            # Check last line of output has any number in it
            is_digit = [char.isdigit() for char in last_line]
            if not any(is_digit):
                mem = 'n/a'
                return mem

            # Get memory as string and its unit
            mem_string = last_line.split(' ')[-2].replace(',', '')
            mem = int(mem_string)
            mem_unit = last_line.split(' ')[-1]

            # Convert bytes based on the unit
            if mem_unit == 'K':
                exponent = 1
            if mem_unit == 'M':
                exponent = 2
            if mem_unit == 'G':
                exponent = 3
            if mem_unit == 'T':
                exponent = 4

            # Memory in bytes
            mem = mem * (k**exponent)

        except FileNotFoundError:
            mem = 'n/a'

    # Default unit is bytes.
    unit = 'b'

    # Convert from bytes to the closets unit
    if human_readable:
        mem, unit = _human_readable_memory(mem)

    return mem, unit


# =====================
# human readable memory
# =====================

def _human_readable_memory(mem_bytes):
    """
    Converts memory in Kilo-Bytes to human readable unit.
    """

    k = 2**10
    counter = 0
    hr_bytes = mem_bytes

    while hr_bytes > k:
        hr_bytes /= k
        counter += 1

    if counter == 0:
        unit = ' b'
    elif counter == 1:
        unit = 'Kb'
    elif counter == 2:
        unit = 'Mb'
    elif counter == 3:
        unit = 'Gb'
    elif counter == 4:
        unit = 'Tb'
    elif counter == 5:
        unit = 'Pb'

    return hr_bytes, unit


# ============================
# restrict to single processor
# ============================

def restrict_to_single_processor():
    """
    To measure the CPU time of all processors we use time.process_time() which
    takes into account of elapsed time of all running threads. However, it
    seems when I use scipy.optimize.differential_evolution method with either
    worker=-1 or worker=1, the CPU time is not measured properly.

    After all failed trials, the only solution that measures time (for only
    scipy.optimize.differential_evolution) is to restrict the whole python
    script to use a single code. This function does that.

    Note, other scipy.optimize methods (like shgo) do not have this issue. That
    means, you can still run the code in parallel and the time.process_time()
    measures the CPU time of all cores properly.
    """

    # Uncomment lines below if measuring elapsed time. These will restrict
    # python to only use one processing thread.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
