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
import subprocess
import sys

# resource is not available in windows
if os.name == 'posix':
    import resource

__all__ = ['get_memory_usage']


# ================
# get memory usage
# ================

def get_memory_usage(human_readable=False):
    """
    Returns the resident memory (or RSS) for the current python process. RSS
    means the memory that only resides on the RAM. If the current process
    overflows some of its memory onto the hard disk swap space, only the
    memory residing on RAM will be reflected in RSS.

    If ``human_readable`` is ``False``, the output is in Bytes. If
    ``human_readable`` is ``True``, the output is converted to a human readable
    unit.
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
        mem, unit = human_readable_memory(mem)

    return mem, unit


# =====================
# human readable memory
# =====================

def human_readable_memory(mem_bytes):
    """
    Converts memory in Bytes to human readable unit.
    """

    k = 2**10
    counter = 0
    mem_hr = mem_bytes

    while mem_hr > k:
        mem_hr /= k
        counter += 1

    if counter == 0:
        unit = ' b'      # Byte
    elif counter == 1:
        unit = 'Kb'      # Kilo byte
    elif counter == 2:
        unit = 'Mb'      # Mega byte
    elif counter == 3:
        unit = 'Gb'      # Giga byte
    elif counter == 4:
        unit = 'Tb'      # Tera byte
    elif counter == 5:
        unit = 'Pb'      # Peta byte
    elif counter == 6:
        unit = 'Eb'      # Exa byte
    elif counter == 7:
        unit = 'Zb'      # Zetta byte
    elif counter == 8:
        unit = 'Yb'      # Yotta byte

    return mem_hr, unit
