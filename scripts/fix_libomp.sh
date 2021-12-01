# SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the license found in the LICENSE.txt file in the root directory
# of this source tree.


# USAGE:
#
# 1. Use default python on the PATH:
#        sh ./fix_libomp.sh
# 2. Define specific python:
#        sh ./fix_libomp.sh /path_to_python_installation/bin/python
#
# This script fixes the following error:
#
#    OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already
#    initialized.
#    OMP: Hint This means that multiple copies of the OpenMP runtime have been
#    linked into the program. That is dangerous, since it can degrade
#    performance or cause incorrect results. The best thing to do is to ensure
#    that only a single OpenMP runtime is linked into the process, e.g. by
#    avoiding static linking of the OpenMP runtime in any library. As an
#    unsafe, unsupported, undocumented workaround you can set the environment
#    variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to
#    execute, but that may cause crashes or silently produce incorrect results.
#    For more information, please see http://openmp.llvm.org/
#    Abort trap: 6
#
# The above error happens only on MacOS. The error is originated because both
# this package (`glearn`) and its dependency package (`imate`) are shipped with
# a version of `libomp` library. Since both of these libraries try to be
# initialized at the runtime, the duplicity leads to a termination error.
#
# To resolve this issue, this script removes the `libomp.dylib` from the
# `imate` package. Also, if `glearn` package comes with `libomp.dylib`, it
# copies it from `glearn` package to `imate` package.

set -e

# Check if the operating system is MacOS.
if [[ $OSTYPE != 'darwin'* ]]; then
  echo 'This script should run only on MacOS.'
  exit
fi

# Get the directory of python
if [[ $1 != '' ]];
then
    # Use user-defined python
    PYTHON=$1
else
    # Use default python
    PYTHON=`which python`
    echo "Using default python at '${PYTHON}'."
fi


# ===========
# Find libomp
# ===========

find_libomp()
{(
    # The output to this function is in the form of stream via echo. The caller
    # function catches the stream from stdout. Note that stderr (for errors)
    # via ">&2" command is not a part of the output.

    set -e

    PYTHON=$1
    package=$2

    # Get package directory
    package_init=`${PYTHON} -c "import ${package}; print(${package}.__file__)"`

    if [[ $package_init == '' ]];
    then
        echo "Cannot find '${package}' package". >&2  # sending error to stderr
        echo ""  # outputing an empty string to stdout in the case of error
    fi

    # Find directory of packages
    package_dir=$(dirname $package_init)
    package_dylibs="${package_dir}/.dylibs"

    if [[ ! -d "${package_dylibs}" ]]
    then
        # Directory does not exists. Use package dir.
        package_dylibs="${package_dir}"
    fi

    # Find libomp within the package
    package_libomp=`find "${package_dylibs}" -name "*omp.dylib"`

    if [[ $package_libomp == '' ]];
    then
        echo "Cannot find '*omp.dylib' in '$package_dir/.dylib'." >&2
        echo ""
    else
        echo "${package_libomp}"
    fi
)}

# ===========

# Find libomp in imate package
package1='imate'
imate_libomp="$(find_libomp ${PYTHON} ${package1})"

# Find libomp in glearn package
package2='glearn'
gp_libomp="$(find_libomp ${PYTHON} ${package2})"

# Remove libomp from imate
if [[ ${imate_libomp} != '' ]];
then
    # Find directory of imate_libomp
    imate_libomp_dir=$(dirname $imate_libomp)

    # Remove libomp of the imate package
    rm -f ${imate_libomp}
    status=$?
    if [ $status -eq 0 ]; then
        echo "Removed '${imate_libomp}'."
    else
        echo "ERROR: removing '${imate_libomp}' failed."
        exit 1
    fi
fi

# Copy libomp from glearn to imate
if [[ ${gp_libomp} != '' ]] && [[ ${imate_libomp} != '' ]];
then
    # Copy libomp of glearn package into imate package
    cp -f ${gp_libomp} ${imate_libomp_dir}
    status=$?
    if [ $status -eq 0 ]; then
        echo "Copied '${gp_libomp}' to '${imate_libomp_dir}'."
    else
        echo "ERROR: copying '${gp_libomp}' to '${imate_libomp_dir}' failed."
        exit 1
    fi
fi
