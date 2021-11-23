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
# this package (`gaussian_proc`) and its dependency package (`imate`) are
# shipped with a version of `libomp` library. Since both of these libraries try
# to be initialized at the runtime, the duplicity leads to a termination error.
#
# To resolve this issue, this script removes the `libomp.dylib` from the
# `imate` package and copies the `libomp.dylib` from `gaussian_proc` back to
# `imate` package.

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
    python=$1
else
    # Use default python
    python=`which python`
    echo "Using default python at '${python}'."
fi


# ===========
# Find libomp
# ===========

find_libomp()
{(
    set -e

    package=$1

    # Get package directory
    package_init=`python -c "import ${package}; print(${package}.__file__)"`

    if [[ $package_init == '' ]];
    then
        echo "Cannot find '${package}' package". >&2
        return 1
    fi

    # Find directory of packages
    package_dir=$(dirname $package_init)
    package_dylibs="${package_dir}/.dylibs"

    if [[ ! -d "${package_dylibs}" ]]
    then
        echo "Directory ${package_dylibs} does not exists."
        return 1
    fi

    # Find libomp within the package
    package_libomp=`find "${package_dylibs}" -name "*omp.dylib"`

    if [[ $package_libomp == '' ]];
    then
        echo "Cannot find '*omp.dylib' in '$package_dir/.dylib'." >&2
        return 1
    else
        echo "${package_libomp}"
    fi
)}

# ===========


# Find libomp in gaussian_proc package
package='gaussian_proc'
gp_libomp="$(find_libomp ${package})"
if [[ ${gp_libomp} == '' ]];
then
  echo "Cannot find libomp in '${package}' package. Nothing to do."
  exit 0
else
    echo "Found '${gp_libomp}'."
fi

# Find libomp in imate package
package='imate'
imate_libomp="$(find_libomp ${package})"
if [[ ${imate_libomp} == '' ]];
then
  echo "No duplicate 'libomp.dylib' was found. Nothing to do."
  exit 0
else
    echo "Found duplicate '${imate_libomp}'"
fi

# Find directory of imate_libomp
imate_libomp_dir=$(dirname $imate_libomp)

# Remove imate_libomp
rm -f ${imate_libomp}
status=$?
if [ $status -eq 0 ]; then
    echo "Removed duplicate '${imate_libomp}'."
else
    echo "ERROR: removing duplicate '${imate_libomp}' failed."
    exit 1
fi

# Copy gp_libomp to imate_libomp_dir
cp -f ${gp_libomp} ${imate_libomp_dir}
status=$?
if [ $status -eq 0 ]; then
    echo "Copied '${gp_libomp}' to '${imate_libomp_dir}'."
else
    echo "ERROR: copying '${gp_libomp}' to '${imate_libomp_dir}' failed."
    exit 1
fi