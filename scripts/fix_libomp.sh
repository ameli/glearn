#!/bin/bash

# SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the license found in the LICENSE.txt file in the root directory
# of this source tree.


# =====
# USAGE
# =====
#
# 1. Use default python on the PATH:
#        sh ./fix_libomp.sh
# 2. Define specific python:
#        sh ./fix_libomp.sh /path_to_python_installation/bin/python
#
# =====================
# Why Using this Script
# =====================
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
#
# ============================
# How This Script Finds Libomp
# ============================
#
# This script uses two methods to find the libomp.dylib for each package:
#
# 1. In the first method, it searches for site-package/.dylibs/libomp.dylib.
#    This file is usually bundled to the package using "delocate" on the
#    package wheel (usually within cibuildwheel). If the package is installed
#    via pip, this lib should exists in site-package/.dylibs. If it does not
#    exists, it uses the second method below. In this case, we call this
#    library as "bundled libomp".
# 2. In the second method, we go through all "*.so" files inside the package.
#    For each "*.so" file, we execute "otool -L file.so", which returns the
#    list of all shared libraries that is linked to this *.so file. Then we
#    grep for libomp.dylib. In the second method, this library is usually
#    installed somewhere on the system (like /usr/local/Cellar/opt/libomp/).
#    The second method usually occurs when the package is compiled locally and
#    not installed via wheel. Hence, it is linked to a libomp that is on the
#    system (not bundled to the package). In this case, we call this linrary
#    as "system library".
#
# ================================================
# How One Libomp is Replaced With The Other Libomp
# ================================================
#
# For both imate and glearn packages, we search of libomp. First, we check for
# bundled lib, if not found, we search for system library.
#
# If both imate and glearn found to have bundled libomp, the we replace the lib
# for glearn with the one for imate, meaning that we keep the imate library,
# and create a symbolic link in glearn to point to the lib in imate.
#
# If either of imate or glearn has system lib, we keep the system lib, and we
# we create a symbolic link from the package that does not have system lib to
# the one that has system lib.
#
# If both packages have system libs, but they are two different openmp libs,
# then it raises an error. If they are the same, there is nothing to do.
#
# ================================================
# Alternative Method To Identify Libomp Dependency
# ================================================
#
# An alternative method to look into which libraries a python package has is
# by "threadpoolctl" package as follow:
#
# python -mpip install threadpoolctl
# python -m pip install imate
# python -m threadpoolctl -i imate


set -e


# ===============
# get package dir
# ===============

get_package_dir()
{(
    set -e

    # Function arguments
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
    echo ${package_dir}
)}


# ==================
# Find Bndled Libomp
# ==================

# This function finds the ./dylib/libomp.dylibs inside the package's .dylibs
# subfolder. This library is usually bundled to the package by applying the
# "delocate" executable on the package's wheel (usually within cibuildhweel)
# after the wheel was built. When you install the package via pip, it is most
# likely that the libomp is bundled to the package inside the .dylibs
# subfolder.

find_bundled_libomp()
{(
    # The output to this function is in the form of stream via echo. The caller
    # function catches the stream from stdout. Note that stderr (for errors)
    # via ">&2" command is not a part of the output.

    # Function arguments
    package_dir=$1

    package_dylibs="${package_dir}/.dylibs"

    if [[ ! -d "${package_dylibs}" ]]
    then
        # Directory does not exists. Use package dir.
        package_dylibs="${package_dir}"
    fi

    # Find libomp within the package
    echo "Searching '${package_dir}/.dylibs/' for 'libomp.dylib' ..." >&2
    package_libomp=`find "${package_dylibs}" -name "*omp.dylib"`

    if [[ $package_libomp == '' ]];
    then
        echo "Did not find 'libomp.dylib' in '$package_dir/.dylib'." >&2
        echo ""
    else
        echo "Found '${package_libomp}'." >&2
        echo "${package_libomp}"
    fi
)}


# ==================
# Find System Libomp
# ==================

# When you compile the package locally, the package uses the libomp that is
# compiled with it, which is usually an openmp that is installed on the system,
# such as the compiler's openmp. In this case, the libomp.dylib is not bundled
# (yet) to the package, rather, is available elsewhere. To find where this lib
# is, we search for all "*.so" files in the package, then apply "otool -L" on
# these files. This tool shows the libraries that each "so" file is linked
# with. We then grep the "omp.dylib" and get the path of libomp.dylib that is
# linked to at least one of the "*.so" files.

find_system_libomp()
{(
    # The output to this function is in the form of stream via echo. The caller
    # function catches the stream from stdout. Note that stderr (for errors)
    # via ">&2" command is not a part of the output.

    set -e

    # Function arguments
    package_dir=$1

    package_libomp=""

    # Iterate over all *.so files in the package directory
    while read -r so_file;
    do
        # Use otool to find all shared libraries linked woth the so file
        echo "Searching for 'libomp.dylib' link in '${so_file}' ... " >&2
        package_libomp=`otool -L ${so_file} | grep -i "omp.dylib" | \
            head -n 1 | cut -d' ' -f1 | tr -d '[:space:]'`

        # If a shared lib named *.omp.dylib foudm exit the search
        if [[ ${package_libomp} != "" ]]; 
        then
            echo "Found '${package_libomp}'." >&2
            break;
        fi
    done < <(find ${package_dir} -name "*.so")

    # Print output
    if [[ ${package_libomp} == '' ]];
    then
        echo "Cannot find 'libomp.dylib' linked in '$package_dir'." >&2
        echo ""
    else
        echo "${package_libomp}"
    fi
)}


# ============
# Replace Lib
# ===========

# Given the paths of two libomp.dylib libraries (on two different locations),
# this function replaces the second lib with the first one. This means the
# first lib remains, and the second one is replaced with the second one.
#
# Alternatively, instead of symbolic link, one can remove the second lib and
# copy the first lib to the second lib. This method has the advantage that if
# the python package corresponding to the first lib is uninstalled, the
# package corresponding to the second lib still can function. To do so, just
# comment the ln command below, and uncomment the rest of this function below.

replace_lib()
{(
    # Function arguments
    libomp_1=$1
    libomp_2=$2

    # Create a symbolic link from the origin (libomp_1) to the destination
    # (libomp_2). This means libomp_1 remains, which libomp_2 becomes a link to
    # the first one.
    # ln -sf ${libomp_1} ${libomp_2}
    # status=$?
    # if [ $status -eq 0 ]; then
    #     echo "Created symbolic link from '${libomp_1}' to '${libomp_2}'." >&2
    # else
    #     echo "Could not create symbolic link from '${libomp_1}' to" \
    #          "'${libomp_2}'. Exiting." >&2
    #     exit 1
    # fi

    # Remove libomp_2
    rm -f ${libomp_2}
    status=$?
    if [ $status -eq 0 ]; then
        echo "Removed '${libomp_2}'."
    else
        echo "ERROR: removing '${libomp_2}' failed."
        exit 1
    fi

    # Copy libomp_1 to libomp_2
    cp -f ${libomp_1} ${libomp_2}
    status=$?
    if [ $status -eq 0 ]; then
        echo "Copied '${libomp_1}' to '${libomp_2}'."
    else
        echo "ERROR: copying '${libomp_1}' to '${libomp_2}' failed."
        exit 1
    fi
)}


# ===========

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
    echo "Using the input python '${PYTHON}'." >&2
else
    # Use default python
    PYTHON=`which python`
    echo "No python path is provided. Using default python at '${PYTHON}'." >&2
fi

# Package names
package_name_1='imate'
package_name_2='glearn'

# Package full path directories in site-package
package_dir_1="$(get_package_dir ${PYTHON} ${package_name_1})"
package_dir_2="$(get_package_dir ${PYTHON} ${package_name_2})"

# Find libomp.dylib in site-packages/package_name/.dylibs/libomp.dylib
bundled_libomp_1="$(find_bundled_libomp ${package_dir_1})"
bundled_libomp_2="$(find_bundled_libomp ${package_dir_2})"

# If the above libomp was not found, alternatively, try to find libomp.dylib on
# system that is linked to site-packages/package_name/*.so
system_libomp_1=""
if [[ ${bundled_libomp_1} == "" ]];
then
    system_libomp_1="$(find_system_libomp ${package_dir_1})"
fi
system_libomp_2=""
if [[ ${bundled_libomp_2} == "" ]];
then
    system_libomp_2="$(find_system_libomp ${package_dir_2})"
fi

# Check any libomp was found for imate
if [[ ${bundled_libomp_1} == "" ]] && [[ ${system_libomp_1} == "" ]];
then
    echo "Count not find any libomp.dylib for 'imate' package. Exiting."
    exit 1;
fi

# Check any libomp was found for glearn
if [[ ${bundled_libomp_2} == "" ]] && [[ ${system_libomp_2} == "" ]];
then
    echo "Count not find any libomp.dylib for 'glearn' package. Exiting."
    exit 1;
fi

# Make both libomp similar by making one to be symbolic link of the other
if [[ ${bundled_libomp_1} != '' ]] && [[ ${bundled_libomp_2} != '' ]];
then

    # If both imate and glearn have their own bundled lib, keep imate lib, but
    # replace glearn's lib and create a symbolic link for glearn to use
    # imate's lib.
    replace_lib ${bundled_libomp_1} ${bundled_libomp_2}

elif [[ ${bundled_libomp_1} != '' ]] && [[ ${system_libomp_2} != '' ]];
then

    # If imate has bundled lib, but glearn is linked to system lib, this means
    # glearn is compiled from source on this machine, but imate was installed
    # via pip. In this case, use glearn's lib.
    replace_lib  ${system_libomp_2} ${bundled_libomp_1}

elif [[ ${system_libomp_1} != '' ]] && [[ ${bundled_libomp_2} != '' ]];
then

    # If glearn has bundled lib, but imate is linked to system lib, this means
    # imate is compiled from source on this machine, but glearn was installed
    # via pip. In this case, use imate's lib.
    replace_lib ${system_libomp_1} ${bundled_libomp_2}

elif [[ ${system_libomp_1} != '' ]] && [[ ${system_libomp_2} != '' ]];
then

    # Both packages use system lib. If both are the same library, do nothing.
    if [[ ${system_libomp_1} != ${system_libomp_2} ]];
    then
        echo "Multiple libomp detected. The 'imate' package uses" \
             "'${system_libomp_1}' while the 'glearn' package uses" \
             "'${system_libomp_2}'. Exiting." >&2
        exit 1
    else
        echo "Both 'imate' and 'glearn' packages share the same library:" \
             "${system_lib_1}. Nothing to do."
    fi
fi
