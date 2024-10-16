/*
 *  SPDX-FileCopyrightText: Copyright 2022, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


// =======
// Headers
// =======

#include "./c_get_config.h"


// =============
// is use openmp
// =============

/// \brief Returns USE_OPENMP.
///

bool is_use_openmp()
{
    #if defined(USE_OPENMP) && (USE_OPENMP == 1)
        return true;
    #else
        return false;
    #endif
}


// ===========
// is use cuda
// ===========

/// \brief Returns USE_CUDA.
///

bool is_use_cuda()
{
    #if defined(USE_CUDA) && (USE_CUDA == 1)
        return true;
    #else
        return false;
    #endif
}


// =======================
// is cuda dynamic loading
// =======================

/// \brief Returns CUDA_DYNAMIC_LOADING.
///

bool is_cuda_dynamic_loading()
{
    #if defined(CUDA_DYNAMIC_LOADING) && (CUDA_DYNAMIC_LOADING == 1)
        return true;
    #else
        return false;
    #endif
}


// =============
// is debug mode
// =============

/// \brief Returns DEBUG_MODE.
///

bool is_debug_mode()
{
    #if defined(DEBUG_MODE) && (DEBUG_MODE == 1)
        return true;
    #else
        return false;
    #endif
}


// =========================
// is cython build in source
// =========================

/// \brief Returns CYTHON_BUILD_IN_SOURCE.
///

bool is_cython_build_in_source()
{
    #if defined(CYTHON_BUILD_IN_SOURCE) && (CYTHON_BUILD_IN_SOURCE == 1)
        return true;
    #else
        return false;
    #endif
}


// =======================
// is cython build for doc
// =======================

/// \brief Returns CYTHON_BUILD_FOR_DOC.
///

bool is_cython_build_for_doc()
{
    #if defined(CYTHON_BUILD_FOR_DOC) && (CYTHON_BUILD_FOR_DOC == 1)
        return true;
    #else
        return false;
    #endif
}
