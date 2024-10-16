/*
 *  SPDX-FileCopyrightText: Copyright 2022, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef _DEFINITIONS_C_GET_CONFIG_H_
#define _DEFINITIONS_C_GET_CONFIG_H_


// ============
// Declarations
// ============

bool is_use_openmp();
bool is_use_cuda();
bool is_cuda_dynamic_loading();
bool is_debug_mode();
bool is_cython_build_in_source();
bool is_cython_build_for_doc();


#endif  // _DEFINITIONS_C_GET_CONFIG_H_
