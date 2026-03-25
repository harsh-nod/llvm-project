//===- comgr-hotswap.cpp - HotSwap ISA rewriting --------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Stub implementations of the HotSwap ISA rewriting APIs. These will be
/// replaced with full LLVM MC-backed implementations in a subsequent step.
///
//===----------------------------------------------------------------------===//

#include "amd_comgr/amd_comgr.h"
#include <cstdlib>
#include <cstring>

amd_comgr_status_t AMD_COMGR_API
amd_comgr_hotswap_rewrite(
    const void *elf_data,
    size_t elf_size,
    const char *source_isa,
    const char *target_isa,
    uint32_t flags,
    const char *rules_json,
    void **out_elf,
    size_t *out_elf_size,
    amd_comgr_hotswap_result_t *result) {
  if (!elf_data || elf_size == 0 || !out_elf || !out_elf_size || !result)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  // Stub: return a copy of the input unchanged
  void *copy = std::malloc(elf_size);
  if (!copy)
    return AMD_COMGR_STATUS_ERROR;
  std::memcpy(copy, elf_data, elf_size);

  *out_elf = copy;
  *out_elf_size = elf_size;
  std::memset(result, 0, sizeof(*result));

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_COMGR_API
amd_comgr_hotswap_needs_transpile(
    const char *source_isa,
    const char *target_isa,
    bool *needs_transpile) {
  if (!source_isa || !target_isa || !needs_transpile)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  // Stub: always return false
  *needs_transpile = false;
  return AMD_COMGR_STATUS_SUCCESS;
}
