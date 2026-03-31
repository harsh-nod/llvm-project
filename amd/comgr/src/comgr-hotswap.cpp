//===- comgr-hotswap.cpp - HotSwap B0-to-A0 ISA rewriting (stub) ---------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "amd_comgr.h"
#include <cstdlib>
#include <cstring>

amd_comgr_status_t AMD_COMGR_API amd_comgr_hotswap_rewrite_b0a0(
    const void *elf_data, size_t elf_size,
    void **out_elf, size_t *out_elf_size) {
  if (!elf_data || elf_size == 0 || !out_elf || !out_elf_size)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  // Stub: return a copy of the input unchanged.
  // Full B0-to-A0 patching implementation follows in subsequent commits.
  void *copy = std::malloc(elf_size);
  if (!copy)
    return AMD_COMGR_STATUS_ERROR;
  std::memcpy(copy, elf_data, elf_size);
  *out_elf = copy;
  *out_elf_size = elf_size;
  return AMD_COMGR_STATUS_SUCCESS;
}
