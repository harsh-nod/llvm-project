//===- comgr-hotswap.cpp - HotSwap ISA stepping rewrite (stub) -----------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "amd_comgr.h"
#include "comgr.h"
#include <cstdlib>
#include <cstring>

amd_comgr_status_t AMD_COMGR_API amd_comgr_hotswap_rewrite(
    const void *elf_data, size_t elf_size,
    const char *source_isa_name, const char *target_isa_name,
    void **out_elf, size_t *out_elf_size) {
  if (!elf_data || elf_size == 0 || !source_isa_name || !target_isa_name ||
      !out_elf || !out_elf_size)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  // Validate and parse both ISA names.
  COMGR::TargetIdentifier SourceIdent, TargetIdent;
  if (COMGR::parseTargetIdentifier(source_isa_name, SourceIdent) ||
      COMGR::parseTargetIdentifier(target_isa_name, TargetIdent))
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  // Currently only GFX1250 B0-to-A0 is supported.
  if (SourceIdent.Processor != "gfx1250" || TargetIdent.Processor != "gfx1250")
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
