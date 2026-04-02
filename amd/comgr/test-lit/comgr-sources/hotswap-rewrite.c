//===- hotswap-rewrite.c - Test HotSwap rewrite API ----------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "amd_comgr.h"
#include "common.h"

int main(int argc, char *argv[]) {
  if (argc < 2) {
    amd_comgr_status_t Status =
        amd_comgr_hotswap_rewrite(NULL, 0, NULL, NULL, NULL, NULL);
    if (Status != AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT)
      fail("rewrite with NULL args: expected INVALID_ARGUMENT");
    printf("NULL_ARGS: INVALID_ARGUMENT\n");
    return 0;
  }

  if (argc < 4)
    fail("usage: hotswap-rewrite <elf_file> <source_isa> <target_isa> [--zero-size]");

  const char *ElfFile = argv[1];
  const char *SourceISA = argv[2];
  const char *TargetISA = argv[3];
  int ZeroSize = (argc > 4 && strcmp(argv[4], "--zero-size") == 0);

  char *ElfBuf;
  size_t ElfSize = (size_t)setBuf(ElfFile, &ElfBuf);

  void *OutElf = NULL;
  size_t OutSize = 0;

  amd_comgr_status_t Status = amd_comgr_hotswap_rewrite(
      ElfBuf, ZeroSize ? 0 : ElfSize, SourceISA, TargetISA, &OutElf, &OutSize);

  if (Status == AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT) {
    printf("RESULT: INVALID_ARGUMENT\n");
    free(ElfBuf);
    return 0;
  }

  if (Status != AMD_COMGR_STATUS_SUCCESS)
    fail("unexpected error status %d", (int)Status);

  if (OutSize != ElfSize)
    fail("output size %zu != input size %zu", OutSize, ElfSize);
  if (memcmp(OutElf, ElfBuf, ElfSize) != 0)
    fail("output content differs from input");
  free(OutElf);
  free(ElfBuf);

  printf("RESULT: SUCCESS\n");
  return 0;
}
