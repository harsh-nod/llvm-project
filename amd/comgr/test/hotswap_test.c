//===- hotswap_test.c -----------------------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "amd_comgr.h"
#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
  amd_comgr_status_t Status;
  amd_comgr_hotswap_result_t Result;
  void *OutElf = NULL;
  size_t OutSize = 0;

  // Test 1: NULL elf_data returns INVALID_ARGUMENT
  Status = amd_comgr_hotswap_rewrite(NULL, 0, NULL, NULL, 0, NULL,
                                     &OutElf, &OutSize, &Result);
  checkStatus(Status, AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT,
              "hotswap_rewrite with NULL elf_data");

  // Test 2: NULL out_elf returns INVALID_ARGUMENT
  const unsigned char TestElf[] = {
      0x7f, 'E', 'L', 'F', 0x02, 0x01, 0x01, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
  Status = amd_comgr_hotswap_rewrite(TestElf, sizeof(TestElf), NULL, NULL,
                                     0, NULL, NULL, &OutSize, &Result);
  checkStatus(Status, AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT,
              "hotswap_rewrite with NULL out_elf");

  // Test 3: NULL result returns INVALID_ARGUMENT
  Status = amd_comgr_hotswap_rewrite(TestElf, sizeof(TestElf), NULL, NULL,
                                     0, NULL, &OutElf, &OutSize, NULL);
  checkStatus(Status, AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT,
              "hotswap_rewrite with NULL result");

  // Test 4: Zero-size input returns INVALID_ARGUMENT
  Status = amd_comgr_hotswap_rewrite(TestElf, 0, NULL, NULL, 0, NULL,
                                     &OutElf, &OutSize, &Result);
  checkStatus(Status, AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT,
              "hotswap_rewrite with zero size");

  // Test 5: No-op flags returns a copy of the input
  memset(&Result, 0, sizeof(Result));
  Status = amd_comgr_hotswap_rewrite(TestElf, sizeof(TestElf), NULL, NULL,
                                     AMD_COMGR_HOTSWAP_FLAG_NONE, NULL,
                                     &OutElf, &OutSize, &Result);
  checkError(Status, "hotswap_rewrite no-op passthrough");

  if (OutSize != sizeof(TestElf))
    fail("hotswap_rewrite no-op: output size %zu != input size %zu",
         OutSize, sizeof(TestElf));
  if (memcmp(OutElf, TestElf, sizeof(TestElf)) != 0)
    fail("hotswap_rewrite no-op: output content differs from input");
  free(OutElf);
  OutElf = NULL;

  // Test 6: NULL args for needs_transpile returns INVALID_ARGUMENT
  Status = amd_comgr_hotswap_needs_transpile(NULL, NULL, NULL);
  checkStatus(Status, AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT,
              "hotswap_needs_transpile with NULL args");

  // Test 7: Same-family ISAs do not need transpile
  bool NeedsTranspile = true;
  Status = amd_comgr_hotswap_needs_transpile(
      "amdgcn-amd-amdhsa--gfx1250", "amdgcn-amd-amdhsa--gfx1250",
      &NeedsTranspile);
  checkError(Status, "hotswap_needs_transpile same ISA");
  if (NeedsTranspile)
    fail("hotswap_needs_transpile: same ISA should not need transpile");

  printf("All hotswap API tests passed.\n");
}
