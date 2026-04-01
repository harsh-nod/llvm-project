//===- hotswap-rewrite.c - Test HotSwap B0-to-A0 API ---------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "amd_comgr.h"
#include "common.h"

int main(int argc, char *argv[]) {
  amd_comgr_status_t Status;

  // Test 1: NULL arguments return INVALID_ARGUMENT
  Status = amd_comgr_hotswap_rewrite_b0a0(NULL, 0, NULL, NULL);
  if (Status != AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT)
    fail("rewrite_b0a0 with NULL args: expected INVALID_ARGUMENT");

  // Test 2: Zero-size input returns INVALID_ARGUMENT
  const unsigned char TestElf[] = {
      0x7f, 'E', 'L', 'F', 0x02, 0x01, 0x01, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
  void *OutElf = NULL;
  size_t OutSize = 0;
  Status = amd_comgr_hotswap_rewrite_b0a0(TestElf, 0, &OutElf, &OutSize);
  if (Status != AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT)
    fail("rewrite_b0a0 with zero size: expected INVALID_ARGUMENT");

  // Test 3: Valid call returns input unchanged (stub behavior)
  Status = amd_comgr_hotswap_rewrite_b0a0(TestElf, sizeof(TestElf),
                                           &OutElf, &OutSize);
  if (Status != AMD_COMGR_STATUS_SUCCESS)
    fail("rewrite_b0a0 passthrough failed");

  if (OutSize != sizeof(TestElf))
    fail("rewrite_b0a0: output size %zu != input size %zu",
         OutSize, sizeof(TestElf));
  if (memcmp(OutElf, TestElf, sizeof(TestElf)) != 0)
    fail("rewrite_b0a0: output content differs from input");
  free(OutElf);

  printf("PASSED\n");
  return 0;
}
