//===- hotswap_test.c - Test HotSwap B0-to-A0 API ------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "amd_comgr.h"
#include "common.h"

int main(int argc, char *argv[]) {
  amd_comgr_status_t Status;

  // Test 1: NULL arguments return INVALID_ARGUMENT
  Status = amd_comgr_hotswap_rewrite_b0a0(NULL, 0, NULL, NULL);
  checkStatus(Status, AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT,
              "rewrite_b0a0 with NULL args");

  // Test 2: Valid call returns input unchanged (stub behavior)
  const unsigned char TestElf[] = {
      0x7f, 'E', 'L', 'F', 0x02, 0x01, 0x01, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
  void *OutElf = NULL;
  size_t OutSize = 0;
  Status = amd_comgr_hotswap_rewrite_b0a0(TestElf, sizeof(TestElf),
                                            &OutElf, &OutSize);
  checkError(Status, "rewrite_b0a0 passthrough");

  if (OutSize != sizeof(TestElf))
    fail("rewrite_b0a0: output size %zu != input size %zu",
         OutSize, sizeof(TestElf));
  if (memcmp(OutElf, TestElf, sizeof(TestElf)) != 0)
    fail("rewrite_b0a0: output content differs from input");
  free(OutElf);

  // Test 3: Zero-size input returns INVALID_ARGUMENT
  Status = amd_comgr_hotswap_rewrite_b0a0(TestElf, 0, &OutElf, &OutSize);
  checkStatus(Status, AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT,
              "rewrite_b0a0 with zero size");

  printf("All hotswap B0-to-A0 API tests passed.\n");
  return 0;
}
