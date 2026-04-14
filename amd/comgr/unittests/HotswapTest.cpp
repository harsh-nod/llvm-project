//===- HotswapTest.cpp - Unit tests for HotSwap internals -----------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-internal.h"
#include "gtest/gtest.h"
#include <cstring>

// ═══════════════════════════════════════════════════════════════════════════════
// EncodeSBranch
// ═══════════════════════════════════════════════════════════════════════════════

TEST(EncodeSBranch, ForwardBranchGFX9) {
  uint8_t out[4] = {};
  // Branch from offset 0 to offset 8: byte_delta = 8 - 0 - 4 = 4,
  // dword_offset = 1.
  ASSERT_TRUE(EncodeSBranch(0, 8, out, /*gfx12=*/false));
  uint32_t encoded;
  std::memcpy(&encoded, out, 4);
  // GFX9 opcode = 0xBF82'0000, offset field = 1 → 0xBF82'0001.
  EXPECT_EQ(encoded, 0xBF820001u);
}

TEST(EncodeSBranch, BackwardBranchGFX9) {
  uint8_t out[4] = {};
  // Branch from offset 16 to offset 0: byte_delta = 0 - 16 - 4 = -20,
  // dword_offset = -5.
  ASSERT_TRUE(EncodeSBranch(16, 0, out, /*gfx12=*/false));
  uint32_t encoded;
  std::memcpy(&encoded, out, 4);
  // -5 as uint16_t = 0xFFFB, combined with GFX9 opcode → 0xBF82'FFFB.
  EXPECT_EQ(encoded, 0xBF82FFFBu);
}

TEST(EncodeSBranch, ForwardBranchGFX12) {
  uint8_t out[4] = {};
  ASSERT_TRUE(EncodeSBranch(0, 8, out, /*gfx12=*/true));
  uint32_t encoded;
  std::memcpy(&encoded, out, 4);
  // GFX12 opcode = 0xBFA0'0000, offset field = 1 → 0xBFA0'0001.
  EXPECT_EQ(encoded, 0xBFA00001u);
}

TEST(EncodeSBranch, UnalignedDeltaFails) {
  uint8_t out[4] = {};
  // byte_delta = 7 - 0 - 4 = 3, not divisible by 4.
  EXPECT_FALSE(EncodeSBranch(0, 7, out));
}

TEST(EncodeSBranch, OutOfRangeFails) {
  uint8_t out[4] = {};
  // dword_offset would be (500000 - 0 - 4) / 4 = 124999, > 32767.
  EXPECT_FALSE(EncodeSBranch(0, 500000, out));
}

TEST(EncodeSBranch, ZeroOffsetBranch) {
  uint8_t out[4] = {};
  // Branch from offset 0 to offset 4: byte_delta = 0, dword_offset = 0.
  ASSERT_TRUE(EncodeSBranch(0, 4, out, /*gfx12=*/false));
  uint32_t encoded;
  std::memcpy(&encoded, out, 4);
  EXPECT_EQ(encoded, S_BRANCH_GFX9);
}

// ═══════════════════════════════════════════════════════════════════════════════
// EncodeSNop
// ═══════════════════════════════════════════════════════════════════════════════

TEST(EncodeSNop, ProducesCorrectEncoding) {
  uint8_t out[4] = {};
  EncodeSNop(out);
  uint32_t encoded;
  std::memcpy(&encoded, out, 4);
  EXPECT_EQ(encoded, S_NOP_OPCODE);
}

// ═══════════════════════════════════════════════════════════════════════════════
// ExtractCPU
// ═══════════════════════════════════════════════════════════════════════════════

TEST(ExtractCPU, FullISAName) {
  EXPECT_EQ(ExtractCPU("amdgcn-amd-amdhsa--gfx1250"), "gfx1250");
}

TEST(ExtractCPU, GFX900) {
  EXPECT_EQ(ExtractCPU("amdgcn-amd-amdhsa--gfx900"), "gfx900");
}

TEST(ExtractCPU, BareGFX) {
  EXPECT_EQ(ExtractCPU("gfx942"), "gfx942");
}

TEST(ExtractCPU, NoGFXPrefix) {
  EXPECT_EQ(ExtractCPU("amdgcn-amd-amdhsa"), "");
}

TEST(ExtractCPU, EmptyString) {
  EXPECT_EQ(ExtractCPU(""), "");
}

TEST(ExtractCPU, StopsAtNonAlphanumeric) {
  EXPECT_EQ(ExtractCPU("amdgcn-amd-amdhsa--gfx1250:sramecc+"),
            "gfx1250");
}

// ═══════════════════════════════════════════════════════════════════════════════
// MallocBuffer
// ═══════════════════════════════════════════════════════════════════════════════

TEST(MallocBuffer, DefaultIsEmpty) {
  MallocBuffer buf;
  EXPECT_FALSE(static_cast<bool>(buf));
  EXPECT_EQ(buf.data, nullptr);
  EXPECT_EQ(buf.size, 0u);
}

TEST(MallocBuffer, AllocNonZero) {
  MallocBuffer buf(64);
  EXPECT_TRUE(static_cast<bool>(buf));
  EXPECT_NE(buf.data, nullptr);
  EXPECT_EQ(buf.size, 64u);
}

TEST(MallocBuffer, MoveConstructor) {
  MallocBuffer a(128);
  uint8_t *orig_data = a.data;
  MallocBuffer b(std::move(a));
  EXPECT_EQ(b.data, orig_data);
  EXPECT_EQ(b.size, 128u);
  EXPECT_EQ(a.data, nullptr);
  EXPECT_EQ(a.size, 0u);
}

TEST(MallocBuffer, MoveAssignment) {
  MallocBuffer a(64);
  MallocBuffer b(32);
  uint8_t *a_data = a.data;
  b = std::move(a);
  EXPECT_EQ(b.data, a_data);
  EXPECT_EQ(b.size, 64u);
  EXPECT_EQ(a.data, nullptr);
  EXPECT_EQ(a.size, 0u);
}

TEST(MallocBuffer, Release) {
  MallocBuffer buf(64);
  uint8_t *p = buf.release();
  EXPECT_NE(p, nullptr);
  EXPECT_EQ(buf.data, nullptr);
  EXPECT_EQ(buf.size, 0u);
  std::free(p);
}

// ═══════════════════════════════════════════════════════════════════════════════
// ParseElfInfo
// ═══════════════════════════════════════════════════════════════════════════════

TEST(ParseElfInfo, RejectsTruncatedInput) {
  uint8_t garbage[] = {0x7f, 'E', 'L', 'F', 0, 0, 0, 0};
  ElfInfo info;
  EXPECT_FALSE(ParseElfInfo(garbage, sizeof(garbage), info));
}

TEST(ParseElfInfo, RejectsNonElfInput) {
  uint8_t not_elf[64] = {};
  ElfInfo info;
  EXPECT_FALSE(ParseElfInfo(not_elf, sizeof(not_elf), info));
}

// ═══════════════════════════════════════════════════════════════════════════════
// BuildNopSledMap (via NopSled struct)
// ═══════════════════════════════════════════════════════════════════════════════

TEST(NopSled, StructDefaults) {
  NopSled sled{};
  EXPECT_EQ(sled.start, 0u);
  EXPECT_EQ(sled.end, 0u);
  EXPECT_EQ(sled.write_pos, 0u);
}

// ═══════════════════════════════════════════════════════════════════════════════
// RewriteRule (trimmed struct)
// ═══════════════════════════════════════════════════════════════════════════════

TEST(RewriteRule, Defaults) {
  RewriteRule rule;
  EXPECT_TRUE(rule.replace_mnemonic.empty());
  EXPECT_TRUE(rule.preserve_operands);
  EXPECT_TRUE(rule.replace_bytes.empty());
}

// ═══════════════════════════════════════════════════════════════════════════════
// ElfInfo / ElfSection / ElfSymbol struct layout
// ═══════════════════════════════════════════════════════════════════════════════

TEST(ElfInfo, DefaultsAreEmpty) {
  ElfInfo info;
  EXPECT_TRUE(info.sections.empty());
  EXPECT_TRUE(info.symbols.empty());
  EXPECT_EQ(info.text_section_idx, -1);
  EXPECT_EQ(info.text_idx, -1);
  EXPECT_EQ(info.text_offset, 0u);
  EXPECT_EQ(info.text_size, 0u);
  EXPECT_EQ(info.text_addr, 0u);
}
