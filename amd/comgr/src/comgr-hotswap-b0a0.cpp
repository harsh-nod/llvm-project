//===- comgr-hotswap-b0a0.cpp - GFX1250 B0-to-A0 instruction patching -----===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-internal.h"

// ── WMMA helpers ─────────────────────────────────────────────────────────────

static WmmaNopReq ClassifyWmmaNops(const std::string &mnemonic) {
  bool is_wmma = (mnemonic.find("v_wmma") == 0);
  bool is_swmmac = (mnemonic.find("v_swmmac") == 0);
  if (!is_wmma && !is_swmmac) return {4, 4};
  if (mnemonic.find("_iu8") != std::string::npos ||
      mnemonic.find("_iu4") != std::string::npos)
    return {8, 4};
  if (mnemonic.find("f8f6f4") != std::string::npos) return {1, 4};
  bool has_f8 = (mnemonic.find("_fp8") != std::string::npos ||
                 mnemonic.find("_f8") != std::string::npos ||
                 mnemonic.find("_bf8") != std::string::npos);
  if (has_f8) {
    if (mnemonic.find("16x16x128") != std::string::npos) return {3, 4};
    return {1, 4};
  }
  if (mnemonic.find("_f16") != std::string::npos ||
      mnemonic.find("_bf16") != std::string::npos)
    return {4, 4};
  return {4, 4};
}

static bool IsValuOpcode(unsigned opcode, const llvm::MCInstrInfo &MCII) {
  const llvm::MCInstrDesc &desc = MCII.get(opcode);
  uint64_t flags = desc.TSFlags;
  if (!(flags & llvm::SIInstrFlags::VALU))
    return false;
  if (flags & llvm::SIInstrFlags::IsWMMA)
    return false;
  llvm::StringRef name = MCII.getName(opcode);
  if (name.starts_with("V_NOP"))
    return false;
  return true;
}


static std::string FormatVgprRange(int base, int count) {
  if (count <= 1) return "v" + std::to_string(base);
  return "v[" + std::to_string(base) + ":" +
         std::to_string(base + count - 1) + "]";
}

// ── Mnemonic swap tables for B0→A0 ──────────────────────────────────────────

static const std::pair<std::string, std::string> kClusterLoadSwaps[] = {
    {"cluster_load_b32", "global_load_b32"},
    {"cluster_load_b64", "global_load_b64"},
    {"cluster_load_b128", "global_load_b128"},
    {"cluster_load_async_to_lds_b8", "global_load_async_to_lds_b8"},
    {"cluster_load_async_to_lds_b32", "global_load_async_to_lds_b32"},
    {"cluster_load_async_to_lds_b64", "global_load_async_to_lds_b64"},
    {"cluster_load_async_to_lds_b128", "global_load_async_to_lds_b128"},
};
static constexpr size_t kClusterLoadSwapsSize = std::size(kClusterLoadSwaps);

static const std::pair<std::string, std::string> kDs2AddrSwaps[] = {
    {"ds_load_2addr_b32", "ds_load_b32"},
    {"ds_load_2addr_b64", "ds_load_b64"},
    {"ds_load_2addr_stride64_b32", "ds_load_b32"},
    {"ds_load_2addr_stride64_b64", "ds_load_b64"},
    {"ds_store_2addr_b32", "ds_store_b32"},
    {"ds_store_2addr_b64", "ds_store_b64"},
    {"ds_store_2addr_stride64_b32", "ds_store_b32"},
    {"ds_store_2addr_stride64_b64", "ds_store_b64"},
    {"ds_storexchg_2addr_rtn_b32", "ds_storexchg_rtn_b32"},
    {"ds_storexchg_2addr_rtn_b64", "ds_storexchg_rtn_b64"},
    {"ds_storexchg_2addr_stride64_rtn_b32", "ds_storexchg_rtn_b32"},
    {"ds_storexchg_2addr_stride64_rtn_b64", "ds_storexchg_rtn_b64"},
};
static constexpr size_t kDs2AddrSwapsSize = std::size(kDs2AddrSwaps);

// ── DS 2-addr expansion ─────────────────────────────────────────────────────

static std::vector<std::string>
ExpandDs2AddrAsm(const std::string &printed_asm,
                 const std::string &from_mnemonic,
                 const std::string &to_mnemonic) {
  size_t start = printed_asm.find_first_not_of(" \t");
  if (start == std::string::npos) return {};
  size_t mnem_end = printed_asm.find_first_of(" \t", start);
  if (mnem_end == std::string::npos) return {};

  std::string operand_str = printed_asm.substr(mnem_end);

  auto extractOffsetVal = [](const std::string &s,
                             const std::string &key) -> std::string {
    size_t pos = s.find(key);
    if (pos == std::string::npos) return "0";
    size_t vstart = pos + key.size();
    size_t vend = vstart;
    while (vend < s.size() && s[vend] != ' ' && s[vend] != '\t' &&
           s[vend] != ',')
      vend++;
    return s.substr(vstart, vend - vstart);
  };
  std::string off0_val = extractOffsetVal(operand_str, "offset0:");
  std::string off1_val = extractOffsetVal(operand_str, "offset1:");

  if (from_mnemonic.find("stride64") != std::string::npos) {
    uint32_t elem_bytes =
        (from_mnemonic.find("_b64") != std::string::npos) ? 8 : 4;
    uint32_t scale = 64 * elem_bytes;
    auto scaleVal = [scale](std::string &val) {
      if (val == "0" || val.empty()) return;
      uint32_t v = 0;
      const char *begin = val.data();
      const char *end = val.data() + val.size();
      int base = 10;
      if (val.size() > 2 && val[0] == '0' && (val[1] == 'x' || val[1] == 'X')) {
        begin += 2;
        base = 16;
      }
      auto [p, ec] = std::from_chars(begin, end, v, base);
      if (ec == std::errc())
        val = std::to_string(v * scale);
    };
    scaleVal(off0_val);
    scaleVal(off1_val);
  }

  auto removeToken = [](std::string &s, const std::string &prefix) {
    size_t pos = s.find(prefix);
    if (pos == std::string::npos) return;
    size_t end = pos + prefix.size();
    while (end < s.size() && s[end] != ' ' && s[end] != '\t' && s[end] != ',')
      end++;
    while (end < s.size() && (s[end] == ' ' || s[end] == '\t' || s[end] == ','))
      end++;
    s.erase(pos, end - pos);
  };
  removeToken(operand_str, "offset0:");
  removeToken(operand_str, "offset1:");

  std::vector<std::string> ops;
  {
    std::string rest = operand_str;
    size_t s = rest.find_first_not_of(" \t");
    if (s != std::string::npos) rest = rest.substr(s);
    while (!rest.empty()) {
      size_t comma = rest.find(',');
      if (comma == std::string::npos) {
        std::string tok = rest;
        s = tok.find_first_not_of(" \t");
        size_t e = tok.find_last_not_of(" \t");
        if (s != std::string::npos && e != std::string::npos)
          tok = tok.substr(s, e - s + 1);
        if (!tok.empty()) ops.push_back(tok);
        break;
      }
      std::string tok = rest.substr(0, comma);
      s = tok.find_first_not_of(" \t");
      size_t e = tok.find_last_not_of(" \t");
      if (s != std::string::npos && e != std::string::npos)
        tok = tok.substr(s, e - s + 1);
      if (!tok.empty()) ops.push_back(tok);
      rest = rest.substr(comma + 1);
      s = rest.find_first_not_of(" \t");
      if (s != std::string::npos) rest = rest.substr(s);
    }
  }

  std::vector<std::string> clean_ops;
  for (auto &op : ops) {
    if (op.find("offset") == std::string::npos)
      clean_ops.push_back(op);
  }
  ops = clean_ops;

  auto extractFirstOfPair = [](const std::string &reg) -> std::string {
    size_t bracket = reg.find('[');
    if (bracket == std::string::npos) return reg;
    size_t colon = reg.find(':', bracket);
    if (colon == std::string::npos) return reg;
    return reg.substr(0, bracket) +
           reg.substr(bracket + 1, colon - bracket - 1);
  };
  auto extractSecondOfPair = [](const std::string &reg) -> std::string {
    size_t colon = reg.find(':');
    size_t close = reg.find(']');
    if (colon == std::string::npos || close == std::string::npos) return reg;
    return reg.substr(0, reg.find('[')) +
           reg.substr(colon + 1, close - colon - 1);
  };

  auto withOffset = [](const std::string &base,
                       const std::string &off) -> std::string {
    if (off == "0" || off.empty()) return base;
    return base + " offset:" + off;
  };

  bool is_load = (from_mnemonic.find("ds_load") == 0);
  bool is_store = (from_mnemonic.find("ds_store_") == 0 &&
                   from_mnemonic.find("xchg") == std::string::npos);

  if (is_load && ops.size() >= 2) {
    std::string d0 = extractFirstOfPair(ops[0]);
    std::string d1 = extractSecondOfPair(ops[0]);
    std::string addr = ops[1];
    return {
        withOffset(to_mnemonic + " " + d0 + ", " + addr, off0_val),
        withOffset(to_mnemonic + " " + d1 + ", " + addr, off1_val),
    };
  }
  if (is_store && ops.size() >= 3) {
    return {
        withOffset(to_mnemonic + " " + ops[0] + ", " + ops[1], off0_val),
        withOffset(to_mnemonic + " " + ops[0] + ", " + ops[2], off1_val),
    };
  }

  return {};
}

// ── BuildNopSledMap ─────────────────────────────────────────────────────────

static std::vector<NopSled>
BuildNopSledMap(const std::vector<InternalDecodedInst> &decoded,
                const llvm::MCInstrInfo &MCII) {
  std::vector<NopSled> sleds;
  for (size_t i = 0; i < decoded.size(); ++i) {
    unsigned opc = decoded[i].inst.getOpcode();
    const llvm::MCInstrDesc &desc = MCII.get(opc);
    if (desc.isTerminator() && desc.isBarrier() &&
        !desc.isBranch() && !desc.isCall()) {
      uint64_t sled_start = decoded[i].offset + decoded[i].size;
      uint64_t sled_end = sled_start;
      for (size_t j = i + 1; j < decoded.size(); ++j) {
        if (decoded[j].mnemonic == "s_nop")
          sled_end = decoded[j].offset + decoded[j].size;
        else
          break;
      }
      if (sled_end > sled_start + 8)
        sleds.push_back({sled_start, sled_end, sled_start});
    }
  }
  return sleds;
}

// ── ValidateWmmaCoexecHazards ────────────────────────────────────────────────

static std::vector<WmmaHazard>
ValidateWmmaCoexecHazards(const std::vector<InternalDecodedInst> &decoded,
                           const uint8_t *text,
                           const LLVMState &llvm_state,
                           const llvm::MCInstrInfo &MCII) {
  (void)text;
  std::vector<WmmaHazard> hazards;
  int wmma_scanned = 0;

  for (size_t i = 0; i < decoded.size(); ++i) {
    const auto &di = decoded[i];
    if (di.mnemonic.find("v_wmma") != 0 &&
        di.mnemonic.find("v_swmmac") != 0)
      continue;

    ++wmma_scanned;
    WmmaNopReq req = ClassifyWmmaNops(di.mnemonic);
    if (req.a0_nops <= req.b0_nops) continue;

    int count = 0;
    for (size_t j = i + 1; j < decoded.size(); ++j) {
      const auto &dj = decoded[j];
      unsigned opc = dj.inst.getOpcode();
      const llvm::MCInstrDesc &desc = MCII.get(opc);

      if (dj.mnemonic == "v_nop") {
        ++count;
        if (count >= req.a0_nops) break;
        continue;
      }
      uint64_t flags = desc.TSFlags;
      if (flags & llvm::SIInstrFlags::SALU) {
        if (desc.isBranch() || desc.isTerminator() || desc.isCall())
          break;
        continue;
      }
      if (IsValuOpcode(opc, MCII)) {
        if (!CheckVgprOverlap(di.inst, dj.inst, *llvm_state.MRI)) {
          ++count;
          if (count >= req.a0_nops) break;
          continue;
        }
        if (count < req.a0_nops) {
          hazards.push_back({i, j, count, req.a0_nops, req.a0_nops - count});
          HotswapLog(HotswapLogLevel::Info) << "hotswap: B0->A0 WMMA co-exec hazard @0x"
                    << std::hex << di.offset
                    << ": " << di.mnemonic
                    << " needs " << std::dec << req.a0_nops
                    << " V_NOPs for A0, only " << count
                    << " found before " << dj.mnemonic
                    << " @0x" << std::hex << dj.offset
                    << std::dec << "\n";
        }
        break;
      }
      break;
    }
  }

  HotswapLog(HotswapLogLevel::Info) << "hotswap: B0->A0 WMMA co-exec validation: "
            << hazards.size() << " hazards ("
            << wmma_scanned << " WMMA instructions scanned)\n";

  return hazards;
}

// ── Common sled-or-trampoline insertion ─────────────────────────────────────

[[nodiscard]] static bool EmitReplacementCode(PatchContext &ctx, uint64_t inst_offset,
                                uint32_t inst_size,
                                const std::vector<uint8_t> &replacement_bytes,
                                const char *patch_desc = nullptr) {
  if (replacement_bytes.empty()) return false;

  uint32_t tramp_size = static_cast<uint32_t>(replacement_bytes.size() + 4);

  NopSled *sled = FindNearestSled(ctx.nop_sleds, inst_offset, tramp_size);
  if (sled) {
    uint64_t tp = sled->write_pos;
    std::memcpy(ctx.text + tp, replacement_bytes.data(),
                replacement_bytes.size());
    uint8_t br_back[4];
    if (EncodeSBranch(tp + replacement_bytes.size(),
                      inst_offset + inst_size, br_back, true)) {
      std::memcpy(ctx.text + tp + replacement_bytes.size(), br_back, 4);
      uint8_t br_fwd[4];
      if (EncodeSBranch(inst_offset, tp, br_fwd, true)) {
        std::memcpy(ctx.text + inst_offset, br_fwd, 4);
        for (uint32_t i = 4; i < inst_size; i += 4) {
          uint8_t nop[4];
          EncodeSNop(nop);
          std::memcpy(ctx.text + inst_offset + i, nop, 4);
        }
        sled->write_pos += tramp_size;
        if (patch_desc) {
          HotswapLog(HotswapLogLevel::Info) << "hotswap: B0->A0 @0x" << std::hex << inst_offset
                    << ": " << patch_desc << " via sled @0x" << tp
                    << std::dec << "\n";
        }
        return true;
      }
    }
  }

  Trampoline t;
  t.original_offset = inst_offset;
  t.original_size = inst_size;
  t.bytes.resize(tramp_size);
  std::memcpy(t.bytes.data(), replacement_bytes.data(),
              replacement_bytes.size());
  uint8_t placeholder[4] = {0};
  std::memcpy(t.bytes.data() + replacement_bytes.size(), placeholder, 4);
  ctx.out_trampolines.push_back(std::move(t));
  if (patch_desc) {
    HotswapLog(HotswapLogLevel::Info) << "hotswap: B0->A0 @0x" << std::hex << inst_offset
              << ": " << patch_desc << " deferred for ELF growth"
              << std::dec << "\n";
  }
  return true;
}

// ── Per-patch functions ─────────────────────────────────────────────────────

[[nodiscard]] static bool ApplyPatch1_ClusterLoad(PatchContext &ctx, size_t idx) {
  auto &di = ctx.decoded[idx];
  for (size_t swap_i = 0; swap_i < kClusterLoadSwapsSize; ++swap_i) {
    const auto &swap = kClusterLoadSwaps[swap_i];
    if (di.mnemonic == swap.first) {
      RewriteRule rule;
      rule.replace_mnemonic = swap.second;
      rule.preserve_operands = true;
      if (ApplyMnemonicSwap(rule, di, ctx.text, ctx.llvm_state)) {
        di.mnemonic = swap.second;
        return true;
      }
      return false;
    }
  }
  return false;
}

[[nodiscard]] static bool ApplyPatch2_Ds2Addr(PatchContext &ctx, size_t idx) {
  auto &di = ctx.decoded[idx];
  for (size_t swap_i = 0; swap_i < kDs2AddrSwapsSize; ++swap_i) {
    const auto &swap = kDs2AddrSwaps[swap_i];
    if (di.mnemonic == swap.first) {
      if (di.mnemonic.find("stride64") == std::string::npos) return false;
      std::string inst_str = PrintInst(di, ctx.llvm_state);
      if (inst_str.empty()) return false;
      std::vector<std::string> asm_lines =
          ExpandDs2AddrAsm(inst_str, swap.first, swap.second);
      if (asm_lines.size() != 2) return false;
      auto bytes0 = AssembleSingleInst(asm_lines[0], ctx.llvm_state);
      auto bytes1 = AssembleSingleInst(asm_lines[1], ctx.llvm_state);
      if (bytes0.empty() || bytes1.empty()) return false;

      uint64_t tramp_offset = ctx.text_size;
      for (auto &t : ctx.out_trampolines)
        tramp_offset += t.bytes.size();

      Trampoline tramp;
      tramp.original_offset = di.offset;
      tramp.original_size = di.size;
      tramp.bytes.insert(tramp.bytes.end(), bytes0.begin(), bytes0.end());
      tramp.bytes.insert(tramp.bytes.end(), bytes1.begin(), bytes1.end());

      uint8_t br_back[4];
      if (!EncodeSBranch(tramp_offset + tramp.bytes.size(),
                         di.offset + di.size, br_back, true))
        return false;
      tramp.bytes.insert(tramp.bytes.end(), br_back, br_back + 4);
      ctx.out_trampolines.push_back(std::move(tramp));
      di.mnemonic = "<replaced>";
      return true;
    }
  }
  return false;
}

[[nodiscard]] static bool ApplyPatch3_SClause(PatchContext &ctx, size_t idx) {
  auto &di = ctx.decoded[idx];
  if (di.mnemonic != "s_clause") return false;
  RewriteRule rule;
  rule.replace_bytes = {0x00, 0x00, 0x80, 0xBF};
  if (ApplyByteReplace(rule, di.offset, di.size, ctx.text, ctx.text_size)) {
    di.mnemonic = "s_nop";
    return true;
  }
  return false;
}

[[nodiscard]] static bool ApplyPatch4_TensorLoadToLds(PatchContext &ctx, size_t idx) {
  auto &di = ctx.decoded[idx];
  if (di.mnemonic != "tensor_load_to_lds") return false;
  if (idx > 0 && ctx.decoded[idx - 1].mnemonic == "s_pack_hh_b32_b16")
    return false;
  std::string inst_str = PrintInst(di, ctx.llvm_state);
  if (inst_str.empty()) return false;

  size_t first_comma = inst_str.find(',');
  if (first_comma == std::string::npos) return false;
  std::string after = inst_str.substr(first_comma + 1);

  size_t s_pos = after.find("s[");
  if (s_pos == std::string::npos) {
    s_pos = after.find_first_of("s");
    if (s_pos == std::string::npos) return false;
  }

  std::string base_sreg;
  size_t bracket_pos = after.find('[', s_pos);
  if (bracket_pos != std::string::npos && bracket_pos == s_pos + 1) {
    size_t colon = after.find(':', bracket_pos);
    if (colon != std::string::npos) {
      std::string num =
          after.substr(bracket_pos + 1, colon - bracket_pos - 1);
      base_sreg = "s" + num;
    }
  } else {
    size_t num_start = s_pos + 1;
    size_t num_end = num_start;
    while (num_end < after.size() && after[num_end] >= '0' &&
           after[num_end] <= '9')
      num_end++;
    if (num_end > num_start)
      base_sreg = "s" + after.substr(num_start, num_end - num_start);
  }
  if (base_sreg.empty()) return false;

  std::string pack_asm =
      "s_pack_hh_b32_b16 " + base_sreg + ", 0, " + base_sreg;
  auto pack_bytes = AssembleSingleInst(pack_asm, ctx.llvm_state);
  if (pack_bytes.empty() || pack_bytes.size() != 4) return false;

  std::vector<uint8_t> replacement;
  replacement.insert(replacement.end(), pack_bytes.begin(), pack_bytes.end());
  replacement.insert(replacement.end(), ctx.text + di.offset,
                     ctx.text + di.offset + di.size);

  if (!EmitReplacementCode(ctx, di.offset, di.size, replacement))
    return false;
  di.mnemonic = "<replaced>";
  return true;
}

static uint32_t ApplyPatch5_WmmaHazard(PatchContext &ctx) {
  uint32_t patched = 0;
  auto hazards = ValidateWmmaCoexecHazards(ctx.decoded, ctx.text, ctx.llvm_state,
                                            *ctx.llvm_state.MCII);
  if (hazards.empty()) return 0;
  auto vnop_bytes = AssembleSingleInst("v_nop", ctx.llvm_state);
  if (vnop_bytes.empty() || vnop_bytes.size() != 4) return 0;
  for (auto &h : hazards) {
    const auto &valu = ctx.decoded[h.valu_idx];
    uint32_t valu_size = valu.size;

    std::vector<uint8_t> replacement;
    for (int n = 0; n < h.deficit; ++n)
      replacement.insert(replacement.end(), vnop_bytes.begin(),
                         vnop_bytes.end());
    replacement.insert(replacement.end(), ctx.text + valu.offset,
                       ctx.text + valu.offset + valu_size);

    if (EmitReplacementCode(ctx, valu.offset, valu_size, replacement))
      ++patched;
  }
  return patched;
}

[[nodiscard]] static bool ApplyPatch6_Fp8WmmaSplit(PatchContext &ctx, size_t idx) {
  auto &di = ctx.decoded[idx];
  if (di.mnemonic.find("16x16x128") == std::string::npos ||
      (di.mnemonic.find("_fp8") == std::string::npos &&
       di.mnemonic.find("_bf8") == std::string::npos) ||
      di.mnemonic.find("f8f6f4") != std::string::npos)
    return false;

  auto [d_base, d_count] =
      GetOperandVgprRange(di.inst, 0, *ctx.llvm_state.MRI);
  auto [a_base, a_count] =
      GetOperandVgprRange(di.inst, 1, *ctx.llvm_state.MRI);
  auto [b_base, b_count] =
      GetOperandVgprRange(di.inst, 2, *ctx.llvm_state.MRI);
  if (d_base < 0 || a_base < 0 || b_base < 0 || a_count < 16 ||
      b_count < 16)
    return false;

  std::string mnem64 = di.mnemonic;
  size_t pos128 = mnem64.find("16x16x128");
  if (pos128 != std::string::npos)
    mnem64.replace(pos128, 9, "16x16x64");
  std::string asm1 =
      mnem64 + " " + FormatVgprRange(d_base, d_count) + ", " +
      FormatVgprRange(a_base, 8) + ", " + FormatVgprRange(b_base, 8) +
      ", " + FormatVgprRange(d_base, d_count);
  std::string asm2 =
      mnem64 + " " + FormatVgprRange(d_base, d_count) + ", " +
      FormatVgprRange(a_base + 8, 8) + ", " +
      FormatVgprRange(b_base + 8, 8) + ", " +
      FormatVgprRange(d_base, d_count);
  auto enc1 = AssembleSingleInst(asm1, ctx.llvm_state);
  auto enc2 = AssembleSingleInst(asm2, ctx.llvm_state);
  if (enc1.empty() || enc2.empty()) return false;

  std::vector<uint8_t> replacement;
  replacement.insert(replacement.end(), enc1.begin(), enc1.end());
  replacement.insert(replacement.end(), enc2.begin(), enc2.end());

  std::string desc = di.mnemonic + " -> 2x " + mnem64;
  if (!EmitReplacementCode(ctx, di.offset, di.size, replacement, desc.c_str()))
    return false;
  di.mnemonic = "<replaced>";
  return true;
}

[[nodiscard]] static bool ApplyPatch7_F4WmmaSplit(PatchContext &ctx, size_t idx) {
  auto &di = ctx.decoded[idx];
  if (di.mnemonic.find("32x16x128_f4") == std::string::npos ||
      di.mnemonic.find("v_wmma") != 0)
    return false;

  bool is_scaled = (di.mnemonic.find("_scale_") != std::string::npos ||
                    di.mnemonic.find("_scale16_") != std::string::npos);
  if (is_scaled) return false;

  auto [d_base, d_count] = GetOperandVgprRange(di.inst, 0, *ctx.llvm_state.MRI);
  auto [a_base, a_count] = GetOperandVgprRange(di.inst, 1, *ctx.llvm_state.MRI);
  auto [b_base, b_count] = GetOperandVgprRange(di.inst, 2, *ctx.llvm_state.MRI);

  if (d_base < 0 || a_base < 0 || b_base < 0 || d_count < 16)
    return false;

  std::string asm1 = "v_wmma_f32_16x16x128_f8f6f4 "
      + FormatVgprRange(d_base, 8) + ", "
      + FormatVgprRange(a_base, 8) + ", "
      + FormatVgprRange(b_base, b_count) + ", "
      + FormatVgprRange(d_base, 8)
      + " matrix_a_fmt:MATRIX_FMT_FP4 matrix_b_fmt:MATRIX_FMT_FP4";
  std::string asm2 = "v_wmma_f32_16x16x128_f8f6f4 "
      + FormatVgprRange(d_base + 8, 8) + ", "
      + FormatVgprRange(a_base + 8, 8) + ", "
      + FormatVgprRange(b_base, b_count) + ", "
      + FormatVgprRange(d_base + 8, 8)
      + " matrix_a_fmt:MATRIX_FMT_FP4 matrix_b_fmt:MATRIX_FMT_FP4";

  auto enc1 = AssembleSingleInst(asm1, ctx.llvm_state);
  auto enc2 = AssembleSingleInst(asm2, ctx.llvm_state);

  if (enc1.empty() || enc2.empty()) return false;

  std::vector<uint8_t> replacement;
  replacement.insert(replacement.end(), enc1.begin(), enc1.end());
  replacement.insert(replacement.end(), enc2.begin(), enc2.end());

  std::string desc =
      di.mnemonic + " -> 2x v_wmma_f32_16x16x128_f8f6f4";
  if (!EmitReplacementCode(ctx, di.offset, di.size, replacement, desc.c_str()))
    return false;
  di.mnemonic = "<replaced>";
  return true;
}

[[nodiscard]] static bool ApplyPatch8_E5M3CvtEmulation(PatchContext &ctx, size_t idx) {
  auto &di = ctx.decoded[idx];
  if (!((di.mnemonic.find("v_cvt_f32_fp8") == 0 ||
         di.mnemonic.find("v_cvt_pk_fp8_f32") == 0 ||
         di.mnemonic.find("v_cvt_sr_fp8_f32") == 0) && di.size >= 8))
    return false;

  uint32_t dw0;
  std::memcpy(&dw0, ctx.text + di.offset, 4);
  bool has_clamp = (dw0 >> 15) & 1;
  if (!has_clamp) return false;

  auto [dst_base, dst_count] = GetOperandVgprRange(di.inst, 0, *ctx.llvm_state.MRI);
  auto [src_base, src_count] = GetOperandVgprRange(di.inst, 1, *ctx.llvm_state.MRI);

  if (dst_base < 0 || src_base < 0) return false;

  std::string kernel = FindKernelAtOffset(ctx.elf_info, di.offset);
  int kd_vgprs = GetKernelVgprCount(ctx.elf_data, ctx.elf_size, ctx.elf_info, kernel);
  ScratchAllocator alloc(ctx.liveness.live_before[idx], kd_vgprs);
  int s0 = alloc.Alloc(), s1 = alloc.Alloc();
  int s2 = alloc.Alloc(), s3 = alloc.Alloc();
  if (s0 < 0 || s1 < 0 || s2 < 0 || s3 < 0) {
    HotswapLog(HotswapLogLevel::Error) << "hotswap: B0->A0 @0x" << std::hex << di.offset
              << ": scratch allocation failed, skipping patch" << std::dec << "\n";
    return false;
  }
  std::string sv0 = "v" + std::to_string(s0);
  std::string sv1 = "v" + std::to_string(s1);
  std::string sv2 = "v" + std::to_string(s2);
  std::string sv3 = "v" + std::to_string(s3);
  for (int v : {s0, s1, s2, s3}) {
    if (v < kd_vgprs)
      ctx.kernel_stats[kernel].scratch_reused++;
    else
      ctx.kernel_stats[kernel].scratch_above_kd++;
  }
  ctx.kernel_stats[kernel].extra_vgprs =
      std::max(ctx.kernel_stats[kernel].extra_vgprs,
               alloc.ExtraVgprsNeeded());
  {
    ScratchPatchInfo spi;
    spi.offset = di.offset;
    for (int v : {s0, s1, s2, s3}) spi.scratch_regs.set(v);
    ctx.out_scratch_patches.push_back(std::move(spi));
  }

  std::string dst = FormatVgprRange(dst_base, dst_count);
  std::string src = "v" + std::to_string(src_base);
  std::vector<std::string> emu_lines;

  if (di.mnemonic == "v_cvt_f32_fp8") {
    int opsel = (dw0 >> 11) & 0x3;
    int byte_offset = opsel * 8;
    emu_lines = {
      "v_bfe_u32 " + sv0 + ", " + src + ", " + std::to_string(byte_offset) + ", 8",
      "v_bfe_u32 " + sv1 + ", " + sv0 + ", 0, 3",
      "v_bfe_u32 " + sv2 + ", " + sv0 + ", 3, 5",
      "v_bfe_u32 " + sv3 + ", " + sv0 + ", 7, 1",
      "v_lshlrev_b32 " + sv1 + ", 20, " + sv1,
      "v_add_nc_u32 " + sv2 + ", 112, " + sv2,
      "v_lshlrev_b32 " + sv2 + ", 23, " + sv2,
      "v_lshlrev_b32 " + sv3 + ", 31, " + sv3,
      "v_or3_b32 " + dst + ", " + sv1 + ", " + sv2 + ", " + sv3,
    };
  } else if (di.mnemonic == "v_cvt_pk_fp8_f32") {
    auto [src2_base, src2_count] = GetOperandVgprRange(di.inst, 2, *ctx.llvm_state.MRI);
    std::string src2 = (src2_base >= 0) ? ("v" + std::to_string(src2_base)) : src;
    emu_lines = {
      "v_bfe_u32 " + sv0 + ", " + src + ", 23, 8",
      "v_bfe_u32 " + sv1 + ", " + src + ", 20, 3",
      "v_lshrrev_b32 " + sv2 + ", 31, " + src,
      "v_sub_nc_u32 " + sv0 + ", " + sv0 + ", 112",
      "v_bfe_u32 " + sv3 + ", " + src + ", 19, 1",
      "v_add_nc_u32 " + sv1 + ", " + sv1 + ", " + sv3,
      "v_max_i32 " + sv0 + ", " + sv0 + ", 0",
      "v_min_i32 " + sv0 + ", " + sv0 + ", 31",
      "v_lshlrev_b32 " + sv0 + ", 3, " + sv0,
      "v_or_b32 " + sv0 + ", " + sv0 + ", " + sv1,
      "v_lshlrev_b32 " + sv2 + ", 7, " + sv2,
      "v_or_b32 " + sv0 + ", " + sv0 + ", " + sv2,
      "v_bfe_u32 " + sv1 + ", " + src2 + ", 23, 8",
      "v_bfe_u32 " + sv2 + ", " + src2 + ", 20, 3",
      "v_lshrrev_b32 " + sv3 + ", 31, " + src2,
      "v_sub_nc_u32 " + sv1 + ", " + sv1 + ", 112",
      "v_bfe_u32 " + dst + ", " + src2 + ", 19, 1",
      "v_add_nc_u32 " + sv2 + ", " + sv2 + ", " + dst,
      "v_max_i32 " + sv1 + ", " + sv1 + ", 0",
      "v_min_i32 " + sv1 + ", " + sv1 + ", 31",
      "v_lshlrev_b32 " + sv1 + ", 3, " + sv1,
      "v_or_b32 " + sv1 + ", " + sv1 + ", " + sv2,
      "v_lshlrev_b32 " + sv3 + ", 7, " + sv3,
      "v_or_b32 " + sv1 + ", " + sv1 + ", " + sv3,
      "v_lshlrev_b32 " + sv1 + ", 8, " + sv1,
      "v_or_b32 " + dst + ", " + sv0 + ", " + sv1,
    };
  } else {
    emu_lines = {
      "v_bfe_u32 " + sv0 + ", " + src + ", 23, 8",
      "v_bfe_u32 " + sv1 + ", " + src + ", 20, 3",
      "v_lshrrev_b32 " + sv2 + ", 31, " + src,
      "v_sub_nc_u32 " + sv0 + ", " + sv0 + ", 112",
      "v_bfe_u32 " + sv3 + ", " + src + ", 19, 1",
      "v_add_nc_u32 " + sv1 + ", " + sv1 + ", " + sv3,
      "v_max_i32 " + sv0 + ", " + sv0 + ", 0",
      "v_min_i32 " + sv0 + ", " + sv0 + ", 31",
      "v_lshlrev_b32 " + sv0 + ", 3, " + sv0,
      "v_or_b32 " + sv0 + ", " + sv0 + ", " + sv1,
      "v_lshlrev_b32 " + sv2 + ", 7, " + sv2,
      "v_or_b32 " + dst + ", " + sv0 + ", " + sv2,
    };
  }

  std::string joined;
  for (const auto &line : emu_lines) {
    if (!joined.empty()) joined += "\n";
    joined += line;
  }
  auto enc = AssembleSingleInst(joined, ctx.llvm_state);
  if (enc.empty()) return false;

  std::string desc = di.mnemonic + " CLAMP=1 (E5M3) -> VALU emulation";
  if (!EmitReplacementCode(ctx, di.offset, di.size, enc, desc.c_str()))
    return false;
  di.mnemonic = "<replaced>";
  return true;
}

[[nodiscard]] static bool ApplyPatch9_Scale16Decomposition(PatchContext &ctx, size_t idx) {
  auto &di = ctx.decoded[idx];

  if (di.mnemonic == "v_wmma_scale16_f32_16x16x128_f8f6f4") {
    std::string inst_str = PrintInst(di, ctx.llvm_state);
    if (inst_str.empty()) return false;

    size_t mnem_start = inst_str.find_first_not_of(" \t");
    if (mnem_start == std::string::npos) mnem_start = 0;
    size_t mnem_end = inst_str.find_first_of(" \t", mnem_start);
    if (mnem_end == std::string::npos) mnem_end = inst_str.size();
    std::string ops_and_mods = inst_str.substr(mnem_end);
    size_t ops_start = ops_and_mods.find_first_not_of(" \t");
    if (ops_start != std::string::npos) ops_and_mods = ops_and_mods.substr(ops_start);

    std::string modifiers;
    {
      size_t mod_pos = ops_and_mods.find("matrix_");
      if (mod_pos == std::string::npos) mod_pos = ops_and_mods.find("neg_");
      if (mod_pos != std::string::npos) {
        modifiers = " " + ops_and_mods.substr(mod_pos);
        while (!modifiers.empty() && modifiers.back() == ' ') modifiers.pop_back();
        ops_and_mods = ops_and_mods.substr(0, mod_pos);
      }
    }

    std::vector<std::string> ops;
    {
      std::istringstream ss(ops_and_mods);
      std::string tok;
      while (std::getline(ss, tok, ',')) {
        size_t s = tok.find_first_not_of(" \t");
        size_t e = tok.find_last_not_of(" \t");
        if (s != std::string::npos && e != std::string::npos)
          ops.push_back(tok.substr(s, e - s + 1));
      }
    }

    if (ops.size() < 6) return false;

    std::string d_str = ops[0];
    std::string a_str = ops[1];
    std::string b_str = ops[2];
    std::string sa_str = ops[4];
    std::string sb_str = ops[5];

    auto extractBase = [](const std::string &s) -> int {
      size_t pos = s.find('[');
      if (pos != std::string::npos) {
        size_t colon = s.find(':', pos);
        if (colon != std::string::npos) {
          int val = -1;
          std::from_chars(s.data() + pos + 1, s.data() + colon, val);
          if (val >= 0) return val;
        }
      }
      size_t vpos = s.find('v');
      if (vpos != std::string::npos) {
        int val = -1;
        std::from_chars(s.data() + vpos + 1, s.data() + s.size(), val);
        if (val >= 0) return val;
      }
      return -1;
    };

    int sa_base = extractBase(sa_str);
    int sb_base = extractBase(sb_str);

    if (sa_base < 0 || sb_base < 0) return false;

    std::string kernel_p9 = FindKernelAtOffset(ctx.elf_info, di.offset);
    int kd_vgprs_p9 = GetKernelVgprCount(ctx.elf_data, ctx.elf_size, ctx.elf_info, kernel_p9);
    ScratchAllocator alloc_p9(ctx.liveness.live_before[idx], kd_vgprs_p9);
    int p9s0 = alloc_p9.Alloc(), p9s1 = alloc_p9.Alloc();
    if (p9s0 < 0 || p9s1 < 0) {
      HotswapLog(HotswapLogLevel::Error) << "hotswap: B0->A0 @0x" << std::hex << di.offset
                << ": scratch allocation failed, skipping patch" << std::dec << "\n";
      return false;
    }
    std::string p9sv0 = "v" + std::to_string(p9s0);
    std::string p9sv1 = "v" + std::to_string(p9s1);
    for (int v : {p9s0, p9s1}) {
      if (v < kd_vgprs_p9)
        ctx.kernel_stats[kernel_p9].scratch_reused++;
      else
        ctx.kernel_stats[kernel_p9].scratch_above_kd++;
    }
    ctx.kernel_stats[kernel_p9].extra_vgprs =
        std::max(ctx.kernel_stats[kernel_p9].extra_vgprs,
                 alloc_p9.ExtraVgprsNeeded());
    {
      ScratchPatchInfo spi;
      spi.offset = di.offset;
      for (int v : {p9s0, p9s1}) spi.scratch_regs.set(v);
      ctx.out_scratch_patches.push_back(std::move(spi));
    }

    std::string repack_sa = "v_perm_b32 " + p9sv0 + ", v" + std::to_string(sa_base) +
        ", v" + std::to_string(sa_base) + ", 0x05010400";
    std::string repack_sb = "v_perm_b32 " + p9sv1 + ", v" + std::to_string(sb_base) +
        ", v" + std::to_string(sb_base) + ", 0x05010400";

    std::string wmma_asm = "v_wmma_scale_f32_16x16x128_f8f6f4 "
        + d_str + ", " + a_str + ", " + b_str + ", "
        + d_str + ", " + p9sv0 + ", " + p9sv1 + modifiers;

    std::string all_asm = repack_sa + "\n" + repack_sb + "\n" + wmma_asm;
    auto enc = AssembleSingleInst(all_asm, ctx.llvm_state);

    if (enc.empty()) return false;

    std::string desc = di.mnemonic + " -> block32 decomposition";
    if (!EmitReplacementCode(ctx, di.offset, di.size, enc, desc.c_str()))
      return false;
    di.mnemonic = "<replaced>";
    return true;
  }

  if (di.mnemonic == "v_wmma_ld_scale16_paired_b64") {
    auto [dst_base, dst_count] = GetOperandVgprRange(di.inst, 0, *ctx.llvm_state.MRI);
    auto [src_base, src_count] = GetOperandVgprRange(di.inst, 1, *ctx.llvm_state.MRI);

    if (dst_base < 0 || src_base < 0) return false;

    std::string load_asm = "v_wmma_ld_scale_paired_b32 v"
        + std::to_string(dst_base) + ", v" + std::to_string(src_base);

    auto enc = AssembleSingleInst(load_asm, ctx.llvm_state);

    if (enc.empty()) return false;
    if (enc.size() > di.size) return false;

    std::memcpy(ctx.text + di.offset, enc.data(), enc.size());
    for (uint32_t i = static_cast<uint32_t>(enc.size()); i < di.size;
         i += 4) {
      uint8_t nop[4]; EncodeSNop(nop);
      std::memcpy(ctx.text + di.offset + i, nop, 4);
    }
    HotswapLog(HotswapLogLevel::Info) << "hotswap: B0->A0 @0x" << std::hex << di.offset
              << ": v_wmma_ld_scale16 -> block32 v_wmma_ld_scale_paired_b32"
              << std::dec << "\n";
    di.mnemonic = "<replaced>";
    return true;
  }

  return false;
}

// ── ApplyGfx1250B0toA0Rules ─────────────────────────────────────────────────

static uint32_t
ApplyGfx1250B0toA0Rules(std::vector<InternalDecodedInst> &decoded,
                        uint8_t *text, uint64_t text_size,
                        const LLVMState &llvm_state,
                        std::vector<Trampoline> &out_trampolines,
                        uint8_t *elf_data, size_t elf_size,
                        const ElfInfo &elf_info,
                        std::vector<ScratchPatchInfo> &out_scratch_patches) {
  uint32_t patched = 0;
  std::vector<NopSled> nop_sleds = BuildNopSledMap(decoded, *llvm_state.MCII);

  CFG cfg = BuildCFG(decoded, *llvm_state.MCII);
  LivenessInfo liveness = ComputeLiveness(decoded, cfg,
                                          *llvm_state.MCII, *llvm_state.MRI);

  if (!liveness.converged) {
    HotswapLog(HotswapLogLevel::Error) << "hotswap: WARNING: liveness analysis did not converge, "
                 "using conservative all-VGPRs-live fallback\n";
    llvm::BitVector all_vgprs(256);
    all_vgprs.set(0, 256);
    for (size_t i = 0; i < liveness.live_before.size(); ++i) {
      liveness.live_before[i] = all_vgprs;
      liveness.live_after[i] = all_vgprs;
    }
  }

  std::unordered_map<std::string, KernelPatchStats> kernel_stats;

  PatchContext ctx{decoded, text, text_size, llvm_state, out_trampolines,
                   nop_sleds, elf_data, elf_size, elf_info, liveness,
                   kernel_stats, out_scratch_patches};

  for (size_t idx = 0; idx < decoded.size(); ++idx) {
    auto &di = decoded[idx];
    if (di.mnemonic == "<unknown>" || di.mnemonic == "<replaced>") continue;

    if (ApplyPatch1_ClusterLoad(ctx, idx)) { ++patched; continue; }
    if (ApplyPatch2_Ds2Addr(ctx, idx)) { ++patched; continue; }
    if (ApplyPatch3_SClause(ctx, idx)) { ++patched; continue; }
    if (ApplyPatch4_TensorLoadToLds(ctx, idx)) { ++patched; continue; }
    if (ApplyPatch6_Fp8WmmaSplit(ctx, idx)) { ++patched; continue; }
    if (ApplyPatch7_F4WmmaSplit(ctx, idx)) { ++patched; continue; }
    if (ApplyPatch8_E5M3CvtEmulation(ctx, idx)) { ++patched; continue; }
    if (ApplyPatch9_Scale16Decomposition(ctx, idx)) { ++patched; continue; }
  }

  patched += ApplyPatch5_WmmaHazard(ctx);

  for (const auto &kv : kernel_stats) {
    const std::string &kname = kv.first;
    const auto &stats = kv.second;
    if (kname.empty()) continue;
    int vgprs_before = GetKernelVgprCount(elf_data, elf_size, elf_info, kname);
    if (stats.extra_vgprs > 0)
      UpdateKernelDescriptor(elf_data, elf_size, elf_info, kname,
                             stats.extra_vgprs, 0);
    int vgprs_after = GetKernelVgprCount(elf_data, elf_size, elf_info, kname);
    HotswapLog(HotswapLogLevel::Info) << "hotswap: liveness: kernel " << kname
              << ": vgprs_before=" << vgprs_before
              << ", vgprs_after=" << vgprs_after
              << ", scratch_reused=" << stats.scratch_reused
              << ", scratch_above_kd=" << stats.scratch_above_kd << "\n";
  }

  return patched;
}

// ── RetargetCodeObjectB0A0Grow ───────────────────────────────────────────────

amd_comgr_status_t
RetargetCodeObjectB0A0Grow(const void *elf_data, size_t elf_size,
                           void **out_data, size_t *out_size,
                           amd_comgr_hotswap_result_t *result) {
  const std::string isa = "amdgcn-amd-amdhsa--gfx1250";

  ElfInfo elf_info;
  const uint8_t *elf = static_cast<const uint8_t *>(elf_data);
  if (!ParseElfInfo(elf, elf_size, elf_info) || elf_info.text_size == 0) {
    MallocBuffer copy(elf_size);
    if (!copy) return AMD_COMGR_STATUS_ERROR;
    std::memcpy(copy.data, elf_data, elf_size);
    *out_size = elf_size;
    *out_data = copy.release();
    return AMD_COMGR_STATUS_SUCCESS;
  }

  LLVMState llvm_state = InitLLVMCached(isa);
  if (!llvm_state.valid) return AMD_COMGR_STATUS_ERROR;

  std::vector<uint8_t> buf(elf, elf + elf_size);
  uint8_t *text = buf.data() + elf_info.text_offset;

  std::vector<InternalDecodedInst> decoded;
  if (!DecodeTextSection(text, elf_info.text_size, llvm_state, decoded))
    return AMD_COMGR_STATUS_ERROR;

  std::vector<Trampoline> deferred;
  std::vector<ScratchPatchInfo> scratch_patches;
  uint32_t count = ApplyGfx1250B0toA0Rules(decoded, text, elf_info.text_size,
                                           llvm_state, deferred,
                                           buf.data(), buf.size(), elf_info,
                                           scratch_patches);
  result->rules_matched = count;

  if (!deferred.empty()) {
    uint64_t tramp_text_offset = elf_info.text_size;
    for (auto &t : deferred) {
      uint64_t tp = tramp_text_offset;
      tramp_text_offset += t.bytes.size();

      uint8_t br_back[4];
      uint64_t br_from = tp + t.bytes.size() - 4;
      uint64_t br_to = t.original_offset + t.original_size;
      if (!EncodeSBranch(br_from, br_to, br_back, true)) continue;
      std::memcpy(t.bytes.data() + t.bytes.size() - 4, br_back, 4);

      uint8_t br_fwd[4];
      if (!EncodeSBranch(t.original_offset, tp, br_fwd, true)) continue;
      std::memcpy(text + t.original_offset, br_fwd, 4);
      for (uint32_t i = 4; i < t.original_size; i += 4) {
        uint8_t nop[4];
        EncodeSNop(nop);
        std::memcpy(text + t.original_offset + i, nop, 4);
      }
    }

    MallocBuffer new_buf = GrowElfWithTrampolines(buf.data(), elf_size, elf_info,
                                                   deferred);
    if (!new_buf) return AMD_COMGR_STATUS_ERROR;

    size_t tramp_total = 0;
    for (auto &t : deferred) tramp_total += t.bytes.size();

    if (!AddTrampolineSymbols(new_buf, deferred,
                              elf_info.text_size,
                              elf_info.text_section_idx))
      return AMD_COMGR_STATUS_ERROR;
    PatchDebugRanges(new_buf.data, new_buf.size, elf_info.text_addr,
                     elf_info.text_size, tramp_total);
    PatchDebugInfo(new_buf.data, new_buf.size, elf_info.text_addr,
                   elf_info.text_size, tramp_total);
    PatchDebugFrame(new_buf.data, new_buf.size, elf_info.text_addr,
                    elf_info.text_size, tramp_total);
    if (!PatchDebugLine(new_buf, deferred,
                        elf_info.text_size, elf_info.text_addr))
      return AMD_COMGR_STATUS_ERROR;

    *out_size = new_buf.size;
    *out_data = new_buf.release();
    result->trampolines_added = static_cast<uint32_t>(deferred.size());
  } else {
    MallocBuffer out(elf_size);
    if (!out) return AMD_COMGR_STATUS_ERROR;
    std::memcpy(out.data, buf.data(), elf_size);
    *out_data = out.release();
    *out_size = elf_size;
  }

  if (!scratch_patches.empty()) {
    ElfInfo verify_elf_info;
    const uint8_t *verify_elf = static_cast<const uint8_t *>(*out_data);
    if (ParseElfInfo(verify_elf, *out_size, verify_elf_info) &&
        verify_elf_info.text_size > 0) {
      bool ok = VerifyPatchCorrectness(verify_elf + verify_elf_info.text_offset,
                                       verify_elf_info.text_size,
                                       llvm_state, scratch_patches);
      if (!ok) {
        HotswapLog(HotswapLogLevel::Error) << "hotswap: WARNING: post-patch verification detected "
                     "possible scratch conflicts (may be false positive from "
                     "liveness approximation)\n";
      }
    }
  }

  return AMD_COMGR_STATUS_SUCCESS;
}
