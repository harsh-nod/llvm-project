//===- comgr-hotswap.cpp - HotSwap ISA rewriting --------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Full LLVM MC-backed implementation of the HotSwap ISA rewriting APIs.
/// This file is self-contained — it does not depend on any rocm-systems
/// headers. All ELF parsing, LLVM MC setup, instruction decode/encode,
/// trampoline building, B0→A0 patching, retarget, transpile, and rewrite-rule
/// logic is implemented directly here.
///
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-internal.h"

// ── RetargetCodeObject ───────────────────────────────────────────────────────

static amd_comgr_status_t
RetargetCodeObject(const void *elf_data, size_t elf_size,
                   const std::string &source_isa,
                   const std::string &target_isa,
                   void **out_data, size_t *out_size,
                   amd_comgr_hotswap_result_t *result) {
  ElfInfo elf_info;
  const uint8_t *elf = static_cast<const uint8_t *>(elf_data);

  MallocBuffer buf(elf_size);
  if (!buf) return AMD_COMGR_STATUS_ERROR;
  std::memcpy(buf.data, elf, elf_size);

  if (!ParseElfInfo(buf.data, elf_size, elf_info) || elf_info.text_size == 0) {
    *out_size = buf.size;
    *out_data = buf.release();
    return AMD_COMGR_STATUS_SUCCESS;
  }

  LLVMState src_state = InitLLVMCached(source_isa);
  if (!src_state.valid)
    return AMD_COMGR_STATUS_ERROR;

  uint8_t *text = buf.data + elf_info.text_offset;
  std::vector<InternalDecodedInst> decoded;
  if (!DecodeTextSection(text, elf_info.text_size, src_state, decoded))
    return AMD_COMGR_STATUS_ERROR;

  std::string tgt_cpu = ExtractCPU(target_isa);
  PatchElfIsa(buf.data, elf_size, tgt_cpu);

  *out_size = buf.size;
  *out_data = buf.release();
  result->rules_matched = static_cast<uint32_t>(decoded.size());
  return AMD_COMGR_STATUS_SUCCESS;
}

// ── JSON parser for rewrite rules (using llvm::json) ─────────────────────────

static RulesFile ParseRulesString(const std::string &json) {
  RulesFile rf;
  auto parsed = llvm::json::parse(json);
  if (!parsed) {
    llvm::consumeError(parsed.takeError());
    return rf;
  }
  auto *root = parsed->getAsObject();
  if (!root) return rf;

  if (auto v = root->getInteger("version"))
    rf.version = static_cast<uint32_t>(*v);
  if (auto v = root->getString("target"))
    rf.target = v->str();

  auto *rules = root->getArray("rules");
  if (!rules) return rf;

  for (const auto &rv : *rules) {
    auto *obj = rv.getAsObject();
    if (!obj) continue;
    RewriteRule rule;
    if (auto v = obj->getString("name")) rule.name = v->str();
    if (auto v = obj->getString("match_mnemonic")) rule.match_mnemonic = v->str();
    if (auto v = obj->getString("match_kernel")) rule.match_kernel = v->str();
    if (auto v = obj->getInteger("match_offset"))
      rule.match_offset = *v;
    else
      rule.match_offset = -1;

    auto action_str = obj->getString("action");
    std::string action = action_str ? action_str->str() : "mnemonic_swap";
    if (action == "asm_replace")
      rule.action = ReplaceAction::AsmReplace;
    else if (action == "byte_replace")
      rule.action = ReplaceAction::ByteReplace;
    else
      rule.action = ReplaceAction::MnemonicSwap;

    if (auto v = obj->getString("replace_mnemonic")) rule.replace_mnemonic = v->str();
    if (auto v = obj->getBoolean("preserve_operands"))
      rule.preserve_operands = *v;
    else
      rule.preserve_operands = true;
    if (auto v = obj->getInteger("extra_vgprs")) rule.extra_vgprs = static_cast<int32_t>(*v);
    if (auto v = obj->getInteger("extra_sgprs")) rule.extra_sgprs = static_cast<int32_t>(*v);

    if (auto *asm_arr = obj->getArray("replace_asm"))
      for (const auto &a : *asm_arr)
        if (auto s = a.getAsString())
          rule.replace_asm.push_back(s->str());

    if (auto *bytes_arr = obj->getArray("replace_bytes"))
      for (const auto &b : *bytes_arr)
        if (auto v = b.getAsInteger())
          rule.replace_bytes.push_back(static_cast<uint8_t>(*v));

    rf.rules.push_back(std::move(rule));
  }
  return rf;
}

// ── RewriteWithRules ─────────────────────────────────────────────────────────

static amd_comgr_status_t
RewriteWithRules(const void *elf_data, size_t elf_size,
                 const std::string &isa_name, const std::string &rules_json,
                 void **out_data, size_t *out_size,
                 amd_comgr_hotswap_result_t *result) {
  RulesFile rules = ParseRulesString(rules_json);
  if (rules.rules.empty()) {
    MallocBuffer copy(elf_size);
    if (!copy) return AMD_COMGR_STATUS_ERROR;
    std::memcpy(copy.data, elf_data, elf_size);
    *out_size = copy.size;
    *out_data = copy.release();
    return AMD_COMGR_STATUS_SUCCESS;
  }

  MallocBuffer buf(elf_size);
  if (!buf) return AMD_COMGR_STATUS_ERROR;
  std::memcpy(buf.data, elf_data, elf_size);

  ElfInfo elf_info;
  if (!ParseElfInfo(buf.data, elf_size, elf_info) || elf_info.text_size == 0) {
    *out_size = buf.size;
    *out_data = buf.release();
    return AMD_COMGR_STATUS_SUCCESS;
  }

  LLVMState llvm_state = InitLLVMImpl(isa_name);
  if (!llvm_state.valid)
    return AMD_COMGR_STATUS_ERROR;

  uint8_t *text = buf.data + elf_info.text_offset;
  std::vector<InternalDecodedInst> decoded;
  if (!DecodeTextSection(text, elf_info.text_size, llvm_state, decoded))
    return AMD_COMGR_STATUS_ERROR;

  std::vector<Trampoline> trampolines;

  for (auto &inst : decoded) {
    for (auto &rule : rules.rules) {
      if (!MatchRule(rule, inst, elf_info)) continue;
      bool applied = false;

      switch (rule.action) {
      case ReplaceAction::MnemonicSwap:
        applied = ApplyMnemonicSwap(rule, inst, text, llvm_state);
        break;
      case ReplaceAction::ByteReplace:
        applied = ApplyByteReplace(rule, inst.offset, inst.size, text,
                                   elf_info.text_size);
        break;
      case ReplaceAction::AsmReplace: {
        uint64_t tramp_offset = elf_info.text_size;
        for (auto &t : trampolines)
          tramp_offset += t.bytes.size();
        Trampoline tramp =
            BuildTrampoline(rule.replace_asm, inst.offset, inst.size,
                            tramp_offset, llvm_state.cpu, llvm_state);
        if (tramp.bytes.empty()) break;
        uint8_t branch_bytes[4];
        bool is_gfx12 = !(llvm_state.cpu.find("gfx9") == 0 || llvm_state.cpu.find("gfx10") == 0);
        if (!EncodeSBranch(inst.offset, tramp_offset, branch_bytes, is_gfx12))
          break;
        std::memcpy(text + inst.offset, branch_bytes, 4);
        for (uint32_t pad = 4; pad < inst.size; pad += 4) {
          uint8_t nop[4];
          EncodeSNop(nop);
          std::memcpy(text + inst.offset + pad, nop, 4);
        }
        trampolines.push_back(std::move(tramp));
        applied = true;
        break;
      }
      }

      if (applied) {
        ++result->rules_matched;
        if (rule.extra_vgprs > 0 || rule.extra_sgprs > 0) {
          std::string kernel = FindKernelAtOffset(elf_info, inst.offset);
          if (!kernel.empty())
            UpdateKernelDescriptor(buf.data, elf_size, elf_info, kernel,
                                  rule.extra_vgprs, rule.extra_sgprs);
        }
        break;
      }
    }
  }

  if (!trampolines.empty()) {
    MallocBuffer new_buf = GrowElfWithTrampolines(buf.data, elf_size, elf_info,
                                                   trampolines);
    if (!new_buf)
      return AMD_COMGR_STATUS_ERROR;
    *out_size = new_buf.size;
    *out_data = new_buf.release();
    result->trampolines_added = static_cast<uint32_t>(trampolines.size());
  } else {
    *out_size = buf.size;
    *out_data = buf.release();
  }

  return AMD_COMGR_STATUS_SUCCESS;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Public API entry points
// ═══════════════════════════════════════════════════════════════════════════════

amd_comgr_status_t AMD_COMGR_API amd_comgr_hotswap_rewrite(
    const void *elf_data, size_t elf_size, const char *source_isa,
    const char *target_isa, uint32_t flags, const char *rules_json,
    void **out_elf, size_t *out_elf_size,
    amd_comgr_hotswap_result_t *result) {
  if (!elf_data || elf_size == 0 || !out_elf || !out_elf_size || !result)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  std::memset(result, 0, sizeof(*result));

  const void *current_elf = elf_data;
  size_t current_size = elf_size;
  MallocBuffer owned;

  // B0→A0 patching
  if (flags & AMD_COMGR_HOTSWAP_FLAG_B0_TO_A0) {
    void *new_data = nullptr;
    size_t new_size = 0;
    amd_comgr_status_t status = RetargetCodeObjectB0A0Grow(
        current_elf, current_size, &new_data, &new_size, result);
    if (status != AMD_COMGR_STATUS_SUCCESS)
      return status;
    owned = MallocBuffer();
    owned.data = static_cast<uint8_t *>(new_data);
    owned.size = new_size;
    current_elf = owned.data;
    current_size = owned.size;
  }

  // Retarget
  if (flags & AMD_COMGR_HOTSWAP_FLAG_RETARGET) {
    if (!source_isa || !target_isa)
      return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
    void *new_data = nullptr;
    size_t new_size = 0;
    amd_comgr_status_t status =
        RetargetCodeObject(current_elf, current_size, std::string(source_isa),
                           std::string(target_isa), &new_data, &new_size,
                           result);
    if (status != AMD_COMGR_STATUS_SUCCESS)
      return status;
    owned = MallocBuffer();
    owned.data = static_cast<uint8_t *>(new_data);
    owned.size = new_size;
    current_elf = owned.data;
    current_size = owned.size;
  }

  // Transpile
  if (flags & AMD_COMGR_HOTSWAP_FLAG_TRANSPILE) {
    if (!source_isa || !target_isa)
      return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
    void *new_data = nullptr;
    size_t new_size = 0;
    amd_comgr_status_t status = TranspileCodeObject(
        current_elf, current_size, std::string(source_isa),
        std::string(target_isa), &new_data, &new_size, result);
    if (status != AMD_COMGR_STATUS_SUCCESS)
      return status;
    owned = MallocBuffer();
    owned.data = static_cast<uint8_t *>(new_data);
    owned.size = new_size;
    current_elf = owned.data;
    current_size = owned.size;
  }

  // Rewrite rules
  if (flags & AMD_COMGR_HOTSWAP_FLAG_REWRITE_RULES) {
    if (!rules_json || !target_isa)
      return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
    void *new_data = nullptr;
    size_t new_size = 0;
    amd_comgr_status_t status = RewriteWithRules(
        current_elf, current_size, std::string(target_isa),
        std::string(rules_json), &new_data, &new_size, result);
    if (status != AMD_COMGR_STATUS_SUCCESS)
      return status;
    owned = MallocBuffer();
    owned.data = static_cast<uint8_t *>(new_data);
    owned.size = new_size;
    current_elf = owned.data;
    current_size = owned.size;
  }

  // If no operations were performed, return a copy of the input
  if (!owned) {
    owned = MallocBuffer(elf_size);
    if (!owned) return AMD_COMGR_STATUS_ERROR;
    std::memcpy(owned.data, elf_data, elf_size);
  }

  *out_elf_size = owned.size;
  *out_elf = owned.release();
  result->status = 0;
  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_COMGR_API amd_comgr_hotswap_needs_transpile(
    const char *source_isa, const char *target_isa, bool *needs_transpile) {
  if (!source_isa || !target_isa || !needs_transpile)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  *needs_transpile =
      NeedsTranspileImpl(std::string(source_isa), std::string(target_isa));
  return AMD_COMGR_STATUS_SUCCESS;
}

// ── Test-only entry points for dataflow analysis ─────────────────────────────
// Test-only entry points — exported via exportmap for the ROCR test suite.
// These bypass normal input validation and should not be called in production.

extern "C" __attribute__((visibility("default")))
int amd_comgr_test_defuse(const char *asm_text, const char *cpu,
                          int *out_defs, int *out_def_count,
                          int *out_uses, int *out_use_count,
                          int max_regs) {
  if (!asm_text || !cpu || !out_defs || !out_def_count ||
      !out_uses || !out_use_count || max_regs <= 0)
    return -1;

  std::string isa = std::string("amdgcn-amd-amdhsa--") + cpu;
  LLVMState state = InitLLVMCached(isa);
  if (!state.valid) return -1;

  auto bytes = AssembleSingleInst(std::string(asm_text), state);
  if (bytes.empty()) return -1;

  std::vector<InternalDecodedInst> decoded;
  if (!DecodeTextSection(bytes.data(), bytes.size(), state, decoded))
    return -1;
  if (decoded.empty()) return -1;

  RegDefUse du = GetInstRegDefUse(decoded[0].inst, *state.MCII, *state.MRI);

  int dc = 0;
  for (int d = du.defs.find_first(); d != -1; d = du.defs.find_next(d)) {
    if (dc < max_regs) out_defs[dc] = d;
    dc++;
  }
  *out_def_count = dc;

  int uc = 0;
  for (int u = du.uses.find_first(); u != -1; u = du.uses.find_next(u)) {
    if (uc < max_regs) out_uses[uc] = u;
    uc++;
  }
  *out_use_count = uc;

  return dc;
}

extern "C" __attribute__((visibility("default")))
int amd_comgr_test_cfg(const char **asm_lines, int num_lines, const char *cpu,
                       uint64_t *out_bb_starts, int *out_bb_succ_counts,
                       int max_blocks) {
  if (!asm_lines || num_lines <= 0 || !cpu || max_blocks <= 0)
    return -1;

  std::string isa = std::string("amdgcn-amd-amdhsa--") + cpu;
  LLVMState state = InitLLVMCached(isa);
  if (!state.valid) return -1;

  std::vector<uint8_t> all_bytes;
  for (int i = 0; i < num_lines; i++) {
    auto bytes = AssembleSingleInst(std::string(asm_lines[i]), state);
    if (bytes.empty()) return -1;
    all_bytes.insert(all_bytes.end(), bytes.begin(), bytes.end());
  }

  std::vector<InternalDecodedInst> decoded;
  if (!DecodeTextSection(all_bytes.data(), all_bytes.size(), state, decoded))
    return -1;

  CFG cfg = BuildCFG(decoded, *state.MCII);
  int n = static_cast<int>(cfg.blocks.size());

  for (int i = 0; i < n && i < max_blocks; i++) {
    if (out_bb_starts) out_bb_starts[i] = cfg.blocks[i].start_offset;
    if (out_bb_succ_counts)
      out_bb_succ_counts[i] = static_cast<int>(cfg.blocks[i].successors.size());
  }

  return n;
}

extern "C" __attribute__((visibility("default")))
int amd_comgr_test_liveness(const char **asm_lines, int num_lines,
                            const char *cpu, int inst_index,
                            int *out_live, int max_regs) {
  if (!asm_lines || num_lines <= 0 || !cpu || inst_index < 0 || max_regs <= 0)
    return -1;

  std::string isa = std::string("amdgcn-amd-amdhsa--") + cpu;
  LLVMState state = InitLLVMCached(isa);
  if (!state.valid) return -1;

  std::vector<uint8_t> all_bytes;
  for (int i = 0; i < num_lines; i++) {
    auto bytes = AssembleSingleInst(std::string(asm_lines[i]), state);
    if (bytes.empty()) return -1;
    all_bytes.insert(all_bytes.end(), bytes.begin(), bytes.end());
  }

  std::vector<InternalDecodedInst> decoded;
  if (!DecodeTextSection(all_bytes.data(), all_bytes.size(), state, decoded))
    return -1;

  if (inst_index >= static_cast<int>(decoded.size())) return -1;

  CFG cfg = BuildCFG(decoded, *state.MCII);
  LivenessInfo liveness =
      ComputeLiveness(decoded, cfg, *state.MCII, *state.MRI);

  const auto &live_set = liveness.live_before[inst_index];
  int count = 0;
  for (int v = live_set.find_first(); v != -1; v = live_set.find_next(v)) {
    if (count < max_regs) out_live[count] = v;
    count++;
  }

  return count;
}

extern "C" __attribute__((visibility("default")))
int amd_comgr_test_scratch_alloc(const int *live_vgprs, int num_live,
                                 int kd_allocated_vgprs) {
  if (num_live < 0 || kd_allocated_vgprs <= 0) return -1;

  llvm::BitVector live(256);
  for (int i = 0; i < num_live; i++)
    live.set(live_vgprs[i]);

  ScratchAllocator alloc(live, kd_allocated_vgprs);
  return alloc.Alloc();
}

extern "C" __attribute__((visibility("default")))
int amd_comgr_test_debug_symbols(const void *elf_data, size_t elf_size,
                                  char *out_names, int max_names) {
  ElfInfo info;
  const uint8_t *elf = static_cast<const uint8_t *>(elf_data);
  if (!ParseElfInfo(elf, elf_size, info))
    return 0;
  int count = 0;
  size_t name_pos = 0;
  for (auto &sym : info.symbols) {
    if (sym.name.find("__hotswap_tramp_") == 0) {
      if (out_names && count < max_names) {
        size_t len = sym.name.size() + 1;
        std::memcpy(out_names + name_pos, sym.name.c_str(), len);
        name_pos += len;
      }
      count++;
    }
  }
  return count;
}
