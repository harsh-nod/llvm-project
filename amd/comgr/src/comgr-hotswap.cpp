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

#include "amd_comgr/amd_comgr.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "llvm/Config/llvm-config.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#if LLVM_VERSION_MAJOR > 13
#include "llvm/MC/TargetRegistry.h"
#else
#include "llvm/Support/TargetRegistry.h"
#endif

// ═══════════════════════════════════════════════════════════════════════════════
// Anonymous namespace: all internal helpers
// ═══════════════════════════════════════════════════════════════════════════════
namespace {

#include "comgr-hotswap-elf.h"
#include "comgr-hotswap-dwarf.h"
#include "comgr-hotswap-llvm.h"
#include "comgr-hotswap-liveness.h"
#include "comgr-hotswap-b0a0.h"
#include "comgr-hotswap-transpiler.h"

// ── RetargetCodeObject ───────────────────────────────────────────────────────

static amd_comgr_status_t
RetargetCodeObject(const void *elf_data, size_t elf_size,
                   const std::string &source_isa,
                   const std::string &target_isa,
                   void **out_data, size_t *out_size,
                   amd_comgr_hotswap_result_t *result) {
  ElfInfo elf_info;
  const uint8_t *elf = static_cast<const uint8_t *>(elf_data);

  uint8_t *buf = static_cast<uint8_t *>(std::malloc(elf_size));
  if (!buf) return AMD_COMGR_STATUS_ERROR;
  std::memcpy(buf, elf, elf_size);

  if (!ParseElfInfo(buf, elf_size, elf_info) || elf_info.text_size == 0) {
    *out_data = buf;
    *out_size = elf_size;
    return AMD_COMGR_STATUS_SUCCESS;
  }

  LLVMState &src_state = InitLLVMCached(source_isa);
  if (!src_state.valid) {
    std::free(buf);
    return AMD_COMGR_STATUS_ERROR;
  }

  uint8_t *text = buf + elf_info.text_offset;
  std::vector<InternalDecodedInst> decoded;
  if (!DecodeTextSection(text, elf_info.text_size, src_state, decoded)) {
    std::free(buf);
    return AMD_COMGR_STATUS_ERROR;
  }

  std::string tgt_cpu = ExtractCPU(target_isa);
  PatchElfIsa(buf, elf_size, tgt_cpu);

  *out_data = buf;
  *out_size = elf_size;
  result->rules_matched = static_cast<uint32_t>(decoded.size());
  return AMD_COMGR_STATUS_SUCCESS;
}

// ── Minimal JSON parser for rewrite rules ────────────────────────────────────

enum class JsonType { Null, Bool, Int, String, Array, Object };

struct JsonValue;
using JsonArray = std::vector<JsonValue>;
using JsonObject = std::vector<std::pair<std::string, JsonValue>>;

struct JsonValue {
  JsonType type = JsonType::Null;
  bool bool_val = false;
  int64_t int_val = 0;
  std::string str_val;
  JsonArray arr_val;
  JsonObject obj_val;

  const JsonValue *get(const std::string &key) const {
    if (type != JsonType::Object) return nullptr;
    for (auto &kv : obj_val)
      if (kv.first == key) return &kv.second;
    return nullptr;
  }
  std::string get_string(const std::string &key,
                         const std::string &def = "") const {
    auto *v = get(key);
    return (v && v->type == JsonType::String) ? v->str_val : def;
  }
  int64_t get_int(const std::string &key, int64_t def = 0) const {
    auto *v = get(key);
    return (v && v->type == JsonType::Int) ? v->int_val : def;
  }
  bool get_bool(const std::string &key, bool def = false) const {
    auto *v = get(key);
    return (v && v->type == JsonType::Bool) ? v->bool_val : def;
  }
};

class JsonParser {
public:
  explicit JsonParser(const std::string &input) : src_(input), pos_(0) {}
  bool Parse(JsonValue &out) {
    SkipWS();
    if (!ParseValue(out)) return false;
    SkipWS();
    return true;
  }

private:
  const std::string &src_;
  size_t pos_;
  char Peek() const { return pos_ < src_.size() ? src_[pos_] : '\0'; }
  char Next() { return pos_ < src_.size() ? src_[pos_++] : '\0'; }
  void SkipWS() {
    while (pos_ < src_.size() &&
           (src_[pos_] == ' ' || src_[pos_] == '\t' || src_[pos_] == '\n' ||
            src_[pos_] == '\r'))
      ++pos_;
  }
  bool ParseValue(JsonValue &out) {
    SkipWS();
    char c = Peek();
    if (c == '"') return ParseString(out);
    if (c == '{') return ParseObject(out);
    if (c == '[') return ParseArray(out);
    if (c == 't' || c == 'f') return ParseBool(out);
    if (c == 'n') return ParseNull(out);
    if (c == '-' || (c >= '0' && c <= '9')) return ParseInt(out);
    return false;
  }
  bool ParseString(JsonValue &out) {
    if (Next() != '"') return false;
    std::string s;
    while (true) {
      if (pos_ >= src_.size()) return false;
      char c = Next();
      if (c == '"') break;
      if (c == '\\') {
        if (pos_ >= src_.size()) return false;
        char esc = Next();
        switch (esc) {
        case '"':  s += '"';  break;
        case '\\': s += '\\'; break;
        case '/':  s += '/';  break;
        case 'n':  s += '\n'; break;
        case 't':  s += '\t'; break;
        default:   s += esc;  break;
        }
      } else {
        s += c;
      }
    }
    out.type = JsonType::String;
    out.str_val = std::move(s);
    return true;
  }
  bool ParseInt(JsonValue &out) {
    size_t start = pos_;
    if (Peek() == '-') Next();
    if (Peek() == '0') {
      Next();
      if (Peek() == 'x' || Peek() == 'X') {
        Next();
        while (pos_ < src_.size() && std::isxdigit(src_[pos_])) Next();
      }
    } else {
      while (pos_ < src_.size() && std::isdigit(src_[pos_])) Next();
    }
    out.type = JsonType::Int;
    try {
      out.int_val = std::stoll(src_.substr(start, pos_ - start), nullptr, 0);
    } catch (...) {
      out.int_val = 0;
    }
    return true;
  }
  bool ParseBool(JsonValue &out) {
    if (src_.compare(pos_, 4, "true") == 0) {
      pos_ += 4;
      out.type = JsonType::Bool;
      out.bool_val = true;
      return true;
    }
    if (src_.compare(pos_, 5, "false") == 0) {
      pos_ += 5;
      out.type = JsonType::Bool;
      out.bool_val = false;
      return true;
    }
    return false;
  }
  bool ParseNull(JsonValue &out) {
    if (src_.compare(pos_, 4, "null") == 0) {
      pos_ += 4;
      out.type = JsonType::Null;
      return true;
    }
    return false;
  }
  bool ParseArray(JsonValue &out) {
    if (Next() != '[') return false;
    out.type = JsonType::Array;
    SkipWS();
    if (Peek() == ']') { Next(); return true; }
    while (true) {
      JsonValue elem;
      if (!ParseValue(elem)) return false;
      out.arr_val.push_back(std::move(elem));
      SkipWS();
      if (Peek() == ']') { Next(); return true; }
      if (Peek() != ',') return false;
      Next();
    }
  }
  bool ParseObject(JsonValue &out) {
    if (Next() != '{') return false;
    out.type = JsonType::Object;
    SkipWS();
    if (Peek() == '}') { Next(); return true; }
    while (true) {
      JsonValue key;
      if (!ParseString(key)) return false;
      SkipWS();
      if (Peek() != ':') return false;
      Next();
      JsonValue val;
      if (!ParseValue(val)) return false;
      out.obj_val.push_back({key.str_val, std::move(val)});
      SkipWS();
      if (Peek() == '}') { Next(); return true; }
      if (Peek() != ',') return false;
      Next();
    }
  }
};

static RulesFile ParseRulesString(const std::string &json) {
  RulesFile rf;
  JsonValue root;
  JsonParser parser(json);
  if (!parser.Parse(root) || root.type != JsonType::Object) return rf;

  rf.version = static_cast<uint32_t>(root.get_int("version", 0));
  rf.target = root.get_string("target");

  auto *rules = root.get("rules");
  if (!rules || rules->type != JsonType::Array) return rf;

  for (auto &rv : rules->arr_val) {
    if (rv.type != JsonType::Object) continue;
    RewriteRule rule;
    rule.name = rv.get_string("name");
    rule.match_mnemonic = rv.get_string("match_mnemonic");
    rule.match_kernel = rv.get_string("match_kernel");
    rule.match_offset = rv.get_int("match_offset", -1);

    std::string action = rv.get_string("action", "mnemonic_swap");
    if (action == "asm_replace")
      rule.action = ReplaceAction::AsmReplace;
    else if (action == "byte_replace")
      rule.action = ReplaceAction::ByteReplace;
    else
      rule.action = ReplaceAction::MnemonicSwap;

    rule.replace_mnemonic = rv.get_string("replace_mnemonic");
    rule.preserve_operands = rv.get_bool("preserve_operands", true);
    rule.extra_vgprs = static_cast<int32_t>(rv.get_int("extra_vgprs", 0));
    rule.extra_sgprs = static_cast<int32_t>(rv.get_int("extra_sgprs", 0));

    auto *asm_arr = rv.get("replace_asm");
    if (asm_arr && asm_arr->type == JsonType::Array)
      for (auto &a : asm_arr->arr_val)
        if (a.type == JsonType::String)
          rule.replace_asm.push_back(a.str_val);

    auto *bytes_arr = rv.get("replace_bytes");
    if (bytes_arr && bytes_arr->type == JsonType::Array)
      for (auto &b : bytes_arr->arr_val)
        if (b.type == JsonType::Int)
          rule.replace_bytes.push_back(static_cast<uint8_t>(b.int_val));

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
    void *copy = std::malloc(elf_size);
    if (!copy) return AMD_COMGR_STATUS_ERROR;
    std::memcpy(copy, elf_data, elf_size);
    *out_data = copy;
    *out_size = elf_size;
    return AMD_COMGR_STATUS_SUCCESS;
  }

  uint8_t *elf = static_cast<uint8_t *>(std::malloc(elf_size));
  if (!elf) return AMD_COMGR_STATUS_ERROR;
  std::memcpy(elf, elf_data, elf_size);

  ElfInfo elf_info;
  if (!ParseElfInfo(elf, elf_size, elf_info) || elf_info.text_size == 0) {
    *out_data = elf;
    *out_size = elf_size;
    return AMD_COMGR_STATUS_SUCCESS;
  }

  LLVMState llvm_state = InitLLVMImpl(isa_name);
  if (!llvm_state.valid) {
    std::free(elf);
    return AMD_COMGR_STATUS_ERROR;
  }

  uint8_t *text = elf + elf_info.text_offset;
  std::vector<InternalDecodedInst> decoded;
  if (!DecodeTextSection(text, elf_info.text_size, llvm_state, decoded)) {
    std::free(elf);
    return AMD_COMGR_STATUS_ERROR;
  }

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
            UpdateKernelDescriptor(elf, elf_size, elf_info, kernel,
                                  rule.extra_vgprs, rule.extra_sgprs);
        }
        break;
      }
    }
  }

  if (!trampolines.empty()) {
    size_t new_elf_size = 0;
    uint8_t *new_elf = GrowElfWithTrampolines(elf, elf_size, elf_info,
                                              trampolines, &new_elf_size);
    if (!new_elf) {
      std::free(elf);
      return AMD_COMGR_STATUS_ERROR;
    }
    std::free(elf);
    *out_data = new_elf;
    *out_size = new_elf_size;
    result->trampolines_added = static_cast<uint32_t>(trampolines.size());
  } else {
    *out_data = elf;
    *out_size = elf_size;
  }

  return AMD_COMGR_STATUS_SUCCESS;
}

} // anonymous namespace

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
  void *allocated_elf = nullptr;

  auto cleanup = [&]() {
    if (allocated_elf) std::free(allocated_elf);
  };

  // B0→A0 patching
  if (flags & AMD_COMGR_HOTSWAP_FLAG_B0_TO_A0) {
    void *new_data = nullptr;
    size_t new_size = 0;
    amd_comgr_status_t status = RetargetCodeObjectB0A0Grow(
        current_elf, current_size, &new_data, &new_size, result);
    if (status != AMD_COMGR_STATUS_SUCCESS) {
      cleanup();
      return status;
    }
    if (allocated_elf) std::free(allocated_elf);
    allocated_elf = new_data;
    current_elf = new_data;
    current_size = new_size;
  }

  // Retarget
  if (flags & AMD_COMGR_HOTSWAP_FLAG_RETARGET) {
    if (!source_isa || !target_isa) {
      cleanup();
      return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
    }
    void *new_data = nullptr;
    size_t new_size = 0;
    amd_comgr_status_t status =
        RetargetCodeObject(current_elf, current_size, std::string(source_isa),
                           std::string(target_isa), &new_data, &new_size,
                           result);
    if (status != AMD_COMGR_STATUS_SUCCESS) {
      cleanup();
      return status;
    }
    if (allocated_elf) std::free(allocated_elf);
    allocated_elf = new_data;
    current_elf = new_data;
    current_size = new_size;
  }

  // Transpile
  if (flags & AMD_COMGR_HOTSWAP_FLAG_TRANSPILE) {
    if (!source_isa || !target_isa) {
      cleanup();
      return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
    }
    void *new_data = nullptr;
    size_t new_size = 0;
    amd_comgr_status_t status = TranspileCodeObject(
        current_elf, current_size, std::string(source_isa),
        std::string(target_isa), &new_data, &new_size, result);
    if (status != AMD_COMGR_STATUS_SUCCESS) {
      cleanup();
      return status;
    }
    if (allocated_elf) std::free(allocated_elf);
    allocated_elf = new_data;
    current_elf = new_data;
    current_size = new_size;
  }

  // Rewrite rules
  if (flags & AMD_COMGR_HOTSWAP_FLAG_REWRITE_RULES) {
    if (!rules_json || !target_isa) {
      cleanup();
      return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
    }
    void *new_data = nullptr;
    size_t new_size = 0;
    amd_comgr_status_t status = RewriteWithRules(
        current_elf, current_size, std::string(target_isa),
        std::string(rules_json), &new_data, &new_size, result);
    if (status != AMD_COMGR_STATUS_SUCCESS) {
      cleanup();
      return status;
    }
    if (allocated_elf) std::free(allocated_elf);
    allocated_elf = new_data;
    current_elf = new_data;
    current_size = new_size;
  }

  // If no operations were performed, return a copy of the input
  if (!allocated_elf) {
    allocated_elf = std::malloc(elf_size);
    if (!allocated_elf) return AMD_COMGR_STATUS_ERROR;
    std::memcpy(allocated_elf, elf_data, elf_size);
    current_size = elf_size;
  }

  *out_elf = allocated_elf;
  *out_elf_size = current_size;
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

extern "C" __attribute__((visibility("default")))
int amd_comgr_test_defuse(const char *asm_text, const char *cpu,
                          int *out_defs, int *out_def_count,
                          int *out_uses, int *out_use_count,
                          int max_regs) {
  if (!asm_text || !cpu || !out_defs || !out_def_count ||
      !out_uses || !out_use_count || max_regs <= 0)
    return -1;

  std::string isa = std::string("amdgcn-amd-amdhsa--") + cpu;
  LLVMState &state = InitLLVMCached(isa);
  if (!state.valid) return -1;

  auto bytes = AssembleSingleInst(std::string(asm_text), state);
  if (bytes.empty()) return -1;

  std::vector<InternalDecodedInst> decoded;
  if (!DecodeTextSection(bytes.data(), bytes.size(), state, decoded))
    return -1;
  if (decoded.empty()) return -1;

  RegDefUse du = GetInstRegDefUse(decoded[0].inst, *state.MCII, *state.MRI);

  int dc = 0;
  for (int d : du.defs) {
    if (dc < max_regs) out_defs[dc] = d;
    dc++;
  }
  *out_def_count = dc;

  int uc = 0;
  for (int u : du.uses) {
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
  LLVMState &state = InitLLVMCached(isa);
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

  CFG cfg = BuildCFG(decoded);
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
  LLVMState &state = InitLLVMCached(isa);
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

  CFG cfg = BuildCFG(decoded);
  LivenessInfo liveness =
      ComputeLiveness(decoded, cfg, *state.MCII, *state.MRI);

  const auto &live_set = liveness.live_before[inst_index];
  int count = 0;
  for (int v : live_set) {
    if (count < max_regs) out_live[count] = v;
    count++;
  }

  return count;
}

extern "C" __attribute__((visibility("default")))
int amd_comgr_test_scratch_alloc(const int *live_vgprs, int num_live,
                                 int kd_allocated_vgprs) {
  if (num_live < 0 || kd_allocated_vgprs <= 0) return -1;

  std::set<int> live;
  for (int i = 0; i < num_live; i++)
    live.insert(live_vgprs[i]);

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
