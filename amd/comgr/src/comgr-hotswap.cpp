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

// ── ELF types ────────────────────────────────────────────────────────────────

struct ElfSection {
  uint32_t name_idx;
  std::string name;
  uint32_t type;
  uint64_t offset;
  uint64_t size;
  uint64_t addr;
};

struct ElfSymbol {
  std::string name;
  uint64_t value;
  uint64_t size;
  uint8_t info;
  uint16_t shndx;
};

struct ElfInfo {
  std::vector<ElfSection> sections;
  std::vector<ElfSymbol> symbols;
  int text_section_idx = -1;
  int text_idx = -1;
  uint64_t text_offset = 0;
  uint64_t text_size = 0;
  uint64_t text_addr = 0;
};

// ── Trampoline ───────────────────────────────────────────────────────────────

struct Trampoline {
  uint64_t original_offset;
  uint32_t original_size;
  std::vector<uint8_t> bytes;
};

// ── NOP sled ─────────────────────────────────────────────────────────────────

struct NopSled {
  uint64_t start;
  uint64_t end;
  uint64_t write_pos;
};

// ── Rewrite-rule types ───────────────────────────────────────────────────────

struct OperandMatch {
  enum class Kind { Wildcard, Immediate, RegClass };
  Kind kind = Kind::Wildcard;
  int64_t imm_value = 0;
  std::string reg_class;
};

enum class ReplaceAction { MnemonicSwap, AsmReplace, ByteReplace };

struct RewriteRule {
  std::string name;
  std::string match_mnemonic;
  std::vector<OperandMatch> operands;
  std::string match_kernel;
  int64_t match_offset = -1;
  ReplaceAction action = ReplaceAction::MnemonicSwap;
  std::string replace_mnemonic;
  bool preserve_operands = true;
  std::vector<std::string> replace_asm;
  std::vector<uint8_t> replace_bytes;
  int32_t extra_vgprs = 0;
  int32_t extra_sgprs = 0;
};

struct RulesFile {
  uint32_t version = 0;
  std::string target;
  std::vector<RewriteRule> rules;
};

// ── WMMA hazard classification ───────────────────────────────────────────────

struct WmmaNopReq { int b0_nops; int a0_nops; };
struct WmmaHazard {
  size_t wmma_idx;
  size_t valu_idx;
  int existing_nops;
  int needed_nops;
  int deficit;
};

// ── s_branch / s_nop encoding ────────────────────────────────────────────────

static constexpr uint32_t S_BRANCH_GFX9  = 0xBF820000u;
static constexpr uint32_t S_BRANCH_GFX12 = 0xBFA00000u;
static constexpr uint32_t S_NOP_OPCODE   = 0xBF800000u;

static bool EncodeSBranch(uint64_t from_offset, uint64_t to_offset,
                          uint8_t out_bytes[4], bool gfx12 = false) {
  int64_t byte_delta = static_cast<int64_t>(to_offset) -
                       static_cast<int64_t>(from_offset) - 4;
  if (byte_delta % 4 != 0) return false;
  int64_t dword_offset = byte_delta / 4;
  if (dword_offset < -32768 || dword_offset > 32767) return false;
  uint32_t opcode = gfx12 ? S_BRANCH_GFX12 : S_BRANCH_GFX9;
  uint32_t encoded = opcode | (static_cast<uint16_t>(dword_offset) & 0xFFFF);
  std::memcpy(out_bytes, &encoded, 4);
  return true;
}

static void EncodeSNop(uint8_t out_bytes[4]) {
  uint32_t encoded = S_NOP_OPCODE;
  std::memcpy(out_bytes, &encoded, 4);
}

// ── ExtractCPU ───────────────────────────────────────────────────────────────

static std::string ExtractCPU(const std::string &isa_name) {
  size_t pos = isa_name.rfind("gfx");
  if (pos != std::string::npos) {
    std::string cpu;
    for (size_t i = pos; i < isa_name.size(); ++i) {
      char c = isa_name[i];
      if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ||
          (c >= 'A' && c <= 'Z'))
        cpu += c;
      else
        break;
    }
    return cpu;
  }
  return "";
}

// ── ELF parsing ──────────────────────────────────────────────────────────────

static bool ParseElfInfo(const uint8_t *elf, size_t elf_size, ElfInfo &info) {
  if (elf_size < 64) return false;
  if (elf[0] != 0x7f || elf[1] != 'E' || elf[2] != 'L' || elf[3] != 'F')
    return false;
  if (elf[4] != 2) return false;

  uint64_t e_shoff;
  uint16_t e_shentsize, e_shnum, e_shstrndx;
  std::memcpy(&e_shoff, elf + 40, 8);
  std::memcpy(&e_shentsize, elf + 58, 2);
  std::memcpy(&e_shnum, elf + 60, 2);
  std::memcpy(&e_shstrndx, elf + 62, 2);

  if (e_shoff == 0 || e_shnum == 0) return false;
  if (e_shoff + static_cast<uint64_t>(e_shnum) * e_shentsize > elf_size)
    return false;

  const char *shstrtab = nullptr;
  uint64_t shstrtab_size = 0;
  if (e_shstrndx < e_shnum) {
    const uint8_t *sh = elf + e_shoff + e_shstrndx * e_shentsize;
    uint64_t sh_offset, sh_size;
    std::memcpy(&sh_offset, sh + 24, 8);
    std::memcpy(&sh_size, sh + 32, 8);
    if (sh_offset + sh_size <= elf_size) {
      shstrtab = reinterpret_cast<const char *>(elf + sh_offset);
      shstrtab_size = sh_size;
    }
  }

  info.sections.resize(e_shnum);
  for (uint16_t i = 0; i < e_shnum; ++i) {
    const uint8_t *sh = elf + e_shoff + i * e_shentsize;
    auto &sec = info.sections[i];
    std::memcpy(&sec.name_idx, sh, 4);
    std::memcpy(&sec.type, sh + 4, 4);
    std::memcpy(&sec.addr, sh + 16, 8);
    std::memcpy(&sec.offset, sh + 24, 8);
    std::memcpy(&sec.size, sh + 32, 8);

    if (shstrtab && sec.name_idx < shstrtab_size)
      sec.name = shstrtab + sec.name_idx;

    if (sec.name == ".text") {
      info.text_section_idx = i;
      info.text_idx = i;
      info.text_offset = sec.offset;
      info.text_size = sec.size;
      info.text_addr = sec.addr;
    }
  }

  for (uint16_t i = 0; i < e_shnum; ++i) {
    auto &sec = info.sections[i];
    if (sec.type != 2 && sec.type != 11)
      continue;

    const uint8_t *sh = elf + e_shoff + i * e_shentsize;
    uint32_t sh_link;
    std::memcpy(&sh_link, sh + 40, 4);

    const char *symstrtab = nullptr;
    uint64_t symstrtab_size = 0;
    if (sh_link < e_shnum) {
      auto &link_sec = info.sections[sh_link];
      if (link_sec.offset + link_sec.size <= elf_size) {
        symstrtab = reinterpret_cast<const char *>(elf + link_sec.offset);
        symstrtab_size = link_sec.size;
      }
    }

    size_t sym_count = sec.size / 24;
    for (size_t j = 0; j < sym_count; ++j) {
      if (sec.offset + (j + 1) * 24 > elf_size)
        break;
      const uint8_t *sym_entry = elf + sec.offset + j * 24;

      ElfSymbol sym;
      uint32_t st_name;
      std::memcpy(&st_name, sym_entry, 4);
      sym.info = sym_entry[4];
      std::memcpy(&sym.shndx, sym_entry + 6, 2);
      std::memcpy(&sym.value, sym_entry + 8, 8);
      std::memcpy(&sym.size, sym_entry + 16, 8);

      if (symstrtab && st_name < symstrtab_size)
        sym.name = symstrtab + st_name;

      info.symbols.push_back(std::move(sym));
    }
  }

  return info.text_section_idx >= 0;
}

static std::string FindKernelAtOffset(const ElfInfo &elf_info,
                                      uint64_t text_offset) {
  for (auto &sym : elf_info.symbols) {
    uint8_t sym_type = sym.info & 0xf;
    if (sym_type != 2 && sym_type != 10)
      continue;
    if (sym.shndx != static_cast<uint16_t>(elf_info.text_section_idx))
      continue;
    uint64_t sym_start = sym.value;
    uint64_t sym_end = sym.value + sym.size;
    if (text_offset >= sym_start && text_offset < sym_end)
      return sym.name;
  }
  return "";
}

// ── ApplyByteReplace ─────────────────────────────────────────────────────────

static bool ApplyByteReplace(const RewriteRule &rule, uint64_t inst_offset,
                             uint32_t inst_size, uint8_t *text,
                             uint64_t text_size) {
  if (rule.replace_bytes.size() > inst_size) return false;
  std::memcpy(text + inst_offset, rule.replace_bytes.data(),
              rule.replace_bytes.size());
  uint32_t remaining =
      inst_size - static_cast<uint32_t>(rule.replace_bytes.size());
  uint64_t pad_offset = inst_offset + rule.replace_bytes.size();
  while (remaining >= 4) {
    uint8_t nop[4];
    EncodeSNop(nop);
    std::memcpy(text + pad_offset, nop, 4);
    pad_offset += 4;
    remaining -= 4;
  }
  return true;
}

// ── UpdateKernelDescriptor ───────────────────────────────────────────────────

static void UpdateKernelDescriptor(uint8_t *elf_data, size_t elf_size,
                                   const ElfInfo &elf_info,
                                   const std::string &kernel_name,
                                   int32_t extra_vgprs, int32_t extra_sgprs) {
  std::string kd_name = kernel_name + ".kd";
  for (auto &sym : elf_info.symbols) {
    if (sym.name != kd_name)
      continue;
    if (sym.shndx >= elf_info.sections.size())
      continue;
    auto &sec = elf_info.sections[sym.shndx];
    uint64_t kd_file_offset = sec.offset + sym.value;
    if (kd_file_offset + 64 > elf_size)
      continue;
    uint8_t *kd = elf_data + kd_file_offset;
    uint32_t rsrc1;
    std::memcpy(&rsrc1, kd + 48, 4);
    if (extra_vgprs > 0) {
      uint32_t current = rsrc1 & 0x3F;
      uint32_t extra_granules =
          (static_cast<uint32_t>(extra_vgprs) + 3) / 4;
      uint32_t new_val = current + extra_granules;
      if (new_val > 63) new_val = 63;
      rsrc1 = (rsrc1 & ~0x3Fu) | new_val;
    }
    if (extra_sgprs > 0) {
      uint32_t current = (rsrc1 >> 6) & 0xF;
      uint32_t extra_granules =
          (static_cast<uint32_t>(extra_sgprs) + 7) / 8;
      uint32_t new_val = current + extra_granules;
      if (new_val > 15) new_val = 15;
      rsrc1 = (rsrc1 & ~(0xFu << 6)) | (new_val << 6);
    }
    std::memcpy(kd + 48, &rsrc1, 4);
    return;
  }
}

// ── NOP sled management ─────────────────────────────────────────────────────

static NopSled *FindNearestSled(std::vector<NopSled> &sleds, uint64_t offset,
                                uint64_t needed) {
  NopSled *best = nullptr;
  int64_t best_dist = INT64_MAX;
  for (auto &sled : sleds) {
    if (sled.write_pos + needed > sled.end) continue;
    int64_t dist = std::abs(static_cast<int64_t>(sled.write_pos) -
                            static_cast<int64_t>(offset));
    if (dist < 131072 && dist < best_dist) {
      best = &sled;
      best_dist = dist;
    }
  }
  return best;
}

// ── GrowElfWithTrampolines ──────────────────────────────────────────────────

static uint8_t *GrowElfWithTrampolines(const uint8_t *elf, size_t elf_size,
                                       const ElfInfo &elf_info,
                                       const std::vector<Trampoline> &trampolines,
                                       size_t *out_size) {
  size_t tramp_total = 0;
  for (auto &t : trampolines)
    tramp_total += t.bytes.size();
  if (tramp_total == 0)
    return nullptr;

  size_t new_elf_size = elf_size + tramp_total;
  uint8_t *new_elf = static_cast<uint8_t *>(std::malloc(new_elf_size));
  if (!new_elf)
    return nullptr;

  uint64_t text_end = elf_info.text_offset + elf_info.text_size;
  std::memcpy(new_elf, elf, text_end);

  uint64_t tramp_pos = text_end;
  for (auto &t : trampolines) {
    std::memcpy(new_elf + tramp_pos, t.bytes.data(), t.bytes.size());
    tramp_pos += t.bytes.size();
  }

  if (text_end < elf_size)
    std::memcpy(new_elf + tramp_pos, elf + text_end, elf_size - text_end);

  uint64_t e_shoff;
  uint16_t e_shentsize;
  std::memcpy(&e_shoff, new_elf + 40, 8);
  std::memcpy(&e_shentsize, new_elf + 58, 2);

  if (e_shoff >= text_end) {
    uint64_t new_shoff = e_shoff + tramp_total;
    std::memcpy(new_elf + 40, &new_shoff, 8);
    e_shoff = new_shoff;
  }

  uint16_t e_shnum;
  std::memcpy(&e_shnum, new_elf + 60, 2);

  for (uint16_t i = 0; i < e_shnum; ++i) {
    uint8_t *sh = new_elf + e_shoff + i * e_shentsize;
    uint64_t sh_offset;
    std::memcpy(&sh_offset, sh + 24, 8);

    if (sh_offset == elf_info.text_offset) {
      uint64_t new_text_size = elf_info.text_size + tramp_total;
      std::memcpy(sh + 32, &new_text_size, 8);
    } else if (sh_offset > elf_info.text_offset) {
      uint64_t new_offset = sh_offset + tramp_total;
      std::memcpy(sh + 24, &new_offset, 8);
    }
  }

  uint64_t e_phoff;
  uint16_t e_phentsize, e_phnum;
  std::memcpy(&e_phoff, new_elf + 32, 8);
  std::memcpy(&e_phentsize, new_elf + 54, 2);
  std::memcpy(&e_phnum, new_elf + 56, 2);

  for (uint16_t i = 0; i < e_phnum; ++i) {
    uint8_t *ph = new_elf + e_phoff + i * e_phentsize;
    uint64_t p_offset, p_filesz, p_memsz;
    std::memcpy(&p_offset, ph + 8, 8);
    std::memcpy(&p_filesz, ph + 32, 8);
    std::memcpy(&p_memsz, ph + 40, 8);

    if (p_offset <= elf_info.text_offset &&
        p_offset + p_filesz >= text_end) {
      p_filesz += tramp_total;
      p_memsz += tramp_total;
      std::memcpy(ph + 32, &p_filesz, 8);
      std::memcpy(ph + 40, &p_memsz, 8);
    } else if (p_offset > elf_info.text_offset) {
      p_offset += tramp_total;
      std::memcpy(ph + 8, &p_offset, 8);
    }
  }

  *out_size = new_elf_size;
  return new_elf;
}

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

static bool IsValuInst(const std::string &mnemonic) {
  if (mnemonic.size() < 2) return false;
  if (mnemonic[0] != 'v' || mnemonic[1] != '_') return false;
  if (mnemonic == "v_nop") return false;
  if (mnemonic.find("v_wmma") == 0) return false;
  if (mnemonic.find("v_swmmac") == 0) return false;
  return true;
}

static bool RangesOverlap(int base1, int count1, int base2, int count2) {
  if (base1 < 0 || base2 < 0) return false;
  return base1 < base2 + count2 && base2 < base1 + count1;
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
static const size_t kClusterLoadSwapsSize =
    sizeof(kClusterLoadSwaps) / sizeof(kClusterLoadSwaps[0]);

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
static const size_t kDs2AddrSwapsSize =
    sizeof(kDs2AddrSwaps) / sizeof(kDs2AddrSwaps[0]);

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
      try {
        uint32_t v = static_cast<uint32_t>(std::stoul(val, nullptr, 0));
        val = std::to_string(v * scale);
      } catch (...) {
      }
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

// ── LLVM MC Context ──────────────────────────────────────────────────────────

struct LLVMState {
  const llvm::Target *target = nullptr;
  std::unique_ptr<llvm::MCRegisterInfo> MRI;
  std::unique_ptr<const llvm::MCAsmInfo> MAI;
  std::unique_ptr<llvm::MCInstrInfo> MCII;
  std::unique_ptr<llvm::MCSubtargetInfo> STI;
  std::unique_ptr<llvm::MCContext> Ctx;
  std::unique_ptr<llvm::MCObjectFileInfo> MOFI;
  std::unique_ptr<llvm::MCDisassembler> disasm;
  std::unique_ptr<llvm::MCInstPrinter> printer;
  llvm::MCCodeEmitter *CE = nullptr;
  std::string cpu;
  bool valid = false;
};

static std::once_flag g_llvm_init_flag;

static void InitLLVMTargets() {
  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUTargetMC();
  LLVMInitializeAMDGPUAsmParser();
  LLVMInitializeAMDGPUDisassembler();
}

static LLVMState InitLLVMImpl(const std::string &isa_name) {
  std::call_once(g_llvm_init_flag, InitLLVMTargets);

  LLVMState state;
  state.cpu = ExtractCPU(isa_name);
  if (state.cpu.empty()) return state;

  std::string error;
  llvm::Triple triple("amdgcn-amd-amdhsa");

  state.target = llvm::TargetRegistry::lookupTarget("amdgcn", triple, error);
  if (!state.target) return state;

  state.MRI.reset(
      state.target->createMCRegInfo(llvm::Triple("amdgcn-amd-amdhsa")));
  if (!state.MRI) return state;

  llvm::MCTargetOptions mc_opts;
#if LLVM_VERSION_MAJOR > 9
  state.MAI.reset(state.target->createMCAsmInfo(
      *state.MRI, llvm::Triple("amdgcn-amd-amdhsa"), mc_opts));
#else
  state.MAI.reset(
      state.target->createMCAsmInfo(*state.MRI, "amdgcn-amd-amdhsa"));
#endif
  if (!state.MAI) return state;

  state.MCII.reset(state.target->createMCInstrInfo());
  if (!state.MCII) return state;

  state.STI.reset(state.target->createMCSubtargetInfo(
      llvm::Triple("amdgcn-amd-amdhsa"), state.cpu, ""));
  if (!state.STI || !state.STI->isCPUStringValid(state.cpu)) return state;

#if LLVM_VERSION_MAJOR > 12
  state.Ctx = std::make_unique<llvm::MCContext>(triple, state.MAI.get(),
                                                state.MRI.get(),
                                                state.STI.get());
  state.MOFI = std::make_unique<llvm::MCObjectFileInfo>();
  state.MOFI->initMCObjectFileInfo(*state.Ctx, false);
  state.Ctx->setObjectFileInfo(state.MOFI.get());
#else
  state.MOFI = std::make_unique<llvm::MCObjectFileInfo>();
  state.Ctx = std::make_unique<llvm::MCContext>(state.MAI.get(),
                                                state.MRI.get(),
                                                state.MOFI.get());
  state.MOFI->InitMCObjectFileInfo(triple, true, *state.Ctx);
#endif

  state.disasm.reset(
      state.target->createMCDisassembler(*state.STI, *state.Ctx));
  if (!state.disasm) return state;

  unsigned asm_variant = state.MAI->getAssemblerDialect();
  state.printer.reset(state.target->createMCInstPrinter(
      triple, asm_variant, *state.MAI, *state.MCII, *state.MRI));

#if LLVM_VERSION_MAJOR > 14
  state.CE = state.target->createMCCodeEmitter(*state.MCII, *state.Ctx);
#else
  state.CE = state.target->createMCCodeEmitter(*state.MCII, *state.MRI,
                                               *state.Ctx);
#endif

  state.valid = true;
  return state;
}

static std::mutex g_llvm_cache_mutex;
static std::map<std::string, LLVMState> g_llvm_cache;

static LLVMState &InitLLVMCached(const std::string &isa_name) {
  std::string cpu = ExtractCPU(isa_name);
  std::lock_guard<std::mutex> lock(g_llvm_cache_mutex);
  auto it = g_llvm_cache.find(cpu);
  if (it != g_llvm_cache.end())
    return it->second;
  g_llvm_cache[cpu] = InitLLVMImpl(isa_name);
  return g_llvm_cache[cpu];
}

// ── Decoded instruction with MCInst ──────────────────────────────────────────

struct InternalDecodedInst {
  uint64_t offset;
  uint32_t size;
  llvm::MCInst inst;
  std::string mnemonic;
};

// ── Instruction decode ───────────────────────────────────────────────────────

static bool DecodeTextSection(const uint8_t *text, uint64_t text_size,
                              const LLVMState &llvm_state,
                              std::vector<InternalDecodedInst> &decoded) {
  uint64_t pos = 0;
  while (pos < text_size) {
    InternalDecodedInst di;
    di.offset = pos;

    llvm::ArrayRef<uint8_t> bytes(text + pos, text_size - pos);
    uint64_t inst_size = 0;

    auto status = llvm_state.disasm->getInstruction(di.inst, inst_size, bytes,
                                                    pos, llvm::nulls());

    if (status == llvm::MCDisassembler::Fail) {
      di.size = 4;
      di.mnemonic = "<unknown>";
      pos += 4;
    } else {
      di.size = static_cast<uint32_t>(inst_size);
      if (llvm_state.printer) {
        std::string str;
        llvm::raw_string_ostream rso(str);
        llvm_state.printer->printInst(&di.inst, 0, "", *llvm_state.STI, rso);
        rso.flush();
        size_t s = str.find_first_not_of(" \t");
        if (s != std::string::npos) {
          size_t e = str.find_first_of(" \t", s);
          di.mnemonic = str.substr(s, e - s);
        }
      } else {
        di.mnemonic = llvm_state.MCII->getName(di.inst.getOpcode()).str();
      }
      pos += inst_size;
    }
    decoded.push_back(std::move(di));
  }
  return true;
}

// ── AssembleSingleInst ───────────────────────────────────────────────────────

static std::vector<uint8_t> AssembleSingleInst(const std::string &asm_str,
                                               const LLVMState &llvm_state) {
  llvm_state.Ctx->reset();

  std::string full_asm = ".text\n" + asm_str;
  llvm::StringRef asm_ref(full_asm);
  auto buf = llvm::MemoryBuffer::getMemBuffer(asm_ref, "", false);
  llvm::SourceMgr src_mgr;
  src_mgr.AddNewSourceBuffer(std::move(buf), llvm::SMLoc());

  std::string data;
  auto data_stream = std::make_unique<llvm::raw_string_ostream>(data);
  auto bos = std::make_unique<llvm::buffer_ostream>(*data_stream);

  llvm::MCTargetOptions mc_opts;
  llvm::Triple triple("amdgcn-amd-amdhsa");

#if LLVM_VERSION_MAJOR > 14
  llvm::MCCodeEmitter *ce =
      llvm_state.target->createMCCodeEmitter(*llvm_state.MCII, *llvm_state.Ctx);
#else
  llvm::MCCodeEmitter *ce = llvm_state.target->createMCCodeEmitter(
      *llvm_state.MCII, *llvm_state.MRI, *llvm_state.Ctx);
#endif
  llvm::MCAsmBackend *mab = llvm_state.target->createMCAsmBackend(
      *llvm_state.STI, *llvm_state.MRI, mc_opts);

  if (!ce || !mab) return {};

#if LLVM_VERSION_MAJOR > 20
  auto streamer = std::unique_ptr<llvm::MCStreamer>(
      llvm_state.target->createMCObjectStreamer(
          triple, *llvm_state.Ctx,
          std::unique_ptr<llvm::MCAsmBackend>(mab),
          mab->createObjectWriter(*bos),
          std::unique_ptr<llvm::MCCodeEmitter>(ce), *llvm_state.STI));
#else
  auto streamer = std::unique_ptr<llvm::MCStreamer>(
      llvm_state.target->createMCObjectStreamer(
          triple, *llvm_state.Ctx,
          std::unique_ptr<llvm::MCAsmBackend>(mab),
          mab->createObjectWriter(*bos),
          std::unique_ptr<llvm::MCCodeEmitter>(ce), *llvm_state.STI,
          mc_opts.MCRelaxAll, mc_opts.MCIncrementalLinkerCompatible, false));
#endif

  if (!streamer) return {};

  auto parser = std::unique_ptr<llvm::MCAsmParser>(
      llvm::createMCAsmParser(src_mgr, *llvm_state.Ctx, *streamer,
                              *llvm_state.MAI));
  auto tap = std::unique_ptr<llvm::MCTargetAsmParser>(
      llvm_state.target->createMCAsmParser(*llvm_state.STI, *parser,
                                           *llvm_state.MCII, mc_opts));
  if (!tap) return {};
  parser->setTargetParser(*tap);

  if (parser->Run(true)) return {};

  bos.reset();
  data_stream->flush();

  const uint8_t *elf_bytes = reinterpret_cast<const uint8_t *>(data.data());
  size_t elf_sz = data.size();
  if (elf_sz < 64) return {};

  ElfInfo asm_elf;
  if (!ParseElfInfo(elf_bytes, elf_sz, asm_elf)) return {};
  if (asm_elf.text_size == 0) return {};

  return std::vector<uint8_t>(elf_bytes + asm_elf.text_offset,
                              elf_bytes + asm_elf.text_offset +
                                  asm_elf.text_size);
}

// ── ApplyMnemonicSwap ────────────────────────────────────────────────────────

static bool ApplyMnemonicSwap(const RewriteRule &rule,
                              InternalDecodedInst &inst, uint8_t *text,
                              const LLVMState &llvm_state) {
  if (!llvm_state.printer) return false;

  std::string orig_str;
  llvm::raw_string_ostream rso(orig_str);
  llvm_state.printer->printInst(&inst.inst, 0, "", *llvm_state.STI, rso);
  rso.flush();

  size_t start = orig_str.find_first_not_of(" \t");
  if (start == std::string::npos) return false;
  size_t end = orig_str.find_first_of(" \t", start);

  std::string new_asm;
  if (end != std::string::npos)
    new_asm = rule.replace_mnemonic + orig_str.substr(end);
  else
    new_asm = rule.replace_mnemonic;

  auto bytes = AssembleSingleInst(new_asm, llvm_state);
  if (bytes.empty() || bytes.size() != inst.size) return false;

  std::memcpy(text + inst.offset, bytes.data(), inst.size);
  return true;
}

// ── BuildTrampoline ──────────────────────────────────────────────────────────

static Trampoline BuildTrampoline(const std::vector<std::string> &asm_lines,
                                  uint64_t original_offset,
                                  uint32_t original_size,
                                  uint64_t trampoline_text_offset,
                                  const std::string &cpu,
                                  const LLVMState &llvm_state) {
  Trampoline result;
  result.original_offset = original_offset;
  result.original_size = original_size;

  std::string asm_source = ".text\n";
  for (auto &line : asm_lines)
    asm_source += line + "\n";

  auto bytes = AssembleSingleInst(asm_source.substr(6), llvm_state);
  if (bytes.empty()) return result;

  result.bytes = std::move(bytes);

  uint64_t branch_back_from = trampoline_text_offset + result.bytes.size();
  uint64_t branch_back_to = original_offset + original_size;

  uint8_t branch_bytes[4];
  bool is_gfx12 = cpu.find("gfx12") == 0;
  if (!EncodeSBranch(branch_back_from, branch_back_to, branch_bytes,
                     is_gfx12)) {
    result.bytes.clear();
    return result;
  }

  result.bytes.insert(result.bytes.end(), branch_bytes, branch_bytes + 4);
  return result;
}

// ── MatchRule ────────────────────────────────────────────────────────────────

static bool MatchRule(const RewriteRule &rule, const InternalDecodedInst &inst,
                      const ElfInfo &elf_info) {
  if (!rule.match_mnemonic.empty() && rule.match_mnemonic != inst.mnemonic)
    return false;
  if (rule.match_offset >= 0 &&
      static_cast<uint64_t>(rule.match_offset) != inst.offset)
    return false;
  if (!rule.match_kernel.empty()) {
    std::string kernel = FindKernelAtOffset(elf_info, inst.offset);
    if (kernel != rule.match_kernel) return false;
  }
  if (!rule.operands.empty()) {
    if (rule.operands.size() >
        static_cast<size_t>(inst.inst.getNumOperands()))
      return false;
    for (size_t i = 0; i < rule.operands.size(); ++i) {
      auto &match = rule.operands[i];
      auto &operand = inst.inst.getOperand(i);
      switch (match.kind) {
      case OperandMatch::Kind::Wildcard:
        break;
      case OperandMatch::Kind::Immediate:
        if (!operand.isImm() || operand.getImm() != match.imm_value)
          return false;
        break;
      case OperandMatch::Kind::RegClass:
        if (!operand.isReg()) return false;
        break;
      }
    }
  }
  return true;
}

// ── VGPR introspection ───────────────────────────────────────────────────────

static int GetVgprNum(unsigned reg, const llvm::MCRegisterInfo &MRI) {
  const char *name = MRI.getName(reg);
  if (!name) return -1;
  std::string rname(name);
  if (rname.find("VGPR") == 0) {
    size_t numstart = 4;
    size_t underscore = rname.find('_', numstart);
    std::string numstr = rname.substr(
        numstart, underscore == std::string::npos ? std::string::npos
                                                  : underscore - numstart);
    try {
      return std::stoi(numstr);
    } catch (...) {
      return -1;
    }
  }
  return -1;
}

static std::pair<int, int> GetVgprRange(unsigned reg,
                                        const llvm::MCRegisterInfo &MRI) {
  const char *name = MRI.getName(reg);
  if (!name) return {-1, 0};
  std::string rname(name);
  if (rname.find("VGPR") != 0) return {-1, 0};
  int count = 1;
  for (char c : rname)
    if (c == '_') count++;
  size_t numstart = 4;
  size_t numend = rname.find_first_not_of("0123456789", numstart);
  if (numend == std::string::npos) numend = rname.size();
  std::string numstr = rname.substr(numstart, numend - numstart);
  int base = -1;
  try {
    base = std::stoi(numstr);
  } catch (...) {
    return {-1, 0};
  }
  return {base, count};
}

static std::pair<int, int>
GetOperandVgprRange(const llvm::MCInst &inst, unsigned op_idx,
                    const llvm::MCRegisterInfo &MRI) {
  if (op_idx >= inst.getNumOperands()) return {-1, 0};
  const auto &op = inst.getOperand(op_idx);
  if (!op.isReg()) return {-1, 0};
  return GetVgprRange(op.getReg(), MRI);
}

static std::string PrintInst(const InternalDecodedInst &di,
                              const LLVMState &llvm_state) {
  std::string inst_str;
  if (llvm_state.printer) {
    llvm::raw_string_ostream rso(inst_str);
    llvm_state.printer->printInst(&di.inst, 0, "", *llvm_state.STI, rso);
    rso.flush();
  }
  return inst_str;
}

// ── WMMA co-execution hazard overlap check ──────────────────────────────────

static bool CheckVgprOverlap(const llvm::MCInst &wmma_inst,
                             const llvm::MCInst &valu_inst,
                             const llvm::MCRegisterInfo &MRI) {
  std::vector<std::pair<int, int>> wmma_input_ranges;
  for (unsigned i = 0; i < wmma_inst.getNumOperands(); ++i) {
    const auto &op = wmma_inst.getOperand(i);
    if (!op.isReg()) continue;
    auto range = GetVgprRange(op.getReg(), MRI);
    if (range.first >= 0) wmma_input_ranges.push_back(range);
  }
  if (wmma_input_ranges.empty()) return false;

  if (valu_inst.getNumOperands() == 0) return false;
  const auto &dest_op = valu_inst.getOperand(0);
  if (!dest_op.isReg()) return false;
  auto valu_dest = GetVgprRange(dest_op.getReg(), MRI);
  if (valu_dest.first < 0) return false;

  for (const auto &wr : wmma_input_ranges) {
    if (RangesOverlap(wr.first, wr.second, valu_dest.first, valu_dest.second))
      return true;
  }
  return false;
}

// ── BuildNopSledMap ─────────────────────────────────────────────────────────

static std::vector<NopSled>
BuildNopSledMap(const std::vector<InternalDecodedInst> &decoded) {
  std::vector<NopSled> sleds;
  for (size_t i = 0; i < decoded.size(); ++i) {
    if (decoded[i].mnemonic == "s_endpgm") {
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
                           const LLVMState &llvm_state) {
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
      if (dj.mnemonic == "v_nop") {
        ++count;
        if (count >= req.a0_nops) break;
        continue;
      }
      if (dj.mnemonic.size() >= 2 && dj.mnemonic[0] == 's' &&
          dj.mnemonic[1] == '_') {
        if (dj.mnemonic.find("s_branch") == 0 ||
            dj.mnemonic.find("s_cbranch") == 0 ||
            dj.mnemonic == "s_endpgm" || dj.mnemonic == "s_setpc" ||
            dj.mnemonic == "s_swappc" || dj.mnemonic == "s_call")
          break;
        continue;
      }
      if (IsValuInst(dj.mnemonic)) {
        if (!CheckVgprOverlap(di.inst, dj.inst, *llvm_state.MRI)) {
          ++count;
          if (count >= req.a0_nops) break;
          continue;
        }
        if (count < req.a0_nops) {
          hazards.push_back({i, j, count, req.a0_nops, req.a0_nops - count});
          std::cerr << "hotswap: B0->A0 WMMA co-exec hazard @0x"
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

  std::cerr << "hotswap: B0->A0 WMMA co-exec validation: "
            << hazards.size() << " hazards ("
            << wmma_scanned << " WMMA instructions scanned)\n";

  return hazards;
}

// ── Per-point VGPR liveness analysis ─────────────────────────────────────────

struct RegDefUse {
  std::set<int> defs;
  std::set<int> uses;
};

static RegDefUse GetInstRegDefUse(const llvm::MCInst &inst,
                                  const llvm::MCInstrInfo &MCII,
                                  const llvm::MCRegisterInfo &MRI) {
  RegDefUse du;
  const llvm::MCInstrDesc &desc = MCII.get(inst.getOpcode());

  auto addVgprRange = [&](unsigned reg, std::set<int> &out) {
    auto [base, count] = GetVgprRange(reg, MRI);
    if (base >= 0) {
      for (int v = base; v < base + count; ++v)
        out.insert(v);
    }
  };

  unsigned num_defs = desc.getNumDefs();
  for (unsigned i = 0; i < inst.getNumOperands(); ++i) {
    const auto &op = inst.getOperand(i);
    if (!op.isReg()) continue;
    if (i < num_defs)
      addVgprRange(op.getReg(), du.defs);
    else
      addVgprRange(op.getReg(), du.uses);
  }

#if LLVM_VERSION_MAJOR >= 16
  for (llvm::MCPhysReg r : desc.implicit_defs())
    addVgprRange(r, du.defs);
  for (llvm::MCPhysReg r : desc.implicit_uses())
    addVgprRange(r, du.uses);
#else
  if (const llvm::MCPhysReg *p = desc.getImplicitDefs())
    for (; *p; ++p) addVgprRange(*p, du.defs);
  if (const llvm::MCPhysReg *p = desc.getImplicitUses())
    for (; *p; ++p) addVgprRange(*p, du.uses);
#endif

  return du;
}

// ── CFG construction ─────────────────────────────────────────────────────────

struct BasicBlock {
  uint64_t start_offset = 0;
  uint64_t end_offset = 0;
  std::vector<size_t> inst_indices;
  std::vector<int> successors;
  std::vector<int> predecessors;
};

struct CFG {
  std::vector<BasicBlock> blocks;
  std::map<uint64_t, int> offset_to_block;
};

static bool IsBranchMnemonic(const std::string &mnem) {
  return mnem == "s_branch";
}

static bool IsCBranchMnemonic(const std::string &mnem) {
  return mnem.find("s_cbranch") == 0;
}

static int64_t GetBranchImm(const llvm::MCInst &inst) {
  for (unsigned i = 0; i < inst.getNumOperands(); ++i) {
    if (inst.getOperand(i).isImm())
      return inst.getOperand(i).getImm();
  }
  return 0;
}

static CFG BuildCFG(const std::vector<InternalDecodedInst> &decoded) {
  CFG cfg;
  if (decoded.empty()) return cfg;

  std::set<uint64_t> bb_starts;
  bb_starts.insert(decoded[0].offset);

  uint64_t text_end = decoded.back().offset + decoded.back().size;

  for (size_t i = 0; i < decoded.size(); ++i) {
    const auto &di = decoded[i];
    bool is_branch = IsBranchMnemonic(di.mnemonic);
    bool is_cbranch = IsCBranchMnemonic(di.mnemonic);

    if (is_branch || is_cbranch) {
      int64_t imm = GetBranchImm(di.inst);
      uint64_t target = di.offset + 4 + (imm * 4);
      if (target < text_end)
        bb_starts.insert(target);
      if (i + 1 < decoded.size())
        bb_starts.insert(decoded[i + 1].offset);
    }
    if (di.mnemonic == "s_endpgm" ||
        di.mnemonic.find("s_setpc") == 0 ||
        di.mnemonic.find("s_swappc") == 0) {
      if (i + 1 < decoded.size())
        bb_starts.insert(decoded[i + 1].offset);
    }
  }

  std::vector<uint64_t> sorted_starts(bb_starts.begin(), bb_starts.end());
  std::sort(sorted_starts.begin(), sorted_starts.end());

  for (int i = 0; i < static_cast<int>(sorted_starts.size()); ++i)
    cfg.offset_to_block[sorted_starts[i]] = i;

  cfg.blocks.resize(sorted_starts.size());
  for (size_t i = 0; i < sorted_starts.size(); ++i)
    cfg.blocks[i].start_offset = sorted_starts[i];

  int current_block = -1;
  for (size_t i = 0; i < decoded.size(); ++i) {
    auto it = cfg.offset_to_block.find(decoded[i].offset);
    if (it != cfg.offset_to_block.end())
      current_block = it->second;
    if (current_block >= 0 &&
        current_block < static_cast<int>(cfg.blocks.size())) {
      cfg.blocks[current_block].inst_indices.push_back(i);
      cfg.blocks[current_block].end_offset =
          decoded[i].offset + decoded[i].size;
    }
  }

  for (int bi = 0; bi < static_cast<int>(cfg.blocks.size()); ++bi) {
    auto &bb = cfg.blocks[bi];
    if (bb.inst_indices.empty()) continue;

    size_t last_idx = bb.inst_indices.back();
    const auto &last = decoded[last_idx];

    if (last.mnemonic == "s_endpgm") {
      /* no successors */
    } else if (last.mnemonic.find("s_setpc") == 0 ||
               last.mnemonic.find("s_swappc") == 0) {
      /* conservative: unknown targets */
    } else if (IsBranchMnemonic(last.mnemonic) ||
               IsCBranchMnemonic(last.mnemonic)) {
      int64_t imm = GetBranchImm(last.inst);
      uint64_t target = last.offset + 4 + (imm * 4);
      auto tgt_it = cfg.offset_to_block.find(target);
      if (tgt_it != cfg.offset_to_block.end())
        bb.successors.push_back(tgt_it->second);
      if (IsCBranchMnemonic(last.mnemonic)) {
        uint64_t fallthrough = last.offset + last.size;
        auto ft_it = cfg.offset_to_block.find(fallthrough);
        if (ft_it != cfg.offset_to_block.end())
          bb.successors.push_back(ft_it->second);
      }
    } else {
      if (bi + 1 < static_cast<int>(cfg.blocks.size()))
        bb.successors.push_back(bi + 1);
    }
  }

  for (int bi = 0; bi < static_cast<int>(cfg.blocks.size()); ++bi) {
    for (int succ : cfg.blocks[bi].successors) {
      if (succ >= 0 && succ < static_cast<int>(cfg.blocks.size()))
        cfg.blocks[succ].predecessors.push_back(bi);
    }
  }

  return cfg;
}

// ── Backward liveness analysis ───────────────────────────────────────────────

struct LivenessInfo {
  std::vector<std::set<int>> live_before;
  std::vector<std::set<int>> live_after;
};

static LivenessInfo ComputeLiveness(
    const std::vector<InternalDecodedInst> &decoded,
    const CFG &cfg,
    const llvm::MCInstrInfo &MCII,
    const llvm::MCRegisterInfo &MRI) {
  size_t n_inst = decoded.size();
  LivenessInfo info;
  info.live_before.resize(n_inst);
  info.live_after.resize(n_inst);

  size_t n_blocks = cfg.blocks.size();
  if (n_blocks == 0) return info;

  std::vector<std::set<int>> bb_live_in(n_blocks);
  std::vector<std::set<int>> bb_live_out(n_blocks);

  bool changed = true;
  int max_iters = 200;
  while (changed && max_iters-- > 0) {
    changed = false;
    for (int bi = static_cast<int>(n_blocks) - 1; bi >= 0; --bi) {
      const auto &bb = cfg.blocks[bi];
      if (bb.inst_indices.empty()) continue;

      std::set<int> new_live_out;
      size_t last_idx = bb.inst_indices.back();
      const auto &last = decoded[last_idx];
      if (last.mnemonic.find("s_setpc") == 0 ||
          last.mnemonic.find("s_swappc") == 0) {
        for (int v = 0; v < 256; ++v)
          new_live_out.insert(v);
      } else {
        for (int succ : bb.successors) {
          if (succ >= 0 && succ < static_cast<int>(n_blocks))
            new_live_out.insert(bb_live_in[succ].begin(),
                                bb_live_in[succ].end());
        }
      }

      std::set<int> live = new_live_out;
      for (int ii = static_cast<int>(bb.inst_indices.size()) - 1;
           ii >= 0; --ii) {
        size_t inst_idx = bb.inst_indices[ii];
        RegDefUse du = GetInstRegDefUse(decoded[inst_idx].inst, MCII, MRI);
        for (int d : du.defs) live.erase(d);
        live.insert(du.uses.begin(), du.uses.end());
      }

      if (live != bb_live_in[bi]) {
        bb_live_in[bi] = std::move(live);
        changed = true;
      }
      if (new_live_out != bb_live_out[bi]) {
        bb_live_out[bi] = std::move(new_live_out);
        changed = true;
      }
    }
  }

  for (int bi = 0; bi < static_cast<int>(n_blocks); ++bi) {
    const auto &bb = cfg.blocks[bi];
    if (bb.inst_indices.empty()) continue;

    std::set<int> live = bb_live_out[bi];
    for (int ii = static_cast<int>(bb.inst_indices.size()) - 1;
         ii >= 0; --ii) {
      size_t inst_idx = bb.inst_indices[ii];
      info.live_after[inst_idx] = live;
      RegDefUse du = GetInstRegDefUse(decoded[inst_idx].inst, MCII, MRI);
      for (int d : du.defs) live.erase(d);
      live.insert(du.uses.begin(), du.uses.end());
      info.live_before[inst_idx] = live;
    }
  }

  return info;
}

// ── Scratch register allocator ───────────────────────────────────────────────

struct ScratchAllocator {
  std::set<int> live_at_point;
  int kd_allocated_vgprs;
  int next_above_kd;
  int extra_allocated = 0;

  ScratchAllocator(const std::set<int> &live, int kd_vgprs)
      : live_at_point(live), kd_allocated_vgprs(kd_vgprs),
        next_above_kd(kd_vgprs) {}

  int Alloc() {
    for (int v = kd_allocated_vgprs - 1; v >= 0; --v) {
      if (live_at_point.find(v) == live_at_point.end()) {
        live_at_point.insert(v);
        return v;
      }
    }
    if (next_above_kd >= 256) return -1;
    int v = next_above_kd++;
    extra_allocated++;
    live_at_point.insert(v);
    return v;
  }

  int ExtraVgprsNeeded() const { return extra_allocated; }
};

struct ScratchPatchInfo {
  uint64_t offset;
  std::set<int> scratch_regs;
};

static int GetKernelVgprCount(const uint8_t *elf_data, size_t elf_size,
                              const ElfInfo &elf_info,
                              const std::string &kernel_name) {
  std::string kd_name = kernel_name + ".kd";
  for (const auto &sym : elf_info.symbols) {
    if (sym.name != kd_name) continue;
    if (sym.shndx >= elf_info.sections.size()) continue;
    const auto &sec = elf_info.sections[sym.shndx];
    uint64_t kd_file_offset = sec.offset + sym.value;
    if (kd_file_offset + 64 > elf_size) continue;
    uint32_t rsrc1;
    std::memcpy(&rsrc1, elf_data + kd_file_offset + 48, 4);
    uint32_t granulated = rsrc1 & 0x3F;
    return static_cast<int>((granulated + 1) * 8);
  }
  return 256;
}

// ── Post-patch verification ──────────────────────────────────────────────────

static void VerifyPatchCorrectness(
    const uint8_t *text, uint64_t text_size,
    const LLVMState &llvm_state,
    const std::vector<ScratchPatchInfo> &scratch_patches) {
  if (scratch_patches.empty()) return;

  std::vector<InternalDecodedInst> decoded;
  if (!DecodeTextSection(text, text_size, llvm_state, decoded)) return;

  CFG cfg = BuildCFG(decoded);
  LivenessInfo liveness = ComputeLiveness(decoded, cfg,
                                          *llvm_state.MCII, *llvm_state.MRI);

  std::map<uint64_t, size_t> offset_to_idx;
  for (size_t i = 0; i < decoded.size(); ++i)
    offset_to_idx[decoded[i].offset] = i;

  for (const auto &sp : scratch_patches) {
    auto it = offset_to_idx.find(sp.offset);
    if (it == offset_to_idx.end()) continue;
    size_t idx = it->second;
    if (idx >= liveness.live_before.size()) continue;

    for (int reg : sp.scratch_regs) {
      if (liveness.live_before[idx].count(reg)) {
        std::cerr << "hotswap: WARNING: scratch v" << reg
                  << " is live at patch point 0x" << std::hex << sp.offset
                  << std::dec << " in post-patch verification\n";
      }
    }
  }
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
  std::vector<NopSled> nop_sleds = BuildNopSledMap(decoded);

  CFG cfg = BuildCFG(decoded);
  LivenessInfo liveness = ComputeLiveness(decoded, cfg,
                                          *llvm_state.MCII, *llvm_state.MRI);

  struct KernelPatchStats {
    int extra_vgprs = 0;
    int scratch_reused = 0;
    int scratch_above_kd = 0;
  };
  std::map<std::string, KernelPatchStats> kernel_stats;

  for (size_t idx = 0; idx < decoded.size(); ++idx) {
    auto &di = decoded[idx];
    if (di.mnemonic == "<unknown>" || di.mnemonic == "<replaced>") continue;

    // Patch 1: CLUSTER_LOAD → GLOBAL_LOAD
    for (size_t swap_i = 0; swap_i < kClusterLoadSwapsSize; ++swap_i) {
      const auto &swap = kClusterLoadSwaps[swap_i];
      if (di.mnemonic == swap.first) {
        RewriteRule rule;
        rule.replace_mnemonic = swap.second;
        rule.preserve_operands = true;
        if (ApplyMnemonicSwap(rule, di, text, llvm_state)) {
          di.mnemonic = swap.second;
          ++patched;
        }
        break;
      }
    }

    // Patch 2: DS 2-addr stride64 → two single-addr ops via trampoline
    for (size_t swap_i = 0; swap_i < kDs2AddrSwapsSize; ++swap_i) {
      const auto &swap = kDs2AddrSwaps[swap_i];
      if (di.mnemonic == swap.first) {
        if (di.mnemonic.find("stride64") == std::string::npos) break;
        std::string inst_str = PrintInst(di, llvm_state);
        if (inst_str.empty()) break;
        std::vector<std::string> asm_lines =
            ExpandDs2AddrAsm(inst_str, swap.first, swap.second);
        if (asm_lines.size() != 2) break;
        auto bytes0 = AssembleSingleInst(asm_lines[0], llvm_state);
        auto bytes1 = AssembleSingleInst(asm_lines[1], llvm_state);
        if (bytes0.empty() || bytes1.empty()) break;

        uint64_t tramp_offset = text_size;
        for (auto &t : out_trampolines)
          tramp_offset += t.bytes.size();

        Trampoline tramp;
        tramp.original_offset = di.offset;
        tramp.original_size = di.size;
        tramp.bytes.insert(tramp.bytes.end(), bytes0.begin(), bytes0.end());
        tramp.bytes.insert(tramp.bytes.end(), bytes1.begin(), bytes1.end());

        uint8_t br_back[4];
        if (!EncodeSBranch(tramp_offset + tramp.bytes.size(),
                           di.offset + di.size, br_back, true))
          break;
        tramp.bytes.insert(tramp.bytes.end(), br_back, br_back + 4);
        out_trampolines.push_back(std::move(tramp));
        di.mnemonic = "<replaced>";
        ++patched;
        break;
      }
    }

    // Patch 3: s_clause → s_nop
    if (di.mnemonic == "s_clause") {
      RewriteRule rule;
      rule.replace_bytes = {0x00, 0x00, 0x80, 0xBF};
      if (ApplyByteReplace(rule, di.offset, di.size, text, text_size)) {
        di.mnemonic = "s_nop";
        ++patched;
      }
    }

    // Patch 4: TENSOR_LOAD_TO_LDS multicast stripping
    if (di.mnemonic == "tensor_load_to_lds") {
      if (idx > 0 && decoded[idx - 1].mnemonic == "s_pack_hh_b32_b16")
        continue;
      std::string inst_str = PrintInst(di, llvm_state);
      if (inst_str.empty()) continue;

      size_t first_comma = inst_str.find(',');
      if (first_comma == std::string::npos) continue;
      std::string after = inst_str.substr(first_comma + 1);

      size_t s_pos = after.find("s[");
      if (s_pos == std::string::npos) {
        s_pos = after.find_first_of("s");
        if (s_pos == std::string::npos) continue;
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
      if (base_sreg.empty()) continue;

      std::string pack_asm =
          "s_pack_hh_b32_b16 " + base_sreg + ", 0, " + base_sreg;
      auto pack_bytes = AssembleSingleInst(pack_asm, llvm_state);
      if (pack_bytes.empty() || pack_bytes.size() != 4) continue;

      NopSled *sled = FindNearestSled(nop_sleds, di.offset, 20);
      if (sled) {
        uint64_t tp = sled->write_pos;
        std::memcpy(text + tp, pack_bytes.data(), 4);
        std::memcpy(text + tp + 4, text + di.offset, di.size);
        uint8_t br_back[4];
        if (!EncodeSBranch(tp + 4 + di.size, di.offset + di.size, br_back,
                           true))
          continue;
        std::memcpy(text + tp + 4 + di.size, br_back, 4);
        uint8_t br_fwd[4];
        if (!EncodeSBranch(di.offset, tp, br_fwd, true)) continue;
        std::memcpy(text + di.offset, br_fwd, 4);
        for (uint32_t i = 4; i < di.size; i += 4) {
          uint8_t nop[4];
          EncodeSNop(nop);
          std::memcpy(text + di.offset + i, nop, 4);
        }
        sled->write_pos += 4 + di.size + 4;
      } else {
        Trampoline t;
        t.original_offset = di.offset;
        t.original_size = di.size;
        t.bytes.resize(4 + di.size + 4);
        std::memcpy(t.bytes.data(), pack_bytes.data(), 4);
        std::memcpy(t.bytes.data() + 4, text + di.offset, di.size);
        uint8_t placeholder[4] = {0};
        std::memcpy(t.bytes.data() + 4 + di.size, placeholder, 4);
        out_trampolines.push_back(std::move(t));
      }
      di.mnemonic = "<replaced>";
      ++patched;
    }

    // Patch 6: 16x16x128 FP8/BF8 WMMA → Two 16x16x64 WMMAs
    if (di.mnemonic.find("16x16x128") != std::string::npos &&
        (di.mnemonic.find("_fp8") != std::string::npos ||
         di.mnemonic.find("_bf8") != std::string::npos) &&
        di.mnemonic.find("f8f6f4") == std::string::npos) {
      auto [d_base, d_count] =
          GetOperandVgprRange(di.inst, 0, *llvm_state.MRI);
      auto [a_base, a_count] =
          GetOperandVgprRange(di.inst, 1, *llvm_state.MRI);
      auto [b_base, b_count] =
          GetOperandVgprRange(di.inst, 2, *llvm_state.MRI);
      if (d_base >= 0 && a_base >= 0 && b_base >= 0 && a_count >= 16 &&
          b_count >= 16) {
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
        auto enc1 = AssembleSingleInst(asm1, llvm_state);
        auto enc2 = AssembleSingleInst(asm2, llvm_state);
        if (!enc1.empty() && !enc2.empty()) {
          uint32_t tramp_size = enc1.size() + enc2.size() + 4;

          NopSled *sled = FindNearestSled(nop_sleds, di.offset, tramp_size);
          if (sled) {
            uint64_t tp = sled->write_pos;
            std::memcpy(text + tp, enc1.data(), enc1.size());
            std::memcpy(text + tp + enc1.size(), enc2.data(), enc2.size());
            uint8_t br_back[4];
            if (EncodeSBranch(tp + enc1.size() + enc2.size(),
                              di.offset + di.size, br_back, true)) {
              std::memcpy(text + tp + enc1.size() + enc2.size(), br_back, 4);
              uint8_t br_fwd[4];
              if (EncodeSBranch(di.offset, tp, br_fwd, true)) {
                std::memcpy(text + di.offset, br_fwd, 4);
                for (uint32_t i = 4; i < di.size; i += 4) {
                  uint8_t nop[4]; EncodeSNop(nop);
                  std::memcpy(text + di.offset + i, nop, 4);
                }
                sled->write_pos += tramp_size;
                std::cerr << "hotswap: B0->A0 @0x" << std::hex << di.offset
                          << ": " << di.mnemonic << " -> 2x " << mnem64
                          << " via sled @0x" << tp << std::dec << "\n";
                di.mnemonic = "<replaced>";
                ++patched;
                continue;
              }
            }
          }
          Trampoline t;
          t.original_offset = di.offset;
          t.original_size = di.size;
          t.bytes.resize(tramp_size);
          std::memcpy(t.bytes.data(), enc1.data(), enc1.size());
          std::memcpy(t.bytes.data() + enc1.size(), enc2.data(), enc2.size());
          uint8_t placeholder[4] = {0};
          std::memcpy(t.bytes.data() + enc1.size() + enc2.size(), placeholder,
                      4);
          out_trampolines.push_back(std::move(t));
          std::cerr << "hotswap: B0->A0 @0x" << std::hex << di.offset
                    << ": " << di.mnemonic << " -> 2x " << mnem64
                    << " deferred for ELF growth" << std::dec << "\n";
          di.mnemonic = "<replaced>";
          ++patched;
        }
      }
    }

    // Patch 7: 32x16x128 F4 WMMA → Two 16x16x128 F8F6F4 WMMAs
    if (di.mnemonic.find("32x16x128_f4") != std::string::npos &&
        di.mnemonic.find("v_wmma") == 0) {
      bool is_scaled = (di.mnemonic.find("_scale_") != std::string::npos ||
                        di.mnemonic.find("_scale16_") != std::string::npos);

      if (!is_scaled) {
        auto [d_base, d_count] = GetOperandVgprRange(di.inst, 0, *llvm_state.MRI);
        auto [a_base, a_count] = GetOperandVgprRange(di.inst, 1, *llvm_state.MRI);
        auto [b_base, b_count] = GetOperandVgprRange(di.inst, 2, *llvm_state.MRI);

        if (d_base >= 0 && a_base >= 0 && b_base >= 0 && d_count >= 16) {
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

          auto enc1 = AssembleSingleInst(asm1, llvm_state);
          auto enc2 = AssembleSingleInst(asm2, llvm_state);

          if (!enc1.empty() && !enc2.empty()) {
            uint32_t tramp_size = enc1.size() + enc2.size() + 4;

            NopSled *sled = FindNearestSled(nop_sleds, di.offset, tramp_size);
            if (sled) {
              uint64_t tp = sled->write_pos;
              std::memcpy(text + tp, enc1.data(), enc1.size());
              std::memcpy(text + tp + enc1.size(), enc2.data(), enc2.size());
              uint8_t br_back[4];
              if (EncodeSBranch(tp + enc1.size() + enc2.size(),
                                di.offset + di.size, br_back, true)) {
                std::memcpy(text + tp + enc1.size() + enc2.size(), br_back, 4);
                uint8_t br_fwd[4];
                if (EncodeSBranch(di.offset, tp, br_fwd, true)) {
                  std::memcpy(text + di.offset, br_fwd, 4);
                  for (uint32_t i = 4; i < di.size; i += 4) {
                    uint8_t nop[4]; EncodeSNop(nop);
                    std::memcpy(text + di.offset + i, nop, 4);
                  }
                  sled->write_pos += tramp_size;
                  std::cerr << "hotswap: B0->A0 @0x" << std::hex << di.offset
                            << ": " << di.mnemonic
                            << " -> 2x v_wmma_f32_16x16x128_f8f6f4 via sled @0x"
                            << tp << std::dec << "\n";
                  di.mnemonic = "<replaced>";
                  ++patched;
                  continue;
                }
              }
            }
            Trampoline t;
            t.original_offset = di.offset;
            t.original_size = di.size;
            t.bytes.resize(tramp_size);
            std::memcpy(t.bytes.data(), enc1.data(), enc1.size());
            std::memcpy(t.bytes.data() + enc1.size(), enc2.data(), enc2.size());
            uint8_t placeholder[4] = {0};
            std::memcpy(t.bytes.data() + enc1.size() + enc2.size(), placeholder, 4);
            out_trampolines.push_back(std::move(t));
            std::cerr << "hotswap: B0->A0 @0x" << std::hex << di.offset
                      << ": " << di.mnemonic << " deferred for ELF growth"
                      << std::dec << "\n";
            di.mnemonic = "<replaced>";
            ++patched;
          }
        }
      }
    }

    // Patch 8: E5M3 CVT (CLAMP=1) → VALU Emulation
    if ((di.mnemonic.find("v_cvt_f32_fp8") == 0 ||
         di.mnemonic.find("v_cvt_pk_fp8_f32") == 0 ||
         di.mnemonic.find("v_cvt_sr_fp8_f32") == 0) && di.size >= 8) {
      uint32_t dw0;
      std::memcpy(&dw0, text + di.offset, 4);
      bool has_clamp = (dw0 >> 15) & 1;

      if (has_clamp) {
        auto [dst_base, dst_count] = GetOperandVgprRange(di.inst, 0, *llvm_state.MRI);
        auto [src_base, src_count] = GetOperandVgprRange(di.inst, 1, *llvm_state.MRI);

        if (dst_base >= 0 && src_base >= 0) {
          std::string kernel_p8 = FindKernelAtOffset(elf_info, di.offset);
          int kd_vgprs_p8 = GetKernelVgprCount(elf_data, elf_size, elf_info, kernel_p8);
          ScratchAllocator alloc_p8(liveness.live_before[idx], kd_vgprs_p8);
          int s0 = alloc_p8.Alloc(), s1 = alloc_p8.Alloc();
          int s2 = alloc_p8.Alloc(), s3 = alloc_p8.Alloc();
          if (s0 < 0 || s1 < 0 || s2 < 0 || s3 < 0) {
            s0 = 252; s1 = 253; s2 = 254; s3 = 255;
          }
          std::string sv0 = "v" + std::to_string(s0);
          std::string sv1 = "v" + std::to_string(s1);
          std::string sv2 = "v" + std::to_string(s2);
          std::string sv3 = "v" + std::to_string(s3);
          for (int v : {s0, s1, s2, s3}) {
            if (v < kd_vgprs_p8)
              kernel_stats[kernel_p8].scratch_reused++;
            else
              kernel_stats[kernel_p8].scratch_above_kd++;
          }
          kernel_stats[kernel_p8].extra_vgprs =
              std::max(kernel_stats[kernel_p8].extra_vgprs,
                       alloc_p8.ExtraVgprsNeeded());
          out_scratch_patches.push_back({di.offset, {s0, s1, s2, s3}});

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
            auto [src2_base, src2_count] = GetOperandVgprRange(di.inst, 2, *llvm_state.MRI);
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
          auto enc = AssembleSingleInst(joined, llvm_state);

          if (!enc.empty()) {
            uint32_t tramp_size = enc.size() + 4;

            NopSled *sled = FindNearestSled(nop_sleds, di.offset, tramp_size);
            if (sled) {
              uint64_t tp = sled->write_pos;
              std::memcpy(text + tp, enc.data(), enc.size());
              uint8_t br_back[4];
              if (EncodeSBranch(tp + enc.size(), di.offset + di.size, br_back, true)) {
                std::memcpy(text + tp + enc.size(), br_back, 4);
                uint8_t br_fwd[4];
                if (EncodeSBranch(di.offset, tp, br_fwd, true)) {
                  std::memcpy(text + di.offset, br_fwd, 4);
                  for (uint32_t i = 4; i < di.size; i += 4) {
                    uint8_t nop[4]; EncodeSNop(nop);
                    std::memcpy(text + di.offset + i, nop, 4);
                  }
                  sled->write_pos += tramp_size;
                  std::cerr << "hotswap: B0->A0 @0x" << std::hex << di.offset
                            << ": " << di.mnemonic << " CLAMP=1 (E5M3) -> VALU emulation"
                            << " via sled @0x" << tp << std::dec << "\n";
                  di.mnemonic = "<replaced>";
                  ++patched;
                  continue;
                }
              }
            }
            Trampoline t;
            t.original_offset = di.offset;
            t.original_size = di.size;
            t.bytes.resize(tramp_size);
            std::memcpy(t.bytes.data(), enc.data(), enc.size());
            uint8_t placeholder[4] = {0};
            std::memcpy(t.bytes.data() + enc.size(), placeholder, 4);
            out_trampolines.push_back(std::move(t));
            std::cerr << "hotswap: B0->A0 @0x" << std::hex << di.offset
                      << ": " << di.mnemonic << " CLAMP=1 (E5M3) deferred for ELF growth"
                      << std::dec << "\n";
            di.mnemonic = "<replaced>";
            ++patched;
          }
        }
      }
    }

    // Patch 9: Block16 Scale → Block32 Decomposition
    if (di.mnemonic == "v_wmma_scale16_f32_16x16x128_f8f6f4") {
      std::string inst_str = PrintInst(di, llvm_state);
      if (!inst_str.empty()) {
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

        if (ops.size() >= 6) {
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
                try { return std::stoi(s.substr(pos + 1, colon - pos - 1)); } catch(...) {}
              }
            }
            size_t vpos = s.find('v');
            if (vpos != std::string::npos) {
              try { return std::stoi(s.substr(vpos + 1)); } catch(...) {}
            }
            return -1;
          };

          int sa_base = extractBase(sa_str);
          int sb_base = extractBase(sb_str);

          if (sa_base >= 0 && sb_base >= 0) {
            std::string kernel_p9 = FindKernelAtOffset(elf_info, di.offset);
            int kd_vgprs_p9 = GetKernelVgprCount(elf_data, elf_size, elf_info, kernel_p9);
            ScratchAllocator alloc_p9(liveness.live_before[idx], kd_vgprs_p9);
            int p9s0 = alloc_p9.Alloc(), p9s1 = alloc_p9.Alloc();
            if (p9s0 < 0 || p9s1 < 0) { p9s0 = 252; p9s1 = 253; }
            std::string p9sv0 = "v" + std::to_string(p9s0);
            std::string p9sv1 = "v" + std::to_string(p9s1);
            for (int v : {p9s0, p9s1}) {
              if (v < kd_vgprs_p9)
                kernel_stats[kernel_p9].scratch_reused++;
              else
                kernel_stats[kernel_p9].scratch_above_kd++;
            }
            kernel_stats[kernel_p9].extra_vgprs =
                std::max(kernel_stats[kernel_p9].extra_vgprs,
                         alloc_p9.ExtraVgprsNeeded());
            out_scratch_patches.push_back({di.offset, {p9s0, p9s1}});

            std::string repack_sa = "v_perm_b32 " + p9sv0 + ", v" + std::to_string(sa_base) +
                ", v" + std::to_string(sa_base) + ", 0x05010400";
            std::string repack_sb = "v_perm_b32 " + p9sv1 + ", v" + std::to_string(sb_base) +
                ", v" + std::to_string(sb_base) + ", 0x05010400";

            std::string wmma_asm = "v_wmma_scale_f32_16x16x128_f8f6f4 "
                + d_str + ", " + a_str + ", " + b_str + ", "
                + d_str + ", " + p9sv0 + ", " + p9sv1 + modifiers;

            std::string all_asm = repack_sa + "\n" + repack_sb + "\n" + wmma_asm;
            auto enc = AssembleSingleInst(all_asm, llvm_state);

            if (!enc.empty()) {
              uint32_t tramp_size = enc.size() + 4;

              NopSled *sled = FindNearestSled(nop_sleds, di.offset, tramp_size);
              if (sled) {
                uint64_t tp = sled->write_pos;
                std::memcpy(text + tp, enc.data(), enc.size());
                uint8_t br_back[4];
                if (EncodeSBranch(tp + enc.size(), di.offset + di.size, br_back, true)) {
                  std::memcpy(text + tp + enc.size(), br_back, 4);
                  uint8_t br_fwd[4];
                  if (EncodeSBranch(di.offset, tp, br_fwd, true)) {
                    std::memcpy(text + di.offset, br_fwd, 4);
                    for (uint32_t i = 4; i < di.size; i += 4) {
                      uint8_t nop[4]; EncodeSNop(nop);
                      std::memcpy(text + di.offset + i, nop, 4);
                    }
                    sled->write_pos += tramp_size;
                    std::cerr << "hotswap: B0->A0 @0x" << std::hex << di.offset
                              << ": " << di.mnemonic
                              << " -> block32 decomposition via sled @0x"
                              << tp << std::dec << "\n";
                    di.mnemonic = "<replaced>";
                    ++patched;
                    continue;
                  }
                }
              }
              Trampoline t;
              t.original_offset = di.offset;
              t.original_size = di.size;
              t.bytes.resize(tramp_size);
              std::memcpy(t.bytes.data(), enc.data(), enc.size());
              uint8_t placeholder[4] = {0};
              std::memcpy(t.bytes.data() + enc.size(), placeholder, 4);
              out_trampolines.push_back(std::move(t));
              std::cerr << "hotswap: B0->A0 @0x" << std::hex << di.offset
                        << ": " << di.mnemonic << " -> block32 deferred for ELF growth"
                        << std::dec << "\n";
              di.mnemonic = "<replaced>";
              ++patched;
            }
          }
        }
      }
    }

    if (di.mnemonic == "v_wmma_ld_scale16_paired_b64") {
      auto [dst_base, dst_count] = GetOperandVgprRange(di.inst, 0, *llvm_state.MRI);
      auto [src_base, src_count] = GetOperandVgprRange(di.inst, 1, *llvm_state.MRI);

      if (dst_base >= 0 && src_base >= 0) {
        std::string load_asm = "v_wmma_ld_scale_paired_b32 v"
            + std::to_string(dst_base) + ", v" + std::to_string(src_base);

        auto enc = AssembleSingleInst(load_asm, llvm_state);

        if (!enc.empty()) {
          if (enc.size() <= di.size) {
            std::memcpy(text + di.offset, enc.data(), enc.size());
            for (uint32_t i = enc.size(); i < di.size; i += 4) {
              uint8_t nop[4]; EncodeSNop(nop);
              std::memcpy(text + di.offset + i, nop, 4);
            }
            std::cerr << "hotswap: B0->A0 @0x" << std::hex << di.offset
                      << ": v_wmma_ld_scale16 -> block32 v_wmma_ld_scale_paired_b32"
                      << std::dec << "\n";
            di.mnemonic = "<replaced>";
            ++patched;
          }
        }
      }
    }
  }

  // Patch 5: WMMA co-execution hazard V_NOP insertion
  auto hazards = ValidateWmmaCoexecHazards(decoded, text, llvm_state);
  if (!hazards.empty()) {
    auto vnop_bytes = AssembleSingleInst("v_nop", llvm_state);
    if (!vnop_bytes.empty() && vnop_bytes.size() == 4) {
      for (auto &h : hazards) {
        const auto &valu = decoded[h.valu_idx];
        uint32_t valu_size = valu.size;
        uint32_t nop_bytes_count = h.deficit * 4;
        uint32_t tramp_size = nop_bytes_count + valu_size + 4;

        NopSled *sled = FindNearestSled(nop_sleds, valu.offset, tramp_size);
        if (sled) {
          uint64_t tp = sled->write_pos;
          for (int n = 0; n < h.deficit; ++n)
            std::memcpy(text + tp + n * 4, vnop_bytes.data(), 4);
          std::memcpy(text + tp + nop_bytes_count, text + valu.offset,
                      valu_size);
          uint8_t br_back[4];
          if (!EncodeSBranch(tp + nop_bytes_count + valu_size,
                             valu.offset + valu_size, br_back, true))
            continue;
          std::memcpy(text + tp + nop_bytes_count + valu_size, br_back, 4);
          uint8_t br_fwd[4];
          if (!EncodeSBranch(valu.offset, tp, br_fwd, true)) continue;
          std::memcpy(text + valu.offset, br_fwd, 4);
          for (uint32_t i = 4; i < valu_size; i += 4) {
            uint8_t nop[4];
            EncodeSNop(nop);
            std::memcpy(text + valu.offset + i, nop, 4);
          }
          sled->write_pos += tramp_size;
          ++patched;
        } else {
          Trampoline t;
          t.original_offset = valu.offset;
          t.original_size = valu_size;
          t.bytes.resize(tramp_size);
          for (int n = 0; n < h.deficit; ++n)
            std::memcpy(t.bytes.data() + n * 4, vnop_bytes.data(), 4);
          std::memcpy(t.bytes.data() + nop_bytes_count,
                      text + valu.offset, valu_size);
          uint8_t placeholder[4] = {0};
          std::memcpy(t.bytes.data() + nop_bytes_count + valu_size,
                      placeholder, 4);
          out_trampolines.push_back(std::move(t));
          ++patched;
        }
      }
    }
  }

  for (const auto &kv : kernel_stats) {
    const std::string &kname = kv.first;
    const auto &stats = kv.second;
    if (kname.empty()) continue;
    int vgprs_before = GetKernelVgprCount(elf_data, elf_size, elf_info, kname);
    if (stats.extra_vgprs > 0)
      UpdateKernelDescriptor(elf_data, elf_size, elf_info, kname,
                             stats.extra_vgprs, 0);
    int vgprs_after = GetKernelVgprCount(elf_data, elf_size, elf_info, kname);
    std::cerr << "hotswap: liveness: kernel " << kname
              << ": vgprs_before=" << vgprs_before
              << ", vgprs_after=" << vgprs_after
              << ", scratch_reused=" << stats.scratch_reused
              << ", scratch_above_kd=" << stats.scratch_above_kd << "\n";
  }

  return patched;
}

// ── PatchElfIsa ──────────────────────────────────────────────────────────────

static bool PatchElfIsa(uint8_t *elf, size_t elf_size,
                        const std::string &target_cpu) {
  if (elf_size < 64) return false;
  if (elf[0] != 0x7f || elf[1] != 'E' || elf[2] != 'L' || elf[3] != 'F')
    return false;
  if (elf[4] != 2) return false;

  struct GfxMach { const char *name; uint32_t mach; };
  static const GfxMach gfx_mach_map[] = {
      {"gfx900", 0x02c},  {"gfx906", 0x02f},  {"gfx908", 0x030},
      {"gfx90a", 0x03f},  {"gfx940", 0x04a},  {"gfx941", 0x04b},
      {"gfx942", 0x04c},  {"gfx950", 0x04f},  {"gfx1010", 0x033},
      {"gfx1030", 0x036}, {"gfx1100", 0x041},  {"gfx1200", 0x048},
      {"gfx1201", 0x04a}, {"gfx1250", 0x049},  {nullptr, 0}};

  uint32_t target_mach = 0;
  for (auto *p = gfx_mach_map; p->name; ++p) {
    if (target_cpu == p->name) {
      target_mach = p->mach;
      break;
    }
  }
  if (target_mach == 0) return false;

  uint32_t e_flags;
  std::memcpy(&e_flags, elf + 48, 4);
  e_flags = (e_flags & ~0xFFu) | (target_mach & 0xFF);
  std::memcpy(elf + 48, &e_flags, 4);

  uint64_t e_shoff;
  uint16_t e_shentsize, e_shnum;
  std::memcpy(&e_shoff, elf + 40, 8);
  std::memcpy(&e_shentsize, elf + 58, 2);
  std::memcpy(&e_shnum, elf + 60, 2);
  if (e_shoff == 0 || e_shnum == 0) return true;

  for (uint16_t i = 0; i < e_shnum; ++i) {
    const uint8_t *sh = elf + e_shoff + i * e_shentsize;
    uint32_t sh_type;
    std::memcpy(&sh_type, sh + 4, 4);
    if (sh_type != 7) continue;
    uint64_t sh_offset, sh_size;
    std::memcpy(&sh_offset, sh + 24, 8);
    std::memcpy(&sh_size, sh + 32, 8);
    if (sh_offset + sh_size > elf_size) continue;
    uint64_t pos = sh_offset;
    while (pos + 12 <= sh_offset + sh_size) {
      uint32_t namesz, descsz, type;
      std::memcpy(&namesz, elf + pos, 4);
      std::memcpy(&descsz, elf + pos + 4, 4);
      std::memcpy(&type, elf + pos + 8, 4);
      uint32_t namesz_aligned = (namesz + 3) & ~3u;
      uint32_t descsz_aligned = (descsz + 3) & ~3u;
      uint64_t note_total = 12 + namesz_aligned + descsz_aligned;
      if (pos + note_total > sh_offset + sh_size) break;
      if (type == 27 && namesz > 0) {
        const char *owner = reinterpret_cast<const char *>(elf + pos + 12);
        if (std::strncmp(owner, "AMDGPU", 6) == 0) {
          uint8_t *desc = elf + pos + 12 + namesz_aligned;
          std::string orig_isa(reinterpret_cast<const char *>(desc), descsz);
          size_t gfx_pos = orig_isa.find("gfx");
          if (gfx_pos != std::string::npos) {
            size_t gfx_end = gfx_pos;
            while (gfx_end < orig_isa.size() && orig_isa[gfx_end] != ':' &&
                   orig_isa[gfx_end] != '\0')
              ++gfx_end;
            std::string orig_gfx =
                orig_isa.substr(gfx_pos, gfx_end - gfx_pos);
            if (target_cpu.size() <= orig_gfx.size()) {
              std::memcpy(desc + gfx_pos, target_cpu.c_str(),
                          target_cpu.size());
              for (size_t j = target_cpu.size(); j < orig_gfx.size(); ++j)
                desc[gfx_pos + j] = '\0';
            }
          }
        }
      }
      pos += note_total;
    }
  }
  return true;
}

// ── NeedsTranspile ───────────────────────────────────────────────────────────

static bool NeedsTranspileImpl(const std::string &source_isa,
                               const std::string &target_isa) {
  std::string src_cpu = ExtractCPU(source_isa);
  std::string tgt_cpu = ExtractCPU(target_isa);
  if (src_cpu.empty() || tgt_cpu.empty()) return false;
  auto isGFX9 = [](const std::string &cpu) {
    return cpu.size() >= 4 && cpu[3] == '9';
  };
  auto isGFX12 = [](const std::string &cpu) {
    return cpu.size() >= 5 && cpu[3] == '1' && cpu[4] == '2';
  };
  return isGFX12(src_cpu) && isGFX9(tgt_cpu);
}

// ── RetargetCodeObjectB0A0Grow ───────────────────────────────────────────────

static amd_comgr_status_t
RetargetCodeObjectB0A0Grow(const void *elf_data, size_t elf_size,
                           void **out_data, size_t *out_size,
                           amd_comgr_hotswap_result_t *result) {
  const std::string isa = "amdgcn-amd-amdhsa--gfx1250";

  ElfInfo elf_info;
  const uint8_t *elf = static_cast<const uint8_t *>(elf_data);
  if (!ParseElfInfo(elf, elf_size, elf_info) || elf_info.text_size == 0) {
    *out_data = std::malloc(elf_size);
    if (!*out_data) return AMD_COMGR_STATUS_ERROR;
    std::memcpy(*out_data, elf_data, elf_size);
    *out_size = elf_size;
    return AMD_COMGR_STATUS_SUCCESS;
  }

  LLVMState &llvm_state = InitLLVMCached(isa);
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

    size_t new_size = 0;
    uint8_t *new_elf = GrowElfWithTrampolines(buf.data(), elf_size, elf_info,
                                              deferred, &new_size);
    if (!new_elf) return AMD_COMGR_STATUS_ERROR;
    *out_data = new_elf;
    *out_size = new_size;
    result->trampolines_added = static_cast<uint32_t>(deferred.size());
  } else {
    uint8_t *out = static_cast<uint8_t *>(std::malloc(elf_size));
    if (!out) return AMD_COMGR_STATUS_ERROR;
    std::memcpy(out, buf.data(), elf_size);
    *out_data = out;
    *out_size = elf_size;
  }

  if (!scratch_patches.empty()) {
    ElfInfo verify_elf_info;
    const uint8_t *verify_elf = static_cast<const uint8_t *>(*out_data);
    if (ParseElfInfo(verify_elf, *out_size, verify_elf_info) &&
        verify_elf_info.text_size > 0) {
      VerifyPatchCorrectness(verify_elf + verify_elf_info.text_offset,
                             verify_elf_info.text_size,
                             llvm_state, scratch_patches);
    }
  }

  return AMD_COMGR_STATUS_SUCCESS;
}

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

// ── TranspileCodeObject (stub — full transpile uses TranslateInstruction) ────

static amd_comgr_status_t
TranspileCodeObject(const void *elf_data, size_t elf_size,
                    const std::string &source_isa,
                    const std::string &target_isa,
                    void **out_data, size_t *out_size,
                    amd_comgr_hotswap_result_t *result) {
  uint8_t *buf = static_cast<uint8_t *>(std::malloc(elf_size));
  if (!buf) return AMD_COMGR_STATUS_ERROR;
  std::memcpy(buf, elf_data, elf_size);

  std::string tgt_cpu = ExtractCPU(target_isa);
  PatchElfIsa(buf, elf_size, tgt_cpu);

  *out_data = buf;
  *out_size = elf_size;
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
        bool is_gfx12 = llvm_state.cpu.find("gfx12") == 0;
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
