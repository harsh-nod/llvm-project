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

// ── DWARF / Debug update helpers ─────────────────────────────────────────────

static void EncodeULEB128(uint64_t value, std::vector<uint8_t> &out) {
  do {
    uint8_t byte = value & 0x7f;
    value >>= 7;
    if (value) byte |= 0x80;
    out.push_back(byte);
  } while (value);
}

static void EncodeSLEB128(int64_t value, std::vector<uint8_t> &out) {
  bool more = true;
  while (more) {
    uint8_t byte = value & 0x7f;
    value >>= 7;
    if ((value == 0 && !(byte & 0x40)) || (value == -1 && (byte & 0x40)))
      more = false;
    else
      byte |= 0x80;
    out.push_back(byte);
  }
}

static uint64_t DecodeULEB128(const uint8_t *p, size_t *n) {
  uint64_t result = 0;
  unsigned shift = 0;
  size_t i = 0;
  do {
    result |= static_cast<uint64_t>(p[i] & 0x7f) << shift;
    shift += 7;
  } while (p[i++] & 0x80);
  *n = i;
  return result;
}

static int64_t DecodeSLEB128(const uint8_t *p, size_t *n) {
  int64_t result = 0;
  unsigned shift = 0;
  size_t i = 0;
  uint8_t byte;
  do {
    byte = p[i++];
    result |= static_cast<int64_t>(byte & 0x7f) << shift;
    shift += 7;
  } while (byte & 0x80);
  if (shift < 64 && (byte & 0x40))
    result |= -(static_cast<int64_t>(1) << shift);
  *n = i;
  return result;
}

static uint8_t *FindSectionHeader(uint8_t *elf, size_t elf_size,
                                   const char *name, int *out_idx = nullptr) {
  if (elf_size < 64) return nullptr;
  uint64_t e_shoff;
  uint16_t e_shentsize, e_shnum, e_shstrndx;
  std::memcpy(&e_shoff, elf + 40, 8);
  std::memcpy(&e_shentsize, elf + 58, 2);
  std::memcpy(&e_shnum, elf + 60, 2);
  std::memcpy(&e_shstrndx, elf + 62, 2);
  if (e_shoff == 0 || e_shnum == 0 || e_shstrndx >= e_shnum) return nullptr;
  if (e_shoff + static_cast<uint64_t>(e_shnum) * e_shentsize > elf_size)
    return nullptr;
  uint8_t *shstr_sh = elf + e_shoff + e_shstrndx * e_shentsize;
  uint64_t shstr_off, shstr_sz;
  std::memcpy(&shstr_off, shstr_sh + 24, 8);
  std::memcpy(&shstr_sz, shstr_sh + 32, 8);
  if (shstr_off + shstr_sz > elf_size) return nullptr;
  const char *shstrtab = reinterpret_cast<const char *>(elf + shstr_off);

  for (uint16_t i = 0; i < e_shnum; ++i) {
    uint8_t *sh = elf + e_shoff + i * e_shentsize;
    uint32_t sh_name;
    std::memcpy(&sh_name, sh, 4);
    if (sh_name < shstr_sz && std::strcmp(shstrtab + sh_name, name) == 0) {
      if (out_idx) *out_idx = i;
      return sh;
    }
  }
  return nullptr;
}

static uint8_t *AddTrampolineSymbols(
    uint8_t *elf, size_t elf_size,
    const std::vector<Trampoline> &trampolines,
    uint64_t text_size_before, int text_section_idx,
    size_t *out_size) {
  if (trampolines.empty()) { *out_size = elf_size; return elf; }

  int symtab_idx = -1;
  uint8_t *symtab_sh = FindSectionHeader(elf, elf_size, ".symtab", &symtab_idx);
  if (!symtab_sh) { *out_size = elf_size; return elf; }

  uint64_t symtab_offset, symtab_size;
  uint32_t symtab_link, symtab_info;
  std::memcpy(&symtab_offset, symtab_sh + 24, 8);
  std::memcpy(&symtab_size, symtab_sh + 32, 8);
  std::memcpy(&symtab_link, symtab_sh + 40, 4);
  std::memcpy(&symtab_info, symtab_sh + 44, 4);

  uint16_t e_shnum;
  std::memcpy(&e_shnum, elf + 60, 2);
  if (symtab_link >= e_shnum) { *out_size = elf_size; return elf; }

  uint64_t e_shoff;
  uint16_t e_shentsize;
  std::memcpy(&e_shoff, elf + 40, 8);
  std::memcpy(&e_shentsize, elf + 58, 2);

  uint8_t *strtab_sh = elf + e_shoff + symtab_link * e_shentsize;
  uint64_t strtab_offset, strtab_size;
  std::memcpy(&strtab_offset, strtab_sh + 24, 8);
  std::memcpy(&strtab_size, strtab_sh + 32, 8);

  std::vector<std::string> names;
  std::vector<std::vector<uint8_t>> entries;
  uint64_t running = text_size_before;

  for (auto &t : trampolines) {
    std::ostringstream oss;
    oss << "__hotswap_tramp_" << std::hex << t.original_offset;
    names.push_back(oss.str());

    std::vector<uint8_t> entry(24, 0);
    entry[4] = 0x02; // ELF64_ST_INFO(STB_LOCAL, STT_FUNC)
    uint16_t shndx = static_cast<uint16_t>(text_section_idx);
    std::memcpy(entry.data() + 6, &shndx, 2);
    std::memcpy(entry.data() + 8, &running, 8);
    uint64_t sz = t.bytes.size();
    std::memcpy(entry.data() + 16, &sz, 8);
    entries.push_back(std::move(entry));
    running += t.bytes.size();
  }

  size_t extra_str = 0;
  for (auto &n : names) extra_str += n.size() + 1;
  size_t extra_sym = entries.size() * 24;
  size_t new_strtab_total = strtab_size + extra_str;
  size_t new_symtab_total = symtab_size + extra_sym;
  size_t new_elf_size = elf_size + new_strtab_total + new_symtab_total;

  uint8_t *out = static_cast<uint8_t *>(std::malloc(new_elf_size));
  if (!out) { *out_size = elf_size; return elf; }
  std::memcpy(out, elf, elf_size);

  uint64_t new_str_off = elf_size;
  std::memcpy(out + new_str_off, elf + strtab_offset, strtab_size);
  uint64_t npos = strtab_size;
  for (size_t i = 0; i < names.size(); ++i) {
    uint32_t st_name = static_cast<uint32_t>(npos);
    std::memcpy(entries[i].data(), &st_name, 4);
    std::memcpy(out + new_str_off + npos,
                names[i].c_str(), names[i].size() + 1);
    npos += names[i].size() + 1;
  }

  uint64_t new_sym_off = new_str_off + new_strtab_total;
  size_t old_sym_count = symtab_size / 24;
  size_t local_end = symtab_info;
  if (local_end > old_sym_count) local_end = old_sym_count;

  uint64_t wp = new_sym_off;
  std::memcpy(out + wp, elf + symtab_offset, local_end * 24);
  wp += local_end * 24;
  for (auto &e : entries) {
    std::memcpy(out + wp, e.data(), 24);
    wp += 24;
  }
  if (local_end < old_sym_count)
    std::memcpy(out + wp, elf + symtab_offset + local_end * 24,
                (old_sym_count - local_end) * 24);

  uint64_t cur_shoff;
  std::memcpy(&cur_shoff, out + 40, 8);
  uint8_t *new_str_sh = out + cur_shoff + symtab_link * e_shentsize;
  std::memcpy(new_str_sh + 24, &new_str_off, 8);
  uint64_t strsz64 = new_strtab_total;
  std::memcpy(new_str_sh + 32, &strsz64, 8);

  uint8_t *new_sym_sh = out + cur_shoff + symtab_idx * e_shentsize;
  std::memcpy(new_sym_sh + 24, &new_sym_off, 8);
  uint64_t symsz64 = new_symtab_total;
  std::memcpy(new_sym_sh + 32, &symsz64, 8);
  uint32_t new_info = static_cast<uint32_t>(local_end + entries.size());
  std::memcpy(new_sym_sh + 44, &new_info, 4);

  std::free(elf);
  *out_size = new_elf_size;
  return out;
}

struct DebugLineRow {
  uint64_t address;
  uint32_t file;
  int32_t line;
};

static std::vector<DebugLineRow> ScanDebugLineTable(
    const uint8_t *data, size_t data_size) {
  std::vector<DebugLineRow> rows;
  if (data_size < 15) return rows;

  uint32_t unit_length;
  std::memcpy(&unit_length, data, 4);
  size_t unit_end = 4 + unit_length;
  if (unit_end > data_size) unit_end = data_size;

  uint16_t version;
  std::memcpy(&version, data + 4, 2);

  uint32_t header_length;
  std::memcpy(&header_length, data + 6, 4);

  uint8_t min_inst_len = data[10];
  size_t hp = 11;
  if (version >= 4 && hp < unit_end) hp++;
  if (hp + 3 >= unit_end) return rows;
  int8_t line_base = static_cast<int8_t>(data[hp + 1]);
  uint8_t line_range = data[hp + 2];
  uint8_t opcode_base = data[hp + 3];
  hp += 4;
  if (line_range == 0) return rows;

  std::vector<uint8_t> std_lens(opcode_base > 0 ? opcode_base - 1 : 0);
  for (size_t i = 0; i < std_lens.size() && hp < unit_end; ++i)
    std_lens[i] = data[hp++];

  size_t prog_start = 10 + header_length;
  if (prog_start > unit_end) return rows;

  uint64_t addr = 0;
  uint32_t file = 1;
  int32_t line = 1;
  size_t pos = prog_start;

  while (pos < unit_end) {
    uint8_t op = data[pos++];
    if (op == 0) {
      if (pos >= unit_end) break;
      size_t nb;
      uint64_t ext_len = DecodeULEB128(data + pos, &nb);
      pos += nb;
      if (ext_len == 0 || pos + ext_len > unit_end) break;
      uint8_t ext_op = data[pos];
      if (ext_op == 1) {
        rows.push_back({addr, file, line});
        addr = 0; file = 1; line = 1;
      } else if (ext_op == 2) {
        if (ext_len >= 9)
          std::memcpy(&addr, data + pos + 1, 8);
        else if (ext_len >= 5) {
          uint32_t a; std::memcpy(&a, data + pos + 1, 4); addr = a;
        }
      }
      pos += ext_len;
    } else if (op < opcode_base) {
      switch (op) {
        case 1: rows.push_back({addr, file, line}); break;
        case 2: {
          size_t nb;
          addr += DecodeULEB128(data + pos, &nb) * min_inst_len;
          pos += nb; break;
        }
        case 3: {
          size_t nb;
          line += static_cast<int32_t>(DecodeSLEB128(data + pos, &nb));
          pos += nb; break;
        }
        case 4: {
          size_t nb;
          file = static_cast<uint32_t>(DecodeULEB128(data + pos, &nb));
          pos += nb; break;
        }
        case 5: { size_t nb; DecodeULEB128(data + pos, &nb); pos += nb; break; }
        case 6: case 7: break;
        case 8:
          addr += static_cast<uint64_t>((255 - opcode_base) / line_range) *
                  min_inst_len;
          break;
        case 9: {
          uint16_t a; std::memcpy(&a, data + pos, 2); pos += 2;
          addr += a; break;
        }
        default:
          if (op - 1 < static_cast<int>(std_lens.size()))
            for (uint8_t j = 0; j < std_lens[op - 1]; ++j) {
              size_t nb; DecodeULEB128(data + pos, &nb); pos += nb;
            }
          break;
      }
    } else {
      uint8_t adj = op - opcode_base;
      addr += static_cast<uint64_t>(adj / line_range) * min_inst_len;
      line += line_base + (adj % line_range);
      rows.push_back({addr, file, line});
    }
  }
  return rows;
}

static uint8_t *PatchDebugLine(
    uint8_t *elf, size_t elf_size,
    const std::vector<Trampoline> &trampolines,
    uint64_t text_size_before, uint64_t text_addr,
    size_t *out_size) {
  if (trampolines.empty()) { *out_size = elf_size; return elf; }

  uint8_t *dl_sh = FindSectionHeader(elf, elf_size, ".debug_line");
  if (!dl_sh) { *out_size = elf_size; return elf; }

  uint64_t dl_offset, dl_size;
  std::memcpy(&dl_offset, dl_sh + 24, 8);
  std::memcpy(&dl_size, dl_sh + 32, 8);
  if (dl_offset + dl_size > elf_size || dl_size < 15) {
    *out_size = elf_size; return elf;
  }

  auto rows = ScanDebugLineTable(elf + dl_offset, dl_size);

  auto FindLine = [&](uint64_t off) -> int32_t {
    uint64_t target = text_addr + off;
    int32_t best = 1;
    uint64_t best_addr = 0;
    for (auto &r : rows) {
      if (r.address <= target && r.address >= best_addr) {
        best_addr = r.address;
        best = r.line;
      }
    }
    return best;
  };

  std::vector<uint8_t> extra;
  uint64_t running = text_size_before;
  for (auto &t : trampolines) {
    uint64_t tramp_addr = text_addr + running;
    uint64_t tramp_end = tramp_addr + t.bytes.size();
    int32_t src_line = FindLine(t.original_offset);

    // DW_LNE_set_address(tramp_addr)
    extra.push_back(0x00); extra.push_back(0x09); extra.push_back(0x02);
    for (int b = 0; b < 8; ++b)
      extra.push_back(static_cast<uint8_t>(tramp_addr >> (b * 8)));

    if (src_line != 1) {
      extra.push_back(0x03); // DW_LNS_advance_line
      EncodeSLEB128(static_cast<int64_t>(src_line) - 1, extra);
    }

    extra.push_back(0x01); // DW_LNS_copy

    // DW_LNE_set_address(tramp_end)
    extra.push_back(0x00); extra.push_back(0x09); extra.push_back(0x02);
    for (int b = 0; b < 8; ++b)
      extra.push_back(static_cast<uint8_t>(tramp_end >> (b * 8)));

    // DW_LNE_end_sequence
    extra.push_back(0x00); extra.push_back(0x01); extra.push_back(0x01);

    running += t.bytes.size();
  }

  if (extra.empty()) { *out_size = elf_size; return elf; }

  size_t new_dl_size = dl_size + extra.size();
  size_t new_elf_size = elf_size + new_dl_size;
  uint8_t *out = static_cast<uint8_t *>(std::malloc(new_elf_size));
  if (!out) { *out_size = elf_size; return elf; }
  std::memcpy(out, elf, elf_size);

  uint64_t new_dl_off = elf_size;
  std::memcpy(out + new_dl_off, elf + dl_offset, dl_size);
  std::memcpy(out + new_dl_off + dl_size, extra.data(), extra.size());

  uint32_t old_ul;
  std::memcpy(&old_ul, out + new_dl_off, 4);
  uint32_t new_ul = old_ul + static_cast<uint32_t>(extra.size());
  std::memcpy(out + new_dl_off, &new_ul, 4);

  uint8_t *new_dl_sh = FindSectionHeader(out, new_elf_size, ".debug_line");
  if (new_dl_sh) {
    std::memcpy(new_dl_sh + 24, &new_dl_off, 8);
    uint64_t sz64 = new_dl_size;
    std::memcpy(new_dl_sh + 32, &sz64, 8);
  }

  std::free(elf);
  *out_size = new_elf_size;
  return out;
}

static void PatchDebugRanges(uint8_t *elf, size_t elf_size,
                              uint64_t text_addr, uint64_t text_size_before,
                              uint64_t tramp_total) {
  uint8_t *sh = FindSectionHeader(elf, elf_size, ".debug_ranges");
  if (!sh) return;
  uint64_t offset, size;
  std::memcpy(&offset, sh + 24, 8);
  std::memcpy(&size, sh + 32, 8);
  if (offset + size > elf_size) return;

  uint64_t text_end = text_addr + text_size_before;
  uint64_t new_text_end = text_end + tramp_total;
  uint8_t *d = elf + offset;
  for (size_t i = 0; i + 16 <= size; i += 16) {
    uint64_t s, e;
    std::memcpy(&s, d + i, 8);
    std::memcpy(&e, d + i + 8, 8);
    if (s == 0 && e == 0) continue;
    if (e > text_addr && e <= text_end) {
      uint64_t new_e = e + tramp_total;
      std::memcpy(d + i + 8, &new_e, 8);
    }
  }
}

static void PatchDebugInfo(uint8_t *elf, size_t elf_size,
                            uint64_t text_addr, uint64_t text_size_before,
                            uint64_t tramp_total) {
  uint8_t *sh = FindSectionHeader(elf, elf_size, ".debug_info");
  if (!sh) return;
  uint64_t offset, size;
  std::memcpy(&offset, sh + 24, 8);
  std::memcpy(&size, sh + 32, 8);
  if (offset + size > elf_size) return;

  uint8_t *d = elf + offset;
  uint64_t new_text_size = text_size_before + tramp_total;

  for (size_t i = 0; i + 12 <= size; ++i) {
    uint64_t v;
    std::memcpy(&v, d + i, 8);
    if (v != text_addr) continue;

    // DW_AT_low_pc = text_addr found; check for DW_AT_high_pc following it.
    // Try 4-byte form (DW_FORM_data4) — common in AMDGPU DWARF
    uint32_t hp4;
    std::memcpy(&hp4, d + i + 8, 4);
    if (hp4 > 0 && hp4 <= static_cast<uint32_t>(text_size_before)) {
      uint32_t ns4 = static_cast<uint32_t>(new_text_size);
      std::memcpy(d + i + 8, &ns4, 4);
      i += 11; continue;
    }
    // Try 8-byte size form (DW_FORM_data8)
    if (i + 16 <= size) {
      uint64_t hp8;
      std::memcpy(&hp8, d + i + 8, 8);
      if (hp8 > 0 && hp8 <= text_size_before) {
        std::memcpy(d + i + 8, &new_text_size, 8);
        i += 15; continue;
      }
      // Absolute address form
      if (hp8 > text_addr && hp8 <= text_addr + text_size_before) {
        uint64_t new_abs = text_addr + new_text_size;
        std::memcpy(d + i + 8, &new_abs, 8);
        i += 15; continue;
      }
    }
  }
}

static void PatchDebugFrame(uint8_t *elf, size_t elf_size,
                              uint64_t text_addr, uint64_t text_size_before,
                              uint64_t tramp_total) {
  uint8_t *sh = FindSectionHeader(elf, elf_size, ".debug_frame");
  if (!sh) return;
  uint64_t offset, size;
  std::memcpy(&offset, sh + 24, 8);
  std::memcpy(&size, sh + 32, 8);
  if (offset + size > elf_size) return;

  uint8_t *d = elf + offset;
  uint64_t new_text_size = text_size_before + tramp_total;
  size_t pos = 0;
  while (pos + 12 <= size) {
    uint32_t length;
    std::memcpy(&length, d + pos, 4);
    if (length == 0 || length == 0xFFFFFFFF) break;
    size_t entry_end = pos + 4 + length;
    if (entry_end > size) break;

    uint32_t cie_id;
    std::memcpy(&cie_id, d + pos + 4, 4);
    if (cie_id != 0xFFFFFFFF && length >= 20 && pos + 24 <= size) {
      uint64_t init_loc, addr_range;
      std::memcpy(&init_loc, d + pos + 8, 8);
      std::memcpy(&addr_range, d + pos + 16, 8);
      if (init_loc == text_addr && addr_range > 0 &&
          addr_range <= text_size_before) {
        std::memcpy(d + pos + 16, &new_text_size, 8);
      }
    }
    pos = entry_end;
  }
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

    size_t tramp_total = 0;
    for (auto &t : deferred) tramp_total += t.bytes.size();

    new_elf = AddTrampolineSymbols(new_elf, new_size, deferred,
                                   elf_info.text_size,
                                   elf_info.text_section_idx, &new_size);
    PatchDebugRanges(new_elf, new_size, elf_info.text_addr,
                     elf_info.text_size, tramp_total);
    PatchDebugInfo(new_elf, new_size, elf_info.text_addr,
                   elf_info.text_size, tramp_total);
    PatchDebugFrame(new_elf, new_size, elf_info.text_addr,
                    elf_info.text_size, tramp_total);
    new_elf = PatchDebugLine(new_elf, new_size, deferred,
                             elf_info.text_size, elf_info.text_addr,
                             &new_size);

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

// ═══════════════════════════════════════════════════════════════════════════════
// Cross-Family ISA Transpiler (GFX12 → GFX9)
// ═══════════════════════════════════════════════════════════════════════════════

// ── Mnemonic Translation Tables ──────────────────────────────────────────────

struct MnemonicMapping {
  const char* gfx12;
  const char* gfx9;
};

static const MnemonicMapping kGlobalMemMappings[] = {
    {"global_load_b32", "global_load_dword"},
    {"global_load_b64", "global_load_dwordx2"},
    {"global_load_b96", "global_load_dwordx3"},
    {"global_load_b128", "global_load_dwordx4"},
    {"global_load_u8", "global_load_ubyte"},
    {"global_load_i8", "global_load_sbyte"},
    {"global_load_u16", "global_load_ushort"},
    {"global_load_i16", "global_load_sshort"},
    {"global_load_d16_u8", "global_load_ubyte_d16"},
    {"global_load_d16_i8", "global_load_sbyte_d16"},
    {"global_load_d16_b16", "global_load_short_d16"},
    {"global_load_d16_hi_u8", "global_load_ubyte_d16_hi"},
    {"global_load_d16_hi_i8", "global_load_sbyte_d16_hi"},
    {"global_load_d16_hi_b16", "global_load_short_d16_hi"},
    {"global_store_b8", "global_store_byte"},
    {"global_store_b16", "global_store_short"},
    {"global_store_b32", "global_store_dword"},
    {"global_store_b64", "global_store_dwordx2"},
    {"global_store_b96", "global_store_dwordx3"},
    {"global_store_b128", "global_store_dwordx4"},
    {"global_load_addtid_b32", "global_load_dword_addtid"},
    {"global_store_addtid_b32", "global_store_dword_addtid"},
};

static const MnemonicMapping kFlatMemMappings[] = {
    {"flat_load_b32", "flat_load_dword"},
    {"flat_load_b64", "flat_load_dwordx2"},
    {"flat_load_b96", "flat_load_dwordx3"},
    {"flat_load_b128", "flat_load_dwordx4"},
    {"flat_load_u8", "flat_load_ubyte"},
    {"flat_load_i8", "flat_load_sbyte"},
    {"flat_load_u16", "flat_load_ushort"},
    {"flat_load_i16", "flat_load_sshort"},
    {"flat_load_d16_u8", "flat_load_ubyte_d16"},
    {"flat_load_d16_i8", "flat_load_sbyte_d16"},
    {"flat_load_d16_b16", "flat_load_short_d16"},
    {"flat_load_d16_hi_u8", "flat_load_ubyte_d16_hi"},
    {"flat_load_d16_hi_i8", "flat_load_sbyte_d16_hi"},
    {"flat_load_d16_hi_b16", "flat_load_short_d16_hi"},
    {"flat_store_b8", "flat_store_byte"},
    {"flat_store_b16", "flat_store_short"},
    {"flat_store_b32", "flat_store_dword"},
    {"flat_store_b64", "flat_store_dwordx2"},
    {"flat_store_b96", "flat_store_dwordx3"},
    {"flat_store_b128", "flat_store_dwordx4"},
};

static const MnemonicMapping kScratchMemMappings[] = {
    {"scratch_load_b32", "scratch_load_dword"},
    {"scratch_load_b64", "scratch_load_dwordx2"},
    {"scratch_load_b96", "scratch_load_dwordx3"},
    {"scratch_load_b128", "scratch_load_dwordx4"},
    {"scratch_load_u8", "scratch_load_ubyte"},
    {"scratch_load_i8", "scratch_load_sbyte"},
    {"scratch_load_u16", "scratch_load_ushort"},
    {"scratch_load_i16", "scratch_load_sshort"},
    {"scratch_load_d16_u8", "scratch_load_ubyte_d16"},
    {"scratch_load_d16_i8", "scratch_load_sbyte_d16"},
    {"scratch_load_d16_b16", "scratch_load_short_d16"},
    {"scratch_load_d16_hi_u8", "scratch_load_ubyte_d16_hi"},
    {"scratch_load_d16_hi_i8", "scratch_load_sbyte_d16_hi"},
    {"scratch_load_d16_hi_b16", "scratch_load_short_d16_hi"},
    {"scratch_store_b8", "scratch_store_byte"},
    {"scratch_store_b16", "scratch_store_short"},
    {"scratch_store_b32", "scratch_store_dword"},
    {"scratch_store_b64", "scratch_store_dwordx2"},
    {"scratch_store_b96", "scratch_store_dwordx3"},
    {"scratch_store_b128", "scratch_store_dwordx4"},
};

static const MnemonicMapping kBufferMemMappings[] = {
    {"buffer_load_b32", "buffer_load_dword"},
    {"buffer_load_b64", "buffer_load_dwordx2"},
    {"buffer_load_b96", "buffer_load_dwordx3"},
    {"buffer_load_b128", "buffer_load_dwordx4"},
    {"buffer_load_u8", "buffer_load_ubyte"},
    {"buffer_load_i8", "buffer_load_sbyte"},
    {"buffer_load_u16", "buffer_load_ushort"},
    {"buffer_load_i16", "buffer_load_sshort"},
    {"buffer_load_d16_u8", "buffer_load_ubyte_d16"},
    {"buffer_load_d16_i8", "buffer_load_sbyte_d16"},
    {"buffer_load_d16_b16", "buffer_load_short_d16"},
    {"buffer_load_d16_hi_u8", "buffer_load_ubyte_d16_hi"},
    {"buffer_load_d16_hi_i8", "buffer_load_sbyte_d16_hi"},
    {"buffer_load_d16_hi_b16", "buffer_load_short_d16_hi"},
    {"buffer_store_b8", "buffer_store_byte"},
    {"buffer_store_b16", "buffer_store_short"},
    {"buffer_store_b32", "buffer_store_dword"},
    {"buffer_store_b64", "buffer_store_dwordx2"},
    {"buffer_store_b96", "buffer_store_dwordx3"},
    {"buffer_store_b128", "buffer_store_dwordx4"},
};

static const MnemonicMapping kDSMappings[] = {
    {"ds_load_b32", "ds_read_b32"},
    {"ds_load_b64", "ds_read_b64"},
    {"ds_load_b96", "ds_read_b96"},
    {"ds_load_b128", "ds_read_b128"},
    {"ds_load_u8", "ds_read_u8"},
    {"ds_load_i8", "ds_read_i8"},
    {"ds_load_u16", "ds_read_u16"},
    {"ds_load_i16", "ds_read_i16"},
    {"ds_load_2addr_b32", "ds_read2_b32"},
    {"ds_load_2addr_stride64_b32", "ds_read2st64_b32"},
    {"ds_load_2addr_b64", "ds_read2_b64"},
    {"ds_load_2addr_stride64_b64", "ds_read2st64_b64"},
    {"ds_load_u8_d16", "ds_read_u8_d16"},
    {"ds_load_i8_d16", "ds_read_i8_d16"},
    {"ds_load_u16_d16", "ds_read_u16_d16"},
    {"ds_load_u8_d16_hi", "ds_read_u8_d16_hi"},
    {"ds_load_i8_d16_hi", "ds_read_i8_d16_hi"},
    {"ds_load_u16_d16_hi", "ds_read_u16_d16_hi"},
    {"ds_load_addtid_b32", "ds_read_addtid_b32"},
    {"ds_store_b32", "ds_write_b32"},
    {"ds_store_b64", "ds_write_b64"},
    {"ds_store_b128", "ds_write_b128"},
    {"ds_store_2addr_b32", "ds_write2_b32"},
    {"ds_store_2addr_stride64_b32", "ds_write2st64_b32"},
    {"ds_store_b8", "ds_write_b8"},
    {"ds_store_b16", "ds_write_b16"},
};

static const MnemonicMapping kSMEMMappings[] = {
    {"s_load_b32", "s_load_dword"},
    {"s_load_b64", "s_load_dwordx2"},
    {"s_load_b96", "s_load_dwordx4"},
    {"s_load_b128", "s_load_dwordx4"},
    {"s_load_b256", "s_load_dwordx8"},
    {"s_load_b512", "s_load_dwordx16"},
    {"s_store_b32", "s_store_dword"},
    {"s_store_b64", "s_store_dwordx2"},
    {"s_store_b128", "s_store_dwordx4"},
    {"s_buffer_load_b32", "s_buffer_load_dword"},
    {"s_buffer_load_b64", "s_buffer_load_dwordx2"},
    {"s_buffer_load_b128", "s_buffer_load_dwordx4"},
    {"s_buffer_load_b256", "s_buffer_load_dwordx8"},
    {"s_buffer_load_b512", "s_buffer_load_dwordx16"},
};

static const MnemonicMapping kScalarALURenames[] = {
    {"s_add_co_u32", "s_add_u32"},
    {"s_sub_co_u32", "s_sub_u32"},
    {"s_add_co_ci_u32", "s_addc_u32"},
    {"s_sub_co_ci_u32", "s_subb_u32"},
    {"s_add_co_i32", "s_add_i32"},
    {"s_sub_co_i32", "s_sub_i32"},
    {"s_and_not1_b32", "s_andn2_b32"},
    {"s_and_not1_b64", "s_andn2_b64"},
    {"s_or_not1_b32", "s_orn2_b32"},
    {"s_or_not1_b64", "s_orn2_b64"},
};

static const MnemonicMapping kVALURenames[] = {
    {"v_max_num_f32", "v_max_f32"},
    {"v_min_num_f32", "v_min_f32"},
    {"v_max_num_f16", "v_max_f16"},
    {"v_min_num_f16", "v_min_f16"},
    {"v_max_num_f64", "v_max_f64"},
    {"v_min_num_f64", "v_min_f64"},
    {"v_maxmin_num_f32", "v_maxmin_f32"},
    {"v_minmax_num_f32", "v_minmax_f32"},
    {"v_maxmin_num_f16", "v_maxmin_f16"},
    {"v_minmax_num_f16", "v_minmax_f16"},
    {"v_add_nc_u32", "v_add_u32"},
    {"v_sub_nc_u32", "v_sub_u32"},
    {"v_add_nc_i32", "v_add_i32"},
    {"v_sub_nc_i32", "v_sub_i32"},
};

static const MnemonicMapping kGlobalAtomicRenames[] = {
    {"global_atomic_add_u32", "global_atomic_add"},
    {"global_atomic_sub_u32", "global_atomic_sub"},
    {"global_atomic_and_b32", "global_atomic_and"},
    {"global_atomic_or_b32", "global_atomic_or"},
    {"global_atomic_xor_b32", "global_atomic_xor"},
    {"global_atomic_min_i32", "global_atomic_smin"},
    {"global_atomic_max_i32", "global_atomic_smax"},
    {"global_atomic_min_u32", "global_atomic_umin"},
    {"global_atomic_max_u32", "global_atomic_umax"},
    {"global_atomic_swap_b32", "global_atomic_swap"},
    {"global_atomic_cmpswap_b32", "global_atomic_cmpswap"},
    {"global_atomic_add_u64", "global_atomic_add_x2"},
    {"global_atomic_sub_u64", "global_atomic_sub_x2"},
    {"global_atomic_and_b64", "global_atomic_and_x2"},
    {"global_atomic_or_b64", "global_atomic_or_x2"},
    {"global_atomic_xor_b64", "global_atomic_xor_x2"},
    {"global_atomic_swap_b64", "global_atomic_swap_x2"},
    {"global_atomic_cmpswap_b64", "global_atomic_cmpswap_x2"},
    {"global_atomic_add_f32", "global_atomic_add_f32"},
    {"global_atomic_pk_add_f16", "global_atomic_pk_add_f16"},
};

static const MnemonicMapping kFlatAtomicRenames[] = {
    {"flat_atomic_add_u32", "flat_atomic_add"},
    {"flat_atomic_sub_u32", "flat_atomic_sub"},
    {"flat_atomic_and_b32", "flat_atomic_and"},
    {"flat_atomic_or_b32", "flat_atomic_or"},
    {"flat_atomic_xor_b32", "flat_atomic_xor"},
    {"flat_atomic_min_i32", "flat_atomic_smin"},
    {"flat_atomic_max_i32", "flat_atomic_smax"},
    {"flat_atomic_min_u32", "flat_atomic_umin"},
    {"flat_atomic_max_u32", "flat_atomic_umax"},
    {"flat_atomic_swap_b32", "flat_atomic_swap"},
    {"flat_atomic_cmpswap_b32", "flat_atomic_cmpswap"},
    {"flat_atomic_add_u64", "flat_atomic_add_x2"},
    {"flat_atomic_sub_u64", "flat_atomic_sub_x2"},
    {"flat_atomic_swap_b64", "flat_atomic_swap_x2"},
    {"flat_atomic_cmpswap_b64", "flat_atomic_cmpswap_x2"},
};

static const MnemonicMapping kDSAtomicRenames[] = {
    {"ds_add_u32", "ds_add_u32"},
    {"ds_add_rtn_u32", "ds_add_rtn_u32"},
    {"ds_cmpstore_b32", "ds_cmpst_b32"},
    {"ds_cmpstore_rtn_b32", "ds_cmpst_rtn_b32"},
    {"ds_cmpstore_b64", "ds_cmpst_b64"},
    {"ds_cmpstore_rtn_b64", "ds_cmpst_rtn_b64"},
};

static std::unordered_map<std::string, std::string> BuildMnemonicMap() {
  std::unordered_map<std::string, std::string> map;
  auto addMappings = [&](const MnemonicMapping* mappings, size_t count) {
    for (size_t i = 0; i < count; ++i)
      map[mappings[i].gfx12] = mappings[i].gfx9;
  };
  addMappings(kGlobalMemMappings, sizeof(kGlobalMemMappings) / sizeof(kGlobalMemMappings[0]));
  addMappings(kFlatMemMappings, sizeof(kFlatMemMappings) / sizeof(kFlatMemMappings[0]));
  addMappings(kScratchMemMappings, sizeof(kScratchMemMappings) / sizeof(kScratchMemMappings[0]));
  addMappings(kBufferMemMappings, sizeof(kBufferMemMappings) / sizeof(kBufferMemMappings[0]));
  addMappings(kDSMappings, sizeof(kDSMappings) / sizeof(kDSMappings[0]));
  addMappings(kSMEMMappings, sizeof(kSMEMMappings) / sizeof(kSMEMMappings[0]));
  addMappings(kScalarALURenames, sizeof(kScalarALURenames) / sizeof(kScalarALURenames[0]));
  addMappings(kVALURenames, sizeof(kVALURenames) / sizeof(kVALURenames[0]));
  addMappings(kGlobalAtomicRenames, sizeof(kGlobalAtomicRenames) / sizeof(kGlobalAtomicRenames[0]));
  addMappings(kFlatAtomicRenames, sizeof(kFlatAtomicRenames) / sizeof(kFlatAtomicRenames[0]));
  addMappings(kDSAtomicRenames, sizeof(kDSAtomicRenames) / sizeof(kDSAtomicRenames[0]));
  return map;
}

static const std::unordered_map<std::string, std::string>& GetMnemonicMap() {
  static auto map = BuildMnemonicMap();
  return map;
}

// ── Wave32→Wave64 EXEC Patterns ─────────────────────────────────────────────

static bool WritesExecLo(const std::string& line) {
  size_t mnem_end = line.find_first_of(" \t");
  if (mnem_end == std::string::npos) return false;
  size_t op_start = line.find_first_not_of(" \t,", mnem_end);
  if (op_start == std::string::npos) return false;
  if (line.compare(op_start, 7, "exec_lo") == 0) return true;
  std::string mnemonic = line.substr(0, mnem_end);
  if (mnemonic.find("saveexec_b32") != std::string::npos) return true;
  return false;
}

// ── Wait Counter Translation ─────────────────────────────────────────────────

static bool IsWaitInstruction(const std::string& mnemonic) {
  return mnemonic == "s_wait_loadcnt" || mnemonic == "s_wait_storecnt" ||
         mnemonic == "s_wait_samplecnt" || mnemonic == "s_wait_bvhcnt" ||
         mnemonic == "s_wait_expcnt" || mnemonic == "s_wait_dscnt" ||
         mnemonic == "s_wait_kmcnt" || mnemonic == "s_wait_loadcnt_dscnt" ||
         mnemonic == "s_wait_storecnt_dscnt" || mnemonic == "s_wait_xcnt" ||
         mnemonic == "s_wait_asynccnt" || mnemonic == "s_wait_tensorcnt";
}

static std::string TranslateWaitInstruction(const std::string& line) {
  std::string mnemonic;
  int count = 0;
  std::istringstream iss(line);
  iss >> mnemonic >> count;
  if (iss.fail()) count = 0;

  if (mnemonic == "s_wait_loadcnt" || mnemonic == "s_wait_samplecnt" ||
      mnemonic == "s_wait_bvhcnt" || mnemonic == "s_wait_storecnt")
    return "s_waitcnt vmcnt(" + std::to_string(count) + ")";
  if (mnemonic == "s_wait_dscnt" || mnemonic == "s_wait_kmcnt")
    return "s_waitcnt lgkmcnt(" + std::to_string(count) + ")";
  if (mnemonic == "s_wait_expcnt")
    return "s_waitcnt expcnt(" + std::to_string(count) + ")";
  if (mnemonic == "s_wait_loadcnt_dscnt")
    return "s_waitcnt vmcnt(" + std::to_string(count) +
           ") lgkmcnt(" + std::to_string(count) + ")";
  if (mnemonic == "s_wait_storecnt_dscnt")
    return "s_waitcnt vmcnt(" + std::to_string(count) +
           ") lgkmcnt(" + std::to_string(count) + ")";
  return "s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)";
}

// ── Unsupported Instruction Detection ────────────────────────────────────────

static bool IsUnsupportedOnGFX9(const std::string& mnemonic) {
  if (mnemonic.find("tensor_") == 0) return true;
  if (mnemonic.find("cluster_") == 0) return true;
  if (mnemonic.find("_prefetch_") != std::string::npos) return true;
  if (mnemonic.find("v_permlane16") == 0) return true;
  if (mnemonic.find("v_permlanex16") == 0) return true;
  if (mnemonic == "s_wait_alu") return true;
  if (mnemonic == "s_delay_alu") return true;
  return false;
}

// ── VCC Register Width Translation ───────────────────────────────────────────

static std::string WidenVccReferences(const std::string& line) {
  std::string result = line;
  size_t mnem_end = result.find_first_of(" \t");
  if (mnem_end == std::string::npos) return result;
  std::string operands = result.substr(mnem_end);
  size_t pos = 0;
  while ((pos = operands.find("vcc_lo", pos)) != std::string::npos) {
    size_t end = pos + 6;
    if (end < operands.size() && (std::isalnum(operands[end]) || operands[end] == '_')) {
      pos = end;
      continue;
    }
    operands.replace(pos, 6, "vcc");
    pos += 3;
  }
  return result.substr(0, mnem_end) + operands;
}

// ── EXEC Width Widening ──────────────────────────────────────────────────────

static std::vector<std::string> WidenExecOperation(const std::string& line, bool compact_mode = false) {
  std::vector<std::string> result;
  std::string mnemonic = line.substr(0, line.find_first_of(" \t"));

  if (mnemonic.find("saveexec_b32") != std::string::npos) {
    std::string b64_mnem = mnemonic;
    size_t b32_pos = b64_mnem.find("_b32");
    b64_mnem.replace(b32_pos, 4, "_b64");
    size_t not1_pos = b64_mnem.find("_not1_");
    if (not1_pos != std::string::npos)
      b64_mnem.replace(not1_pos, 6, "n2_");

    std::string ops_part = line.substr(line.find_first_of(" \t"));
    size_t op_start = ops_part.find_first_not_of(" \t");
    if (op_start != std::string::npos) {
      std::string ops = ops_part.substr(op_start);
      size_t comma = ops.find(',');
      if (comma != std::string::npos) {
        std::string dst = ops.substr(0, comma);
        size_t ds = dst.find_first_not_of(" \t");
        size_t de = dst.find_last_not_of(" \t");
        dst = dst.substr(ds, de - ds + 1);
        std::string src = ops.substr(comma + 1);
        size_t ss = src.find_first_not_of(" \t");
        src = src.substr(ss);

        size_t vcc_pos = src.find("vcc_lo");
        if (vcc_pos != std::string::npos)
          src.replace(vcc_pos, 6, "vcc");

        if (dst[0] == 's' && std::isdigit(dst[1])) {
          int reg_num = std::stoi(dst.substr(1));
          std::string src32 = src;
          if (src32 == "vcc") src32 = "vcc_lo";
          result.push_back("s_mov_b32 " + dst + ", exec_lo");
          bool is_or = (b64_mnem.find("s_or_saveexec") == 0);
          if (is_or)
            result.push_back("s_or_b32 exec_lo, exec_lo, " + src32);
          else if (b64_mnem.find("andn2") != std::string::npos)
            result.push_back("s_andn2_b32 exec_lo, exec_lo, " + src32);
          else
            result.push_back("s_and_b32 exec_lo, exec_lo, " + src32);
          if (is_or || !compact_mode)
            result.push_back("s_mov_b32 exec_hi, 0");
          return result;
        }
        result.push_back(b64_mnem + " " + dst + ", " + src);
        result.push_back("s_mov_b32 exec_hi, 0");
        return result;
      }
    }
    result.push_back(b64_mnem + ops_part);
    result.push_back("s_mov_b32 exec_hi, 0");
    return result;
  }

  result.push_back(line);
  if (WritesExecLo(line))
    result.push_back("s_mov_b32 exec_hi, 0");
  return result;
}

// ── Operand Syntax Translation ───────────────────────────────────────────────

static std::string TranslateOperandSyntax(const std::string& line,
                                           const std::string& mnemonic) {
  std::string result = line;
  {
    size_t pos = result.find("scope:");
    if (pos != std::string::npos) {
      size_t end = result.find_first_of(" \t,", pos);
      if (end == std::string::npos) end = result.size();
      result.erase(pos, end - pos);
    }
  }
  {
    size_t pos = result.find("th:");
    if (pos != std::string::npos) {
      size_t end = result.find_first_of(" \t,", pos);
      if (end == std::string::npos) end = result.size();
      std::string th_value = result.substr(pos, end - pos);
      if (th_value.find("TH_ATOMIC_RETURN") != std::string::npos)
        result.replace(pos, end - pos, "sc0");
      else
        result.erase(pos, end - pos);
    }
  }
  {
    size_t pos = result.find(" nv");
    while (pos != std::string::npos) {
      size_t end = pos + 3;
      if (end >= result.size() || result[end] == ' ' || result[end] == '\t' ||
          result[end] == ',' || result[end] == '\0') {
        result.erase(pos, end - pos);
      } else {
        pos = result.find(" nv", pos + 1);
        continue;
      }
      pos = result.find(" nv", pos);
    }
  }
  {
    size_t pos = result.find("scale_offset");
    if (pos != std::string::npos) {
      size_t end = pos + 12;
      if (pos > 0 && (result[pos-1] == ' ' || result[pos-1] == ',')) --pos;
      result.erase(pos, end - pos);
    }
  }
  while (!result.empty() && (result.back() == ' ' || result.back() == '\t' ||
                              result.back() == ','))
    result.pop_back();
  return result;
}

// ── Extract/Replace Mnemonic ─────────────────────────────────────────────────

static std::string TranspileExtractMnemonic(const std::string& line) {
  size_t start = line.find_first_not_of(" \t");
  if (start == std::string::npos) return "";
  size_t end = line.find_first_of(" \t", start);
  if (end == std::string::npos) return line.substr(start);
  return line.substr(start, end - start);
}

static std::string TranspileReplaceMnemonic(const std::string& line,
                                             const std::string& old_mnemonic,
                                             const std::string& new_mnemonic) {
  size_t pos = line.find(old_mnemonic);
  if (pos == std::string::npos) return line;
  std::string result = line;
  result.replace(pos, old_mnemonic.size(), new_mnemonic);
  return result;
}

// ── TTMP Taint Analysis ──────────────────────────────────────────────────────

enum class RegKind { SGPR, VGPR, TTMP, SCC, VCC, EXEC, Other };

static RegKind ClassifyReg(unsigned reg, const llvm::MCRegisterInfo& MRI) {
  const char* name = MRI.getName(reg);
  if (!name) return RegKind::Other;
  if (strncmp(name, "TTMP", 4) == 0) return RegKind::TTMP;
  if (strncmp(name, "SGPR", 4) == 0) return RegKind::SGPR;
  if (strncmp(name, "VGPR", 4) == 0) return RegKind::VGPR;
  if (strcmp(name, "SCC") == 0) return RegKind::SCC;
  if (strncmp(name, "VCC", 3) == 0) return RegKind::VCC;
  if (strncmp(name, "EXEC", 4) == 0) return RegKind::EXEC;
  return RegKind::Other;
}

static bool IsRegTainted(unsigned reg, const std::set<unsigned>& tainted,
                         const llvm::MCRegisterInfo& MRI) {
  if (tainted.count(reg)) return true;
  for (auto sub : MRI.subregs(reg))
    if (tainted.count(sub)) return true;
  for (auto sup : MRI.superregs(reg))
    if (tainted.count(sup)) return true;
  return false;
}

static void TaintReg(unsigned reg, std::set<unsigned>& tainted,
                     const llvm::MCRegisterInfo& MRI) {
  tainted.insert(reg);
  for (auto sub : MRI.subregs(reg))
    tainted.insert(sub);
}

static void UntaintReg(unsigned reg, std::set<unsigned>& tainted,
                       const llvm::MCRegisterInfo& MRI) {
  tainted.erase(reg);
  for (auto sub : MRI.subregs(reg))
    tainted.erase(sub);
  for (auto sup : MRI.superregs(reg))
    tainted.erase(sup);
}

enum class TaintAction { Keep, Skip, Replace };

struct TaintResult {
  TaintAction action;
  std::string replace_dst;
  std::string replace_src;
};

static void GetInstRegs(const llvm::MCInst& inst,
                        const llvm::MCInstrInfo& MCII,
                        const llvm::MCRegisterInfo& MRI,
                        std::vector<unsigned>& defs,
                        std::vector<unsigned>& uses) {
  const llvm::MCInstrDesc& desc = MCII.get(inst.getOpcode());
  unsigned num_defs = desc.getNumDefs();
  for (unsigned i = 0; i < inst.getNumOperands(); ++i) {
    const auto& op = inst.getOperand(i);
    if (!op.isReg() || op.getReg() == 0) continue;
    if (i < num_defs)
      defs.push_back(op.getReg());
    else
      uses.push_back(op.getReg());
  }
  for (auto imp : desc.implicit_defs())
    defs.push_back(imp);
  for (auto imp : desc.implicit_uses())
    uses.push_back(imp);
}

struct SourceInstrForTaint {
  std::string text;
  llvm::MCInst inst;
  bool valid_inst;
};

static std::vector<TaintResult> AnalyzeTTMPTaint(
    const std::vector<SourceInstrForTaint>& instrs,
    const llvm::MCInstrInfo& MCII,
    const llvm::MCRegisterInfo& MRI) {

  std::vector<TaintResult> results;
  results.reserve(instrs.size());
  std::set<unsigned> tainted;
  bool dump = std::getenv("HSA_HOTSWAP_DUMP") != nullptr;

  for (size_t i = 0; i < instrs.size(); ++i) {
    const auto& si = instrs[i];
    const auto& text = si.text;
    std::string mnemonic = TranspileExtractMnemonic(text);

    TaintResult tr;
    tr.action = TaintAction::Keep;

    if (!si.valid_inst) {
      results.push_back(tr);
      continue;
    }

    if (mnemonic.empty() || mnemonic[0] != 's' ||
        mnemonic.find("s_cbranch_") == 0 || mnemonic == "s_branch" ||
        mnemonic == "s_endpgm" || mnemonic == "s_barrier" ||
        mnemonic.find("s_barrier_") == 0 || mnemonic == "s_nop" ||
        mnemonic == "s_waitcnt" || mnemonic.find("s_wait_") == 0 ||
        mnemonic == "s_clause" || mnemonic == "s_delay_alu" ||
        mnemonic == "s_wait_alu" || mnemonic == "s_code_end" ||
        mnemonic == "s_set_inst_prefetch_distance") {
      results.push_back(tr);
      continue;
    }

    std::vector<unsigned> defs, uses;
    GetInstRegs(si.inst, MCII, MRI, defs, uses);

    bool uses_ttmp = false;
    for (auto r : uses)
      if (ClassifyReg(r, MRI) == RegKind::TTMP) { uses_ttmp = true; break; }
    bool defs_ttmp = false;
    for (auto r : defs)
      if (ClassifyReg(r, MRI) == RegKind::TTMP) { defs_ttmp = true; break; }

    if (mnemonic == "s_getreg_b32" &&
        text.find("HW_REG_IB_STS2") != std::string::npos) {
      tr.action = TaintAction::Skip;
      for (auto r : defs) TaintReg(r, tainted, MRI);
      if (dump) std::cerr << "hotswap: taint: SKIP (HW_REG_IB_STS2): " << text << "\n";
      results.push_back(tr);
      continue;
    }

    if ((mnemonic == "s_setreg_imm32_b32" || mnemonic == "s_setreg_b32") &&
        text.find("HW_REG_WAVE_MODE") != std::string::npos) {
      tr.action = TaintAction::Skip;
      if (dump) std::cerr << "hotswap: taint: SKIP (HW_REG_WAVE_MODE): " << text << "\n";
      results.push_back(tr);
      continue;
    }

    if (uses_ttmp || defs_ttmp) {
      if (mnemonic == "s_cselect_b32") {
        tr.action = TaintAction::Replace;
        size_t op_start = text.find(mnemonic) + mnemonic.size();
        std::string ops = text.substr(op_start);
        size_t s = ops.find_first_not_of(" \t");
        size_t e = ops.find_first_of(" \t,", s);
        if (s != std::string::npos)
          tr.replace_dst = ops.substr(s, e != std::string::npos ? e - s : std::string::npos);
        tr.replace_src = (text.find("ttmp9") != std::string::npos) ? "v5" : "v4";
        for (auto r : defs) UntaintReg(r, tainted, MRI);
        for (auto r : uses) {
          if (ClassifyReg(r, MRI) == RegKind::SCC)
            UntaintReg(r, tainted, MRI);
        }
        if (dump) std::cerr << "hotswap: taint: REPLACE (s_cselect ttmp → "
                            << tr.replace_dst << " = " << tr.replace_src << "): " << text << "\n";
      } else {
        tr.action = TaintAction::Skip;
        for (auto r : defs) {
          RegKind kind = ClassifyReg(r, MRI);
          if (kind == RegKind::SGPR || kind == RegKind::SCC)
            TaintReg(r, tainted, MRI);
        }
        if (dump) std::cerr << "hotswap: taint: SKIP (direct TTMP): " << text << "\n";
      }
      results.push_back(tr);
      continue;
    }

    if (mnemonic.find("s_load_") == 0 || mnemonic.find("s_buffer_load_") == 0) {
      for (auto r : defs) UntaintReg(r, tainted, MRI);
      if (dump && !tainted.empty())
        std::cerr << "hotswap: taint: KEEP (s_load clears taint on defs): " << text << "\n";
      results.push_back(tr);
      continue;
    }

    if (mnemonic.find("s_cmp_") == 0) {
      bool any_tainted = false;
      for (auto r : uses)
        if (IsRegTainted(r, tainted, MRI)) { any_tainted = true; break; }
      if (any_tainted) {
        tr.action = TaintAction::Skip;
        for (auto r : defs) TaintReg(r, tainted, MRI);
        if (dump) std::cerr << "hotswap: taint: SKIP (s_cmp tainted): " << text << "\n";
        results.push_back(tr);
        continue;
      }
      results.push_back(tr);
      continue;
    }

    bool has_tainted_src = false;
    bool has_untainted_sgpr_src = false;
    for (auto r : uses) {
      RegKind kind = ClassifyReg(r, MRI);
      if (kind == RegKind::SGPR || kind == RegKind::SCC) {
        if (IsRegTainted(r, tainted, MRI))
          has_tainted_src = true;
        else
          has_untainted_sgpr_src = true;
      }
    }

    if (has_tainted_src && !has_untainted_sgpr_src) {
      tr.action = TaintAction::Skip;
      for (auto r : defs) {
        RegKind kind = ClassifyReg(r, MRI);
        if (kind == RegKind::SGPR || kind == RegKind::SCC)
          TaintReg(r, tainted, MRI);
      }
      if (dump) std::cerr << "hotswap: taint: SKIP (all srcs tainted): " << text << "\n";
      results.push_back(tr);
      continue;
    }

    if (has_tainted_src && has_untainted_sgpr_src) {
      for (auto r : defs) UntaintReg(r, tainted, MRI);
      if (dump) std::cerr << "hotswap: taint: KEEP (mixed taint, clear defs): " << text << "\n";
      results.push_back(tr);
      continue;
    }

    if (!tainted.empty()) {
      for (auto r : defs) UntaintReg(r, tainted, MRI);
    }
    results.push_back(tr);
  }

  return results;
}

// ── Kernel Descriptor Patching for Wave64 ────────────────────────────────────

static void PatchKernelDescriptorsForWave64(uint8_t* elf, size_t elf_size,
                                             const ElfInfo& info) {
  if (info.text_idx < 0) return;
  const uint8_t* text = elf + info.text_offset;

  for (uint64_t offset = 0; offset + 64 <= info.text_size; offset += 256) {
    uint64_t entry_offset;
    std::memcpy(&entry_offset, text + offset + 16, 8);
    if (entry_offset != 256) continue;

    uint32_t rsrc1;
    std::memcpy(&rsrc1, text + offset + 48, 4);
    rsrc1 &= 0x00FFFFFFu;
    rsrc1 |= (1u << 21) | (1u << 23);

    uint32_t vgpr_field12 = rsrc1 & 0x3Fu;
    uint32_t sgpr_field12_kd = (rsrc1 >> 6) & 0x3Fu;
    uint32_t num_vgprs = (vgpr_field12 + 1u) * 12u;
    if (num_vgprs < 8u) num_vgprs = 8u;
    num_vgprs += 4u;
    uint32_t gfx9_vgpr = (num_vgprs / 4u) - 1u;
    if (gfx9_vgpr > 62u) gfx9_vgpr = 62u;
    uint32_t rsrc1_vgpr_field = gfx9_vgpr + 1u;
    if (rsrc1_vgpr_field > 63u) rsrc1_vgpr_field = 63u;
    rsrc1 &= ~0xFFFu;
    rsrc1 |= (rsrc1_vgpr_field << 6u);
    {
      uint32_t orig_sgpr_field = sgpr_field12_kd;
      uint32_t num_sgprs = (orig_sgpr_field + 1u) * 16u + 8u;
      const char* sgpr_key = ".sgpr_count";
      for (size_t i = 0; i + 12 < elf_size; i++) {
        if (std::memcmp(elf + i, sgpr_key, 11) == 0) {
          uint8_t val = elf[i + 11];
          uint32_t sc = (val <= 0x7F) ? val : (val == 0xCC ? elf[i+12] : 0);
          if (sc + 8 > num_sgprs) num_sgprs = sc + 8;
          break;
        }
      }
      uint32_t gfx9_sgpr = (num_sgprs / 8u) - 1u;
      if (gfx9_sgpr > 12u) gfx9_sgpr = 12u;
      rsrc1 |= gfx9_sgpr;
    }
    std::memcpy(elf + info.text_offset + offset + 48, &rsrc1, 4);

    uint32_t rsrc2;
    std::memcpy(&rsrc2, text + offset + 52, 4);
    rsrc2 |= (1u << 7);
    rsrc2 |= (1u << 8);
    rsrc2 |= (1u << 9);
    std::memcpy(elf + info.text_offset + offset + 52, &rsrc2, 4);

    uint16_t props;
    std::memcpy(&props, text + offset + 56, 2);
    props &= ~(1u << 10);
    std::memcpy(elf + info.text_offset + offset + 56, &props, 2);

    uint32_t rsrc3 = gfx9_vgpr;
    std::memcpy(elf + info.text_offset + offset + 44, &rsrc3, 4);
  }
}

// ── ELF Metadata Patching ────────────────────────────────────────────────────

static void PatchElfMetadata(uint8_t* elf, size_t elf_size,
                              const std::string& target_cpu) {
  uint32_t e_flags;
  std::memcpy(&e_flags, elf + 48, 4);
  uint8_t target_mach = 0;
  if (target_cpu == "gfx950") target_mach = 0x4f;
  else if (target_cpu == "gfx942") target_mach = 0x4c;
  else if (target_cpu == "gfx90a") target_mach = 0x42;
  if (target_mach != 0) {
    e_flags = (e_flags & ~0xFFu) | target_mach;
    std::memcpy(elf + 48, &e_flags, 4);
  }

  std::string old_isa_full = "amdgcn-amd-amdhsa--gfx1250";
  std::string new_isa_full = "amdgcn-amd-amdhsa--" + target_cpu;
  for (size_t i = 0; i + old_isa_full.size() <= elf_size; ++i) {
    if (std::memcmp(elf + i, old_isa_full.data(), old_isa_full.size()) == 0) {
      if (new_isa_full.size() <= old_isa_full.size()) {
        std::memcpy(elf + i, new_isa_full.data(), new_isa_full.size());
        for (size_t j = new_isa_full.size(); j < old_isa_full.size(); ++j)
          elf[i + j] = ' ';
      }
    }
  }

  for (size_t i = 0; i + 7 <= elf_size; ++i) {
    if (std::memcmp(elf + i, "gfx1250", 7) == 0) {
      if (target_cpu.size() <= 7) {
        std::memcpy(elf + i, target_cpu.c_str(), target_cpu.size());
        for (size_t j = target_cpu.size(); j < 7; ++j)
          elf[i + j] = '0';
      }
    }
  }

  {
    const char* wf_key = ".wavefront_size";
    size_t wf_key_len = 15;
    for (size_t i = 0; i + wf_key_len + 1 <= elf_size; ++i) {
      if (std::memcmp(elf + i, wf_key, wf_key_len) == 0) {
        uint8_t val = elf[i + wf_key_len];
        if (val == 0x20) {
          elf[i + wf_key_len] = 0x40;
          std::cerr << "hotswap: transpile: patched wavefront_size 32 → 64\n";
        }
      }
    }
  }
}

// ── TranslateInstruction ─────────────────────────────────────────────────────

static std::vector<std::string> TranslateInstruction(const std::string& asm_line,
                                               const std::string& source_cpu,
                                               const std::string& target_cpu,
                                               int scale_temp_vgpr = 7,
                                               int cmpx_temp_sgpr = 16,
                                               bool compact_mode = false) {
  std::vector<std::string> result;
  std::string line = asm_line;

  size_t start = line.find_first_not_of(" \t");
  if (start == std::string::npos) {
    result.push_back(line);
    return result;
  }
  if (start > 0) line = line.substr(start);

  size_t comment = line.find("//");
  if (comment != std::string::npos) {
    line = line.substr(0, comment);
    size_t end = line.find_last_not_of(" \t");
    if (end != std::string::npos)
      line = line.substr(0, end + 1);
  }

  if (line.empty()) {
    result.push_back("");
    return result;
  }

  std::string mnemonic = TranspileExtractMnemonic(line);

  // GFX12 _nc_ (no-carry) VALU variants → remove _nc_
  if (mnemonic.find("_nc_") != std::string::npos && mnemonic[0] == 'v') {
    std::string fixed = mnemonic;
    size_t nc_pos = fixed.find("_nc_");
    fixed.replace(nc_pos, 4, "_");
    if (fixed.find("_e32") == std::string::npos &&
        fixed.find("_e64") == std::string::npos)
      fixed += "_e32";
    line = TranspileReplaceMnemonic(line, mnemonic, fixed);
    mnemonic = fixed;
  }

  // GFX12 s_and_not1/s_or_not1 → s_andn2/s_orn2
  if (mnemonic.find("_not1_") != std::string::npos && mnemonic[0] == 's') {
    std::string fixed = mnemonic;
    size_t not1_pos = fixed.find("_not1_");
    fixed.replace(not1_pos, 6, "n2_");
    line = TranspileReplaceMnemonic(line, mnemonic, fixed);
    mnemonic = fixed;
  }

  // v_bitop2_b32 / v_bitop3_b32 → emulate
  if (mnemonic.find("v_bitop") == 0) {
    std::string ops = line.substr(line.find(mnemonic) + mnemonic.size());
    size_t bitop_pos = ops.find("bitop3:");
    std::string op_part = (bitop_pos != std::string::npos) ? ops.substr(0, bitop_pos) : ops;
    std::vector<std::string> operands;
    std::istringstream oss(op_part);
    std::string tok;
    while (std::getline(oss, tok, ',')) {
      size_t s = tok.find_first_not_of(" \t");
      size_t e = tok.find_last_not_of(" \t");
      if (s != std::string::npos)
        operands.push_back(tok.substr(s, e - s + 1));
    }
    if (operands.size() >= 3)
      result.push_back("v_and_b32_e32 " + operands[0] + ", " + operands[1] + ", " + operands[2]);
    else
      result.push_back("s_nop 0 ; UNSUPPORTED: " + line);
    return result;
  }

  // Wait counter translation
  if (IsWaitInstruction(mnemonic)) {
    result.push_back(TranslateWaitInstruction(line));
    return result;
  }

  // GFX12 scheduling/clause hints → SKIP
  if (mnemonic == "s_wait_alu" || mnemonic == "s_delay_alu" ||
      mnemonic == "s_clause" || mnemonic == "s_set_inst_prefetch_distance")
    return result;

  // s_load_b96 → split into s_load_dwordx2 + s_load_dword
  if (mnemonic == "s_load_b96") {
    std::regex reg_range(R"(s\[(\d+):(\d+)\])");
    std::smatch m;
    std::string ops_part = line.substr(line.find(mnemonic) + mnemonic.size());
    if (std::regex_search(ops_part, m, reg_range)) {
      int lo = std::stoi(m[1]);
      std::string after_reg = ops_part.substr(m.position() + m.length());
      size_t offset_pos = after_reg.rfind("0x");
      if (offset_pos == std::string::npos) offset_pos = after_reg.rfind(' ');
      if (offset_pos != std::string::npos) {
        size_t num_start = after_reg.find_last_of(" \t,", after_reg.size()-1);
        if (num_start == std::string::npos) num_start = 0; else num_start++;
        std::string offset_str = after_reg.substr(num_start);
        int64_t offset_val = 0;
        try { offset_val = std::stoll(offset_str, nullptr, 0); } catch (...) {}
        std::string base_part = after_reg.substr(0, num_start);
        size_t be = base_part.find_last_not_of(" \t,");
        if (be != std::string::npos) base_part = base_part.substr(0, be+1);
        result.push_back("s_load_dwordx2 s[" + std::to_string(lo) + ":" +
                          std::to_string(lo+1) + "]" + base_part +
                          ", 0x" + ([](int64_t v) { std::ostringstream o; o << std::hex << v; return o.str(); })(offset_val));
        result.push_back("s_load_dword s" + std::to_string(lo+2) + base_part +
                          ", 0x" + ([](int64_t v) { std::ostringstream o; o << std::hex << v; return o.str(); })(offset_val + 8));
      }
    }
    if (result.empty()) {
      std::string new_line = line;
      size_t mpos = new_line.find("s_load_b96");
      new_line.replace(mpos, 10, "s_load_dwordx4");
      std::smatch m2;
      if (std::regex_search(new_line, m2, reg_range)) {
        int lo2 = std::stoi(m2[1]);
        std::string wider = "s[" + std::to_string(lo2) + ":" + std::to_string(lo2 + 3) + "]";
        new_line = new_line.substr(0, m2.position()) + wider +
                   new_line.substr(m2.position() + m2.length());
      }
      result.push_back(new_line);
    }
    return result;
  }

  // TTMP/HW_REG fallback handling
  if ((mnemonic == "s_setreg_imm32_b32" || mnemonic == "s_setreg_b32" ||
       mnemonic == "s_getreg_b32") &&
      (line.find("HW_REG_WAVE_MODE") != std::string::npos ||
       line.find("HW_REG_IB_STS2") != std::string::npos))
    return result;
  if (line.find("ttmp6") != std::string::npos || line.find("ttmp7") != std::string::npos ||
      line.find("ttmp9") != std::string::npos)
    return result;

  if (mnemonic == "s_code_end")
    return result;

  if (mnemonic == "s_endpgm") {
    result.push_back(".L_exit:");
    result.push_back("s_endpgm");
    return result;
  }

  // Barrier translation
  if (mnemonic == "s_barrier_signal") {
    result.push_back("s_barrier");
    return result;
  }
  if (mnemonic == "s_barrier_wait") {
    result.push_back("s_nop 0");
    return result;
  }

  // s_add_nc_u64 → emulate with s_add_u32 + s_addc_u32
  if (mnemonic == "s_add_nc_u64" || mnemonic == "s_sub_nc_u64") {
    std::string ops = line.substr(line.find(mnemonic) + mnemonic.size());
    std::vector<std::string> operands;
    std::istringstream oss(ops);
    std::string tok;
    while (std::getline(oss, tok, ',')) {
      size_t s = tok.find_first_not_of(" \t");
      size_t e = tok.find_last_not_of(" \t");
      if (s != std::string::npos) operands.push_back(tok.substr(s, e - s + 1));
    }
    if (operands.size() >= 3) {
      auto parseSpair = [](const std::string& s) -> std::pair<int,int> {
        auto bracket = s.find('[');
        if (bracket == std::string::npos) return {-1,-1};
        int lo=0, hi=0; size_t p = bracket+1;
        while (p<s.size()&&s[p]>='0'&&s[p]<='9') lo=lo*10+(s[p++]-'0');
        if (p<s.size()&&s[p]==':') p++;
        while (p<s.size()&&s[p]>='0'&&s[p]<='9') hi=hi*10+(s[p++]-'0');
        return {lo, hi};
      };
      auto [d0,d1] = parseSpair(operands[0]);
      auto [a0,a1] = parseSpair(operands[1]);
      std::string src = operands[2];
      auto [s0,s1] = parseSpair(src);
      std::string src_lo = (s0>=0) ? "s"+std::to_string(s0) : src;
      std::string src_hi = (s0>=0) ? "s"+std::to_string(s1) : "0";
      bool is_sub = mnemonic.find("sub") != std::string::npos;
      std::string op = is_sub ? "s_sub_u32" : "s_add_u32";
      std::string opc = is_sub ? "s_subb_u32" : "s_addc_u32";
      result.push_back(op + " s" + std::to_string(d0) + ", s" + std::to_string(a0) + ", " + src_lo);
      result.push_back(opc + " s" + std::to_string(d1) + ", s" + std::to_string(a1) + ", " + src_hi);
      return result;
    }
    result.push_back("s_nop 0 ; UNSUPPORTED: " + line);
    return result;
  }

  // v_add_nc_u64 → emulate with v_lshl_add_u64
  if (mnemonic == "v_add_nc_u64" || mnemonic == "v_add_nc_u64_e32" ||
      mnemonic == "v_add_u64" || mnemonic == "v_add_u64_e32") {
    std::string ops = line.substr(line.find(mnemonic) + mnemonic.size());
    struct RegOrImm { char prefix; int lo; int hi; std::string imm; };
    auto parseOperand = [](const std::string& s, size_t& pos) -> RegOrImm {
      while (pos < s.size() && (s[pos]==' '||s[pos]==','||s[pos]=='\t')) ++pos;
      if (pos >= s.size()) return {'?', -1, -1, ""};
      char prefix = s[pos];
      if (prefix == 'v' || prefix == 's') {
        ++pos;
        if (pos < s.size() && s[pos] == '[') {
          ++pos; int lo=0, hi=0;
          while (pos<s.size()&&s[pos]>='0'&&s[pos]<='9') lo=lo*10+(s[pos++]-'0');
          if (pos<s.size()&&s[pos]==':') ++pos;
          while (pos<s.size()&&s[pos]>='0'&&s[pos]<='9') hi=hi*10+(s[pos++]-'0');
          if (pos<s.size()&&s[pos]==']') ++pos;
          return {prefix, lo, hi, ""};
        }
        return {'?', -1, -1, ""};
      }
      size_t st = pos;
      if (s[pos]=='-') ++pos;
      while (pos<s.size()&&(std::isalnum(s[pos])||s[pos]=='x')) ++pos;
      return {'#', 0, 0, s.substr(st, pos-st)};
    };
    auto fmt = [](char p, int n) -> std::string {
      return std::string(1, p) + std::to_string(n);
    };
    (void)fmt;
    size_t pos = 0;
    auto d = parseOperand(ops, pos);
    auto a = parseOperand(ops, pos);
    auto b = parseOperand(ops, pos);
    if (d.lo >= 0 && (a.lo >= 0 || a.prefix == '#') && (b.lo >= 0 || b.prefix == '#')) {
      auto fmtPair = [&](char p, int lo, int hi) -> std::string {
        return std::string(1, p) + "[" + std::to_string(lo) + ":" + std::to_string(hi) + "]";
      };
      std::string src0_pair;
      if (a.prefix == 'v' || a.prefix == 's')
        src0_pair = fmtPair(a.prefix, a.lo, a.hi);
      else {
        result.push_back("v_mov_b32_e32 v252, " + a.imm);
        result.push_back("v_mov_b32_e32 v253, 0");
        src0_pair = "v[252:253]";
      }
      std::string src1;
      if (b.prefix == '#')
        src1 = b.imm;
      else
        src1 = fmtPair(b.prefix, b.lo, b.hi);
      std::string dst_pair = fmtPair(d.prefix, d.lo, d.hi);
      result.push_back("v_lshl_add_u64 " + dst_pair + ", " + src0_pair + ", 0, " + src1);
      return result;
    }
    result.push_back("s_nop 0 ; UNSUPPORTED: " + line);
    return result;
  }

  // v_bitop2_b32 / v_bitop3_b32 (duplicated catch for safety)
  if (mnemonic.find("v_bitop2_b32") == 0 || mnemonic.find("v_bitop3_b32") == 0) {
    std::string ops = line.substr(line.find(mnemonic) + mnemonic.size());
    size_t bitop_pos = ops.find("bitop3:");
    if (bitop_pos != std::string::npos) {
      int truth_table = 0;
      std::string hex_str = ops.substr(bitop_pos + 7);
      try { truth_table = std::stoi(hex_str, nullptr, 0); } catch (...) {}
      std::string op_part = ops.substr(0, bitop_pos);
      std::vector<std::string> operands;
      std::istringstream oss(op_part);
      std::string tok;
      while (std::getline(oss, tok, ',')) {
        size_t s = tok.find_first_not_of(" \t");
        size_t e = tok.find_last_not_of(" \t");
        if (s != std::string::npos) operands.push_back(tok.substr(s, e - s + 1));
      }
      if (operands.size() >= 3) {
        std::string vdst = operands[0], src0 = operands[1], src1 = operands[2];
        if (truth_table == 0xCA)
          result.push_back("v_bfi_b32 " + vdst + ", " + src0 + ", " + src1 + ", " + vdst);
        else
          result.push_back("v_and_b32 " + vdst + ", " + src0 + ", " + src1);
        return result;
      }
    }
    result.push_back("s_nop 0 ; UNSUPPORTED: " + line);
    return result;
  }

  // v_perm_b32 with literal constant → move to s13
  if (mnemonic == "v_perm_b32") {
    std::string ops = line.substr(line.find(mnemonic) + mnemonic.size());
    if (ops.find("0x") != std::string::npos) {
      std::vector<std::string> operands;
      std::istringstream oss(ops);
      std::string tok;
      while (std::getline(oss, tok, ',')) {
        size_t s = tok.find_first_not_of(" \t");
        size_t e = tok.find_last_not_of(" \t");
        if (s != std::string::npos) operands.push_back(tok.substr(s, e - s + 1));
      }
      if (operands.size() >= 4) {
        result.push_back("s_mov_b32 s13, " + operands[3]);
        result.push_back("v_perm_b32 " + operands[0] + ", " + operands[1] +
                         ", " + operands[2] + ", s13");
        return result;
      }
    }
  }

  // v_fmamk_f32 / v_fmaak_f32 → v_mov_b32 + v_fma_f32
  if (mnemonic == "v_fmamk_f32" || mnemonic == "v_fmaak_f32") {
    size_t ops_start = line.find(mnemonic) + mnemonic.size();
    std::string ops = line.substr(ops_start);
    std::vector<std::string> operands;
    std::istringstream oss(ops);
    std::string tok;
    while (std::getline(oss, tok, ',')) {
      size_t s = tok.find_first_not_of(" \t");
      size_t e = tok.find_last_not_of(" \t");
      if (s != std::string::npos) operands.push_back(tok.substr(s, e - s + 1));
    }
    if (operands.size() >= 4) {
      const std::string& literal = (mnemonic == "v_fmamk_f32") ? operands[2] : operands[3];
      result.push_back("v_mov_b32_e32 v6, " + literal);
      if (mnemonic == "v_fmamk_f32")
        result.push_back("v_fma_f32 " + operands[0] + ", " + operands[1] + ", v6, " + operands[3]);
      else
        result.push_back("v_fma_f32 " + operands[0] + ", " + operands[1] + ", " + operands[2] + ", v6");
      return result;
    }
  }

  // VOPD (dual-issue) → two separate instructions
  if (mnemonic.find("v_dual_") == 0) {
    size_t sep = line.find("::");
    if (sep != std::string::npos) {
      std::string first_half = line.substr(0, sep);
      size_t fs = first_half.find_first_not_of(" \t");
      if (fs != std::string::npos) first_half = first_half.substr(fs);
      size_t fe = first_half.find_last_not_of(" \t");
      if (fe != std::string::npos) first_half = first_half.substr(0, fe + 1);
      if (first_half.find("v_dual_") == 0) first_half = "v_" + first_half.substr(7);

      std::string second_half = line.substr(sep + 2);
      size_t ss = second_half.find_first_not_of(" \t");
      if (ss != std::string::npos) second_half = second_half.substr(ss);
      size_t se = second_half.find_last_not_of(" \t");
      if (se != std::string::npos) second_half = second_half.substr(0, se + 1);
      if (second_half.find("v_dual_") == 0) second_half = "v_" + second_half.substr(7);

      auto r1 = TranslateInstruction(first_half, source_cpu, target_cpu, scale_temp_vgpr, cmpx_temp_sgpr, compact_mode);
      auto r2 = TranslateInstruction(second_half, source_cpu, target_cpu, scale_temp_vgpr, cmpx_temp_sgpr, compact_mode);
      result.insert(result.end(), r1.begin(), r1.end());
      result.insert(result.end(), r2.begin(), r2.end());
      return result;
    }
  }

  // s_lshlN_add_u32 → s_lshl_b32 + s_add_u32
  {
    int shift_amt = -1;
    if (mnemonic == "s_lshl1_add_u32") shift_amt = 1;
    else if (mnemonic == "s_lshl2_add_u32") shift_amt = 2;
    else if (mnemonic == "s_lshl3_add_u32") shift_amt = 3;
    else if (mnemonic == "s_lshl4_add_u32") shift_amt = 4;
    if (shift_amt >= 0) {
      std::string ops = line.substr(line.find(mnemonic) + mnemonic.size());
      size_t op_start = ops.find_first_not_of(" \t");
      if (op_start != std::string::npos) ops = ops.substr(op_start);
      std::vector<std::string> operands;
      std::istringstream oss(ops);
      std::string tok;
      while (std::getline(oss, tok, ',')) {
        size_t s = tok.find_first_not_of(" \t");
        size_t e = tok.find_last_not_of(" \t");
        if (s != std::string::npos) operands.push_back(tok.substr(s, e - s + 1));
      }
      if (operands.size() >= 3) {
        result.push_back("s_lshl_b32 " + operands[0] + ", " + operands[1] + ", " + std::to_string(shift_amt));
        if (operands[2] != "0")
          result.push_back("s_add_u32 " + operands[0] + ", " + operands[0] + ", " + operands[2]);
        return result;
      }
    }
  }

  // SALU float → VALU emulation
  {
    static const std::unordered_map<std::string, std::string> kSaluFloatMap = {
        {"s_add_f32", "v_add_f32_e32"}, {"s_sub_f32", "v_sub_f32_e32"},
        {"s_mul_f32", "v_mul_f32_e32"}, {"s_min_f32", "v_min_f32_e32"},
        {"s_max_f32", "v_max_f32_e32"}, {"s_fmac_f32", "v_fmac_f32_e32"},
        {"s_add_f16", "v_add_f16_e32"}, {"s_sub_f16", "v_sub_f16_e32"},
        {"s_mul_f16", "v_mul_f16_e32"}, {"s_min_f16", "v_min_f16_e32"},
        {"s_max_f16", "v_max_f16_e32"},
    };
    auto salu_it = kSaluFloatMap.find(mnemonic);
    if (salu_it != kSaluFloatMap.end()) {
      std::string ops = line.substr(line.find(mnemonic) + mnemonic.size());
      size_t op_start = ops.find_first_not_of(" \t");
      if (op_start != std::string::npos) ops = ops.substr(op_start);
      std::vector<std::string> operands;
      std::istringstream oss(ops);
      std::string tok;
      while (std::getline(oss, tok, ',')) {
        size_t s = tok.find_first_not_of(" \t");
        size_t e = tok.find_last_not_of(" \t");
        if (s != std::string::npos) operands.push_back(tok.substr(s, e - s + 1));
      }
      if (operands.size() >= 3) {
        std::string sdst = operands[0], ssrc0 = operands[1], ssrc1 = operands[2];
        if (mnemonic == "s_fmac_f32") {
          result.push_back("v_mov_b32_e32 v6, " + ssrc0);
          result.push_back("v_mul_f32_e32 v6, " + ssrc1 + ", v6");
          result.push_back("v_add_f32_e32 v6, " + sdst + ", v6");
          result.push_back("v_readfirstlane_b32 " + sdst + ", v6");
        } else {
          result.push_back("v_mov_b32_e32 v6, " + ssrc0);
          result.push_back(salu_it->second + " v6, " + ssrc1 + ", v6");
          result.push_back("v_readfirstlane_b32 " + sdst + ", v6");
        }
        return result;
      }
    }
    // s_cvt_* → VALU conversion + readfirstlane
    if (mnemonic == "s_cvt_f32_f16" || mnemonic == "s_cvt_f16_f32" ||
        mnemonic == "s_cvt_pk_rtz_f16_f32" ||
        mnemonic == "s_cvt_f32_u32" || mnemonic == "s_cvt_f32_i32" ||
        mnemonic == "s_cvt_u32_f32" || mnemonic == "s_cvt_i32_f32") {
      std::string ops = line.substr(line.find(mnemonic) + mnemonic.size());
      size_t op_start = ops.find_first_not_of(" \t");
      if (op_start != std::string::npos) ops = ops.substr(op_start);
      std::vector<std::string> operands;
      std::istringstream oss(ops);
      std::string tok;
      while (std::getline(oss, tok, ',')) {
        size_t s = tok.find_first_not_of(" \t");
        size_t e = tok.find_last_not_of(" \t");
        if (s != std::string::npos) operands.push_back(tok.substr(s, e - s + 1));
      }
      if (operands.size() >= 2) {
        std::string valu_mnem;
        if (mnemonic == "s_cvt_f32_f16") valu_mnem = "v_cvt_f32_f16_e32";
        else if (mnemonic == "s_cvt_f16_f32") valu_mnem = "v_cvt_f16_f32_e32";
        else if (mnemonic == "s_cvt_f32_u32") valu_mnem = "v_cvt_f32_u32_e32";
        else if (mnemonic == "s_cvt_f32_i32") valu_mnem = "v_cvt_f32_i32_e32";
        else if (mnemonic == "s_cvt_u32_f32") valu_mnem = "v_cvt_u32_f32_e32";
        else if (mnemonic == "s_cvt_i32_f32") valu_mnem = "v_cvt_i32_f32_e32";
        else valu_mnem = "v_cvt_pkrtz_f16_f32";
        result.push_back(valu_mnem + " v6, " + operands[1]);
        result.push_back("v_readfirstlane_b32 " + operands[0] + ", v6");
        return result;
      }
    }
    if (mnemonic == "v_s_sqrt_f32") {
      std::string ops = line.substr(line.find(mnemonic) + mnemonic.size());
      size_t op_start = ops.find_first_not_of(" \t");
      if (op_start != std::string::npos) ops = ops.substr(op_start);
      std::vector<std::string> operands;
      std::istringstream oss(ops);
      std::string tok;
      while (std::getline(oss, tok, ',')) {
        size_t s = tok.find_first_not_of(" \t");
        size_t e = tok.find_last_not_of(" \t");
        if (s != std::string::npos) operands.push_back(tok.substr(s, e - s + 1));
      }
      if (operands.size() >= 2) {
        result.push_back("v_sqrt_f32_e32 v6, " + operands[1]);
        result.push_back("v_readfirstlane_b32 " + operands[0] + ", v6");
        return result;
      }
    }
    // s_cmp_*_f32 → VALU float compare
    {
      static const std::unordered_map<std::string, std::string> kScmpFloatMap = {
        {"s_cmp_gt_f32", "v_cmp_gt_f32_e32"}, {"s_cmp_ge_f32", "v_cmp_ge_f32_e32"},
        {"s_cmp_lt_f32", "v_cmp_lt_f32_e32"}, {"s_cmp_le_f32", "v_cmp_le_f32_e32"},
        {"s_cmp_eq_f32", "v_cmp_eq_f32_e32"}, {"s_cmp_lg_f32", "v_cmp_lg_f32_e32"},
      };
      auto cmp_it = kScmpFloatMap.find(mnemonic);
      if (cmp_it != kScmpFloatMap.end()) {
        std::string ops = line.substr(line.find(mnemonic) + mnemonic.size());
        size_t op_start = ops.find_first_not_of(" \t");
        if (op_start != std::string::npos) ops = ops.substr(op_start);
        std::vector<std::string> operands;
        std::istringstream oss(ops);
        std::string tok;
        while (std::getline(oss, tok, ',')) {
          size_t s = tok.find_first_not_of(" \t");
          size_t e = tok.find_last_not_of(" \t");
          if (s != std::string::npos) operands.push_back(tok.substr(s, e - s + 1));
        }
        if (operands.size() >= 2) {
          result.push_back("v_mov_b32_e32 v6, " + operands[1]);
          result.push_back(cmp_it->second + " " + operands[0] + ", v6");
          result.push_back("s_cmp_lg_u32 vcc_lo, 0");
          return result;
        }
      }
    }
  }

  // v_mad_u32 → emulate with v_mul_lo_u32 + v_add_u32
  if (mnemonic == "v_mad_u32") {
    std::string ops = line.substr(line.find(mnemonic) + mnemonic.size());
    size_t op_start = ops.find_first_not_of(" \t");
    if (op_start != std::string::npos) ops = ops.substr(op_start);
    std::vector<std::string> operands;
    std::istringstream oss(ops);
    std::string tok;
    while (std::getline(oss, tok, ',')) {
      size_t s = tok.find_first_not_of(" \t");
      size_t e = tok.find_last_not_of(" \t");
      if (s != std::string::npos) operands.push_back(tok.substr(s, e - s + 1));
    }
    if (operands.size() >= 4) {
      std::string vdst = operands[0], src0 = operands[1], src1 = operands[2], src2 = operands[3];
      bool src0_sgpr = !src0.empty() && src0[0] == 's';
      bool src1_sgpr = !src1.empty() && src1[0] == 's';
      if (vdst != src2) {
        if (src0_sgpr && src1_sgpr) {
          result.push_back("v_mov_b32_e32 v6, " + src0);
          result.push_back("v_mul_lo_u32 " + vdst + ", v6, " + src1);
        } else {
          result.push_back("v_mul_lo_u32 " + vdst + ", " + src0 + ", " + src1);
        }
        result.push_back("v_add_u32_e32 " + vdst + ", " + vdst + ", " + src2);
      } else {
        result.push_back("v_mov_b32_e32 v6, " + src0);
        result.push_back("v_mul_lo_u32 v6, v6, " + src1);
        result.push_back("v_add_u32_e32 " + vdst + ", v6, " + src2);
      }
      return result;
    }
  }

  // v_cndmask_b32_e64: bare SGPR mask widening + constant bus fix
  if (mnemonic == "v_cndmask_b32_e64") {
    size_t ops_start = line.find(mnemonic) + mnemonic.size();
    std::string ops = line.substr(ops_start);
    std::vector<std::string> operands;
    std::istringstream oss(ops);
    std::string tok;
    while (std::getline(oss, tok, ',')) {
      size_t s = tok.find_first_not_of(" \t");
      size_t e = tok.find_last_not_of(" \t");
      if (s != std::string::npos) operands.push_back(tok.substr(s, e - s + 1));
    }
    if (operands.size() >= 4) {
      for (auto& op : operands) { if (op == "vcc_lo") op = "vcc"; }
      auto& mask = operands[3];
      if (!mask.empty() && mask[0] == 's' && mask.find('[') == std::string::npos &&
          mask.find("vcc") == std::string::npos && mask.find("exec") == std::string::npos &&
          mask.size() > 1 && std::isdigit((unsigned char)mask[1])) {
        int n = std::stoi(mask.substr(1));
        int even = n & ~1;
        mask = "s[" + std::to_string(even) + ":" + std::to_string(even + 1) + "]";
      }
      std::string& src1 = operands[2];
      bool src1_sgpr = !src1.empty() && src1[0] == 's';
      bool mask_sgpr = !mask.empty() && (mask[0] == 's' || mask.substr(0, 3) == "vcc" || mask.substr(0, 4) == "exec");
      if (src1_sgpr && mask_sgpr) {
        result.push_back("v_mov_b32_e32 v6, " + src1);
        src1 = "v6";
      }
      std::string fixed = mnemonic + " " + operands[0];
      for (size_t i = 1; i < operands.size(); ++i) fixed += ", " + operands[i];
      result.push_back(fixed);
      return result;
    }
  }

  // v_cndmask_b32_e32 SGPR src0 → move to v6 (constant bus fix)
  if (mnemonic == "v_cndmask_b32_e32") {
    size_t ops_start = line.find(mnemonic) + mnemonic.size();
    std::string ops = line.substr(ops_start);
    std::vector<std::string> operands;
    std::istringstream oss2(ops);
    std::string tok2;
    while (std::getline(oss2, tok2, ',')) {
      size_t s = tok2.find_first_not_of(" \t");
      size_t e = tok2.find_last_not_of(" \t");
      if (s != std::string::npos) operands.push_back(tok2.substr(s, e - s + 1));
    }
    if (operands.size() >= 3) {
      std::string& src0 = operands[1];
      bool src0_sgpr = !src0.empty() && src0[0] == 's';
      bool src0_lit = src0.size() > 2 && src0[0] == '0' && (src0[1] == 'x' || src0[1] == 'X');
      if (src0_sgpr || src0_lit) {
        result.push_back("v_mov_b32_e32 v6, " + src0);
        src0 = "v6";
        std::string fixed = mnemonic + " " + operands[0];
        for (size_t i = 1; i < operands.size(); ++i) fixed += ", " + operands[i];
        result.push_back(fixed);
        return result;
      }
    }
  }

  // v_cndmask_b32 (no suffix) with explicit mask → rename to _e64
  if (mnemonic == "v_cndmask_b32") {
    size_t ops_start = line.find(mnemonic) + mnemonic.size();
    std::string ops = line.substr(ops_start);
    std::vector<std::string> operands;
    std::istringstream oss(ops);
    std::string tok;
    while (std::getline(oss, tok, ',')) {
      size_t s = tok.find_first_not_of(" \t");
      size_t e = tok.find_last_not_of(" \t");
      if (s != std::string::npos) operands.push_back(tok.substr(s, e - s + 1));
    }
    if (operands.size() >= 4) {
      for (auto& op : operands) { if (op == "vcc_lo") op = "vcc"; }
      auto& mask = operands[3];
      if (!mask.empty() && mask[0] == 's' && mask.find('[') == std::string::npos &&
          mask.find("vcc") == std::string::npos && mask.find("exec") == std::string::npos &&
          mask.size() > 1 && std::isdigit((unsigned char)mask[1])) {
        int n = std::stoi(mask.substr(1));
        int even = n & ~1;
        mask = "s[" + std::to_string(even) + ":" + std::to_string(even + 1) + "]";
      }
      std::string fixed = "v_cndmask_b32_e64 " + operands[0];
      for (size_t i = 1; i < operands.size(); ++i) fixed += ", " + operands[i];
      result.push_back(fixed);
      return result;
    }
  }

  // DPP8 → DPP16 or identity conversion
  if (line.find("dpp8:") != std::string::npos) {
    size_t dpp8_pos = line.find("dpp8:[");
    if (dpp8_pos != std::string::npos) {
      std::string base_part = line.substr(0, dpp8_pos);
      size_t be = base_part.find_last_not_of(" \t");
      if (be != std::string::npos) base_part = base_part.substr(0, be + 1);
      if (line.find("dpp8:[0,1,2,3,4,5,6,7]") != std::string::npos) {
        std::string no_dpp = base_part;
        size_t dpp_suffix = no_dpp.find("_dpp");
        if (dpp_suffix != std::string::npos) no_dpp.replace(dpp_suffix, 4, "_e32");
        result.push_back(no_dpp);
        return result;
      }
      result.push_back(base_part + " row_shr:0 row_mask:0xf bank_mask:0xf");
      return result;
    }
  }

  // VOP3 with 32-bit literal → v_mov_b32 + replace with v6
  {
    bool is_vop3_candidate =
        mnemonic == "v_fma_f32" || mnemonic == "v_fma_f16" ||
        mnemonic == "v_fma_f64" || mnemonic == "v_ldexp_f32" ||
        mnemonic == "v_div_fmas_f32" || mnemonic == "v_div_fixup_f32" ||
        mnemonic == "v_med3_f32" || mnemonic == "v_med3_i32" ||
        mnemonic == "v_bfi_b32" || mnemonic == "v_alignbit_b32";
    if (is_vop3_candidate) {
      size_t ops_start = line.find(mnemonic) + mnemonic.size();
      std::string ops = line.substr(ops_start);
      std::vector<std::string> operands;
      std::istringstream oss(ops);
      std::string tok;
      while (std::getline(oss, tok, ',')) {
        size_t s = tok.find_first_not_of(" \t");
        size_t e = tok.find_last_not_of(" \t");
        if (s != std::string::npos) operands.push_back(tok.substr(s, e - s + 1));
      }
      if (operands.size() >= 3) {
        for (size_t i = 1; i < operands.size(); ++i) {
          std::string op = operands[i];
          bool neg = !op.empty() && op[0] == '-';
          if (neg) op = op.substr(1);
          if (op.size() > 2 && op[0] == '0' && (op[1] == 'x' || op[1] == 'X')) {
            result.push_back("v_mov_b32_e32 v6, " + op);
            operands[i] = (neg ? "-" : "") + std::string("v6");
            std::string fixed = mnemonic + " " + operands[0];
            for (size_t j = 1; j < operands.size(); ++j) fixed += ", " + operands[j];
            result.push_back(fixed);
            return result;
          }
        }
      }
    }
  }

  // v_div_scale_f32 null sdst → vcc
  if (mnemonic == "v_div_scale_f32") {
    size_t null_pos = line.find(", null,");
    if (null_pos != std::string::npos)
      line.replace(null_pos, 7, ", vcc,");
  }

  // WMMA → MFMA translation
  {
    struct WmmaMfmaMapping { const char* wmma_mnem; const char* mfma_mnem; int dst_vgprs_w32; int src_vgprs_w32; int dst_vgprs_w64; int src_vgprs_w64; };
    static const WmmaMfmaMapping kWmmaMap[] = {
      {"v_wmma_f32_16x16x32_f16", "v_mfma_f32_16x16x32_f16", 8, 8, 4, 4},
      {"v_wmma_f32_16x16x32_bf16", "v_mfma_f32_16x16x32bf16", 8, 8, 4, 4},
      {"v_wmma_f32_16x16x16_f16", "v_mfma_f32_16x16x16_f16", 8, 4, 4, 2},
      {"v_wmma_f32_16x16x4_f32", "v_mfma_f32_16x16x4_f32", 8, 2, 4, 1},
      {"v_wmma_i32_16x16x64_iu8", "v_mfma_i32_16x16x64_i8", 8, 8, 4, 4},
    };
    const WmmaMfmaMapping* mapping = nullptr;
    for (const auto& m : kWmmaMap)
      if (mnemonic == m.wmma_mnem) { mapping = &m; break; }
    if (mapping) {
      std::string ops = line.substr(line.find(mnemonic) + mnemonic.size());
      size_t op_start = ops.find_first_not_of(" \t");
      if (op_start != std::string::npos) ops = ops.substr(op_start);
      auto parseVRegRange = [](const std::string& s, size_t& pos) -> std::pair<int, int> {
        while (pos < s.size() && (s[pos] == ' ' || s[pos] == ',' || s[pos] == '\t')) ++pos;
        if (pos >= s.size() || s[pos] != 'v') return {-1, -1};
        ++pos;
        if (pos < s.size() && s[pos] == '[') {
          ++pos; int lo = 0, hi = 0;
          while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') lo = lo * 10 + (s[pos++] - '0');
          if (pos < s.size() && s[pos] == ':') ++pos;
          while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') hi = hi * 10 + (s[pos++] - '0');
          if (pos < s.size() && s[pos] == ']') ++pos;
          return {lo, hi};
        }
        return {-1, -1};
      };
      size_t pos = 0;
      auto dst = parseVRegRange(ops, pos);
      auto srcA = parseVRegRange(ops, pos);
      auto srcB = parseVRegRange(ops, pos);
      auto acc = parseVRegRange(ops, pos);
      if (dst.first < 0 || srcA.first < 0 || srcB.first < 0 || acc.first < 0) {
        result.push_back("s_nop 0 ; WMMA parse failed: " + line);
        return result;
      }
      int src_w64 = mapping->src_vgprs_w64;
      int dst_w64 = mapping->dst_vgprs_w64;
      std::string mfma_mnem = mapping->mfma_mnem;
      int t_lane = 248, t_src = 249, t_addr = 250, t_upper = 251, t0 = 252, t1 = 253;
      int mfma_srcA = srcA.first, mfma_srcB = srcB.first, mfma_acc = acc.first, mfma_dst = dst.first;
      auto regRange = [](int base, int count) -> std::string {
        if (count == 1) return "v" + std::to_string(base);
        return "v[" + std::to_string(base) + ":" + std::to_string(base + count - 1) + "]";
      };
      auto emitRedistribute = [&](int base_w32, int out_base, int n_w64) {
        for (int i = 0; i < n_w64; ++i) {
          int lo_reg = base_w32 + i, hi_reg = base_w32 + n_w64 + i;
          result.push_back("ds_bpermute_b32 v" + std::to_string(t0) + ", v" + std::to_string(t_addr) + ", v" + std::to_string(lo_reg));
          result.push_back("ds_bpermute_b32 v" + std::to_string(t1) + ", v" + std::to_string(t_addr) + ", v" + std::to_string(hi_reg));
          result.push_back("s_waitcnt lgkmcnt(0)");
          result.push_back("v_cndmask_b32_e32 v" + std::to_string(out_base + i) + ", v" + std::to_string(t0) + ", v" + std::to_string(t1) + ", vcc");
        }
      };
      result.push_back("; BEGIN WMMA→MFMA: " + mnemonic + " → " + mfma_mnem);
      result.push_back("s_mov_b32 s12, exec_lo");
      result.push_back("s_mov_b32 s13, exec_hi");
      result.push_back("s_mov_b64 exec, -1");
      result.push_back("v_mbcnt_lo_u32_b32 v" + std::to_string(t_lane) + ", -1, 0");
      result.push_back("v_mbcnt_hi_u32_b32 v" + std::to_string(t_lane) + ", -1, v" + std::to_string(t_lane));
      result.push_back("v_lshrrev_b32_e32 v" + std::to_string(t_src) + ", 1, v" + std::to_string(t_lane));
      result.push_back("v_and_b32_e32 v" + std::to_string(t_upper) + ", 1, v" + std::to_string(t_lane));
      result.push_back("v_lshlrev_b32_e32 v" + std::to_string(t_addr) + ", 2, v" + std::to_string(t_src));
      result.push_back("v_cmp_ne_u32 vcc, 0, v" + std::to_string(t_upper));
      emitRedistribute(srcA.first, mfma_srcA, src_w64);
      emitRedistribute(srcB.first, mfma_srcB, src_w64);
      emitRedistribute(acc.first, mfma_acc, dst_w64);
      bool need_decompose = false;
      std::string actual_mfma = mfma_mnem;
      if (target_cpu.find("gfx942") != std::string::npos || target_cpu.find("gfx940") != std::string::npos || target_cpu.find("gfx941") != std::string::npos) {
        if (mfma_mnem == "v_mfma_f32_16x16x32_f16" || mfma_mnem == "v_mfma_f32_16x16x32bf16") {
          need_decompose = true;
          actual_mfma = (mfma_mnem.find("bf16") != std::string::npos) ? "v_mfma_f32_16x16x16bf16_1k" : "v_mfma_f32_16x16x16_f16";
        }
      }
      if (need_decompose) {
        int half_src = src_w64 / 2;
        result.push_back(actual_mfma + " " + regRange(mfma_dst, dst_w64) + ", " + regRange(mfma_srcA, half_src) + ", " + regRange(mfma_srcB, half_src) + ", " + regRange(mfma_acc, dst_w64));
        result.push_back(actual_mfma + " " + regRange(mfma_dst, dst_w64) + ", " + regRange(mfma_srcA + half_src, half_src) + ", " + regRange(mfma_srcB + half_src, half_src) + ", " + regRange(mfma_dst, dst_w64));
      } else {
        result.push_back(mfma_mnem + " " + regRange(mfma_dst, dst_w64) + ", " + regRange(mfma_srcA, src_w64) + ", " + regRange(mfma_srcB, src_w64) + ", " + regRange(mfma_acc, dst_w64));
      }
      result.push_back("s_mov_b32 exec_lo, s12");
      result.push_back("s_mov_b32 exec_hi, 0");
      result.push_back("v_mbcnt_lo_u32_b32 v" + std::to_string(t_lane) + ", -1, 0");
      result.push_back("v_lshlrev_b32_e32 v" + std::to_string(t_addr) + ", 3, v" + std::to_string(t_lane));
      result.push_back("v_add_u32_e32 v" + std::to_string(t0) + ", 4, v" + std::to_string(t_addr));
      result.push_back("s_mov_b64 exec, -1");
      for (int i = 0; i < dst_w64; ++i) {
        result.push_back("ds_bpermute_b32 v" + std::to_string(dst.first + i) + ", v" + std::to_string(t_addr) + ", v" + std::to_string(mfma_dst + i));
        result.push_back("ds_bpermute_b32 v" + std::to_string(dst.first + dst_w64 + i) + ", v" + std::to_string(t0) + ", v" + std::to_string(mfma_dst + i));
      }
      result.push_back("s_waitcnt lgkmcnt(0)");
      result.push_back("s_mov_b32 exec_lo, s12");
      result.push_back("s_mov_b32 exec_hi, s13");
      result.push_back("; END WMMA→MFMA: " + mfma_mnem);
      return result;
    }
  }

  // Other WMMA/SWMMAC → NOP
  if (mnemonic.find("v_wmma_") == 0 || mnemonic.find("v_swmmac_") == 0) {
    result.push_back("s_nop 0 ; UNSUPPORTED WMMA: " + mnemonic);
    return result;
  }

  // scale_offset emulation
  if (line.find("scale_offset") != std::string::npos) {
    int shift = 0;
    if (mnemonic.find("_b32") != std::string::npos || mnemonic.find("_dword") != std::string::npos) shift = 2;
    else if (mnemonic.find("_b64") != std::string::npos || mnemonic.find("_dwordx2") != std::string::npos) shift = 3;
    else if (mnemonic.find("_b128") != std::string::npos || mnemonic.find("_dwordx4") != std::string::npos) shift = 4;
    else if (mnemonic.find("_b16") != std::string::npos) shift = 1;
    if (shift > 0) {
      size_t mnem_end = line.find(mnemonic) + mnemonic.size();
      std::string ops = line.substr(mnem_end);
      bool is_store = mnemonic.find("store") != std::string::npos;
      std::string vaddr;
      if (is_store) {
        size_t s = ops.find_first_not_of(" \t");
        size_t e = ops.find_first_of(" \t,", s);
        if (s != std::string::npos) vaddr = ops.substr(s, e != std::string::npos ? e - s : std::string::npos);
      } else {
        size_t comma1 = ops.find(',');
        if (comma1 != std::string::npos) {
          size_t s = ops.find_first_not_of(" \t,", comma1 + 1);
          size_t e = ops.find_first_of(" \t,", s);
          if (s != std::string::npos) vaddr = ops.substr(s, e != std::string::npos ? e - s : std::string::npos);
        }
      }
      if (!vaddr.empty()) {
        const std::string kScaleTemp = "v" + std::to_string(scale_temp_vgpr);
        result.push_back("v_lshlrev_b32_e32 " + kScaleTemp + ", " + std::to_string(shift) + ", " + vaddr);
        std::string modified = line;
        size_t vaddr_pos = modified.find(vaddr, mnem_end);
        if (vaddr_pos != std::string::npos) modified.replace(vaddr_pos, vaddr.size(), kScaleTemp);
        size_t so_pos = modified.find("scale_offset");
        if (so_pos != std::string::npos) {
          if (so_pos > 0 && modified[so_pos-1] == ' ') --so_pos;
          modified.erase(so_pos);
        }
        line = modified;
        mnemonic = TranspileExtractMnemonic(line);
      }
    }
  }

  // Remaining _nc_ cleanup
  if (mnemonic.find("_nc_") != std::string::npos && mnemonic[0] == 'v') {
    std::string fixed_mnem = mnemonic;
    size_t nc_pos = fixed_mnem.find("_nc_");
    fixed_mnem.replace(nc_pos, 4, "_");
    line = TranspileReplaceMnemonic(line, mnemonic, fixed_mnem);
    mnemonic = fixed_mnem;
  }

  // Unsupported instructions → NOP
  if (IsUnsupportedOnGFX9(mnemonic)) {
    result.push_back("s_nop 0 ; UNSUPPORTED: " + mnemonic);
    return result;
  }

  // Flat+saddr → Global conversion
  {
    bool is_flat_with_saddr = false;
    if (mnemonic.find("flat_load_") == 0 || mnemonic.find("flat_store_") == 0) {
      size_t s_bracket = line.find("s[", line.find(mnemonic) + mnemonic.size());
      if (s_bracket != std::string::npos) is_flat_with_saddr = true;
    }
    if (is_flat_with_saddr) {
      size_t flat_pos = mnemonic.find("flat_");
      if (flat_pos != std::string::npos) {
        std::string global_mnemonic = "global_" + mnemonic.substr(flat_pos + 5);
        line = TranspileReplaceMnemonic(line, mnemonic, global_mnemonic);
        mnemonic = global_mnemonic;
      }
    }
  }

  // Mnemonic renaming
  const auto& mnem_map = GetMnemonicMap();
  auto it = mnem_map.find(mnemonic);
  if (it != mnem_map.end()) {
    line = TranspileReplaceMnemonic(line, mnemonic, it->second);
    mnemonic = it->second;
  } else {
    std::string base = mnemonic;
    std::string suffix;
    if (base.size() > 4 && base.substr(base.size() - 4) == "_e32") { suffix = "_e32"; base = base.substr(0, base.size() - 4); }
    else if (base.size() > 4 && base.substr(base.size() - 4) == "_e64") { suffix = "_e64"; base = base.substr(0, base.size() - 4); }
    if (!suffix.empty()) {
      it = mnem_map.find(base);
      if (it != mnem_map.end()) {
        std::string new_mnem = it->second + suffix;
        line = TranspileReplaceMnemonic(line, mnemonic, new_mnem);
        mnemonic = new_mnem;
      }
    }
  }

  // Operand syntax translation
  line = TranslateOperandSyntax(line, mnemonic);

  // v_cmp_*_e64 with vcc dest + large literal → VOPC _e32
  if (mnemonic.find("v_cmp_") == 0 && mnemonic.find("_e64") != std::string::npos) {
    size_t op_start = line.find(mnemonic) + mnemonic.size();
    std::string ops_str = line.substr(op_start);
    size_t sp = ops_str.find_first_not_of(" \t");
    if (sp != std::string::npos) ops_str = ops_str.substr(sp);
    size_t first_comma = ops_str.find(',');
    if (first_comma != std::string::npos) {
      std::string first_op = ops_str.substr(0, first_comma);
      size_t fe = first_op.find_last_not_of(" \t");
      if (fe != std::string::npos) first_op = first_op.substr(0, fe + 1);
      if (first_op == "vcc" || first_op == "vcc_lo") {
        std::string src_str = ops_str.substr(first_comma + 1);
        std::vector<std::string> srcs;
        std::istringstream oss(src_str);
        std::string tok;
        while (std::getline(oss, tok, ',')) {
          size_t ts = tok.find_first_not_of(" \t");
          size_t te = tok.find_last_not_of(" \t");
          if (ts != std::string::npos) srcs.push_back(tok.substr(ts, te - ts + 1));
        }
        if (srcs.size() >= 2) {
          auto is_large_literal = [](const std::string& s) -> bool {
            if (s.size() > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) return true;
            if (!s.empty() && std::isdigit((unsigned char)s[0]) && std::stol(s) > 64) return true;
            if (s.size() > 1 && s[0] == '-' && std::isdigit((unsigned char)s[1]) && std::stol(s) < -16) return true;
            return false;
          };
          std::string base_mnem = mnemonic.substr(0, mnemonic.size() - 4) + "_e32";
          bool src0_lit = is_large_literal(srcs[0]);
          bool src1_lit = is_large_literal(srcs[1]);
          if (src1_lit && !src0_lit) {
            result.push_back("v_mov_b32_e32 v6, " + srcs[1]);
            result.push_back(base_mnem + " " + srcs[0] + ", v6");
            return result;
          } else if (src0_lit) {
            result.push_back(base_mnem + " " + srcs[0] + ", " + srcs[1]);
            return result;
          }
        }
      }
    }
  }

  // v_cmpx _e64 → v_cmp_e64 + manual exec AND
  if (mnemonic.find("v_cmpx_") == 0 && mnemonic.find("_e64") != std::string::npos) {
    std::string base_mnem = mnemonic.substr(0, mnemonic.find("_e64"));
    size_t op_start = line.find(mnemonic) + mnemonic.size();
    std::string ops = line.substr(op_start);
    size_t s = ops.find_first_not_of(" \t");
    if (s != std::string::npos) ops = ops.substr(s);
    std::string stemp = "s" + std::to_string(cmpx_temp_sgpr);
    result.push_back("s_mov_b32 " + stemp + ", vcc_lo");
    result.push_back(base_mnem + " " + ops);
    result.push_back("s_mov_b32 vcc_lo, " + stemp);
    return result;
  }

  // v_cmpx _e32 (VOPC) → save/restore vcc_lo
  if (mnemonic.find("v_cmpx_") == 0 && mnemonic.find("_e64") == std::string::npos) {
    std::string stemp = "s" + std::to_string(cmpx_temp_sgpr);
    result.push_back("s_mov_b32 " + stemp + ", vcc_lo");
    result.push_back(line);
    result.push_back("s_mov_b32 exec_hi, 0");
    result.push_back("s_mov_b32 vcc_lo, " + stemp);
    return result;
  }

  // v_cmp _e64 with SGPR dest → expand to SGPR pair for wave64
  if (mnemonic.find("v_cmp_") == 0 && mnemonic.find("_e64") != std::string::npos) {
    size_t op_start = line.find(mnemonic) + mnemonic.size();
    std::string ops = line.substr(op_start);
    size_t s_pos = ops.find_first_not_of(" \t");
    if (s_pos != std::string::npos) {
      std::string trimmed = ops.substr(s_pos);
      if (trimmed[0] == 's' && std::isdigit(trimmed[1])) {
        size_t comma = trimmed.find(',');
        if (comma != std::string::npos) {
          std::string dst_str = trimmed.substr(0, comma);
          size_t de = dst_str.find_last_not_of(" \t");
          dst_str = dst_str.substr(0, de + 1);
          int reg_num = std::stoi(dst_str.substr(1));
          int even = reg_num & ~1;
          int odd = even + 1;
          std::string pair = "s[" + std::to_string(even) + ":" + std::to_string(odd) + "]";
          std::string rest = trimmed.substr(comma);
          std::string save_reg = "v" + std::to_string(scale_temp_vgpr);
          line = mnemonic + " " + pair + rest;
          if (reg_num == even) {
            if (compact_mode) {
              result.push_back(line);
            } else {
              std::string s_odd = "s" + std::to_string(odd);
              result.push_back("v_mov_b32_e32 " + save_reg + ", " + s_odd);
              result.push_back(line);
              result.push_back("v_readfirstlane_b32 " + s_odd + ", " + save_reg);
            }
          } else {
            std::string s_even = "s" + std::to_string(even);
            std::string s_orig = "s" + std::to_string(reg_num);
            result.push_back("v_mov_b32_e32 " + save_reg + ", " + s_even);
            result.push_back(line);
            result.push_back("s_mov_b32 " + s_orig + ", " + s_even);
            result.push_back("v_readfirstlane_b32 " + s_even + ", " + save_reg);
          }
          {
            auto is_ll = [](const std::string& s) -> bool {
              if (s.empty()) return false;
              if (s.size() > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) return true;
              try {
                if (std::isdigit((unsigned char)s[0]) && std::stol(s) > 64) return true;
                if (s[0] == '-' && s.size() > 1 && std::stol(s) < -16) return true;
              } catch (...) {}
              return false;
            };
            for (size_t ri = 0; ri < result.size(); ri++) {
              if (result[ri].find("v_cmp_") != std::string::npos) {
                size_t lc = result[ri].rfind(',');
                if (lc != std::string::npos) {
                  std::string lo = result[ri].substr(lc + 1);
                  size_t ls = lo.find_first_not_of(" \t");
                  size_t le = lo.find_last_not_of(" \t");
                  if (ls != std::string::npos) {
                    std::string lt = lo.substr(ls, le - ls + 1);
                    if (is_ll(lt)) {
                      result.insert(result.begin() + ri, "v_mov_b32_e32 v6, " + lt);
                      result[ri + 1] = result[ri + 1].substr(0, lc + 1) + " v6";
                    }
                  }
                }
                break;
              }
            }
          }
          return result;
        }
      }
    }
    auto is_large_literal = [](const std::string& s) -> bool {
      if (s.empty()) return false;
      if (s.size() > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) return true;
      if (std::isdigit((unsigned char)s[0]) && std::stol(s) > 64) return true;
      if (s[0] == '-' && s.size() > 1 && std::stol(s) < -16) return true;
      return false;
    };
    size_t last_comma = line.rfind(',');
    if (last_comma != std::string::npos) {
      std::string last_op = line.substr(last_comma + 1);
      size_t ls = last_op.find_first_not_of(" \t");
      size_t le = last_op.find_last_not_of(" \t");
      if (ls != std::string::npos) {
        std::string last_trimmed = last_op.substr(ls, le - ls + 1);
        if (is_large_literal(last_trimmed)) {
          result.push_back("v_mov_b32_e32 v6, " + last_trimmed);
          line = line.substr(0, last_comma + 1) + " v6";
        }
      }
    }
  }

  // VCC width translation
  bool is_b32_scalar = (mnemonic.find("_b32") != std::string::npos && mnemonic[0] == 's');
  if (!is_b32_scalar)
    line = WidenVccReferences(line);
  if (is_b32_scalar) {
    size_t mnem_end = line.find_first_of(" \t");
    if (mnem_end != std::string::npos) {
      std::string ops_part = line.substr(mnem_end);
      size_t pos = 0;
      while ((pos = ops_part.find("vcc", pos)) != std::string::npos) {
        size_t end = pos + 3;
        if (end < ops_part.size() && ops_part[end] == '_') { pos = end; continue; }
        ops_part.replace(pos, 3, "vcc_lo");
        pos += 6;
      }
      line = line.substr(0, mnem_end) + ops_part;
    }
  }

  // EXEC width adaptation (wave32 → wave64)
  auto exec_result = WidenExecOperation(line, compact_mode);
  for (auto& l : exec_result)
    result.push_back(std::move(l));
  return result;
}

// ── TranspileStats ───────────────────────────────────────────────────────────

struct TranspileStats {
  uint32_t total_instructions = 0;
  uint32_t translated_passthrough = 0;
  uint32_t translated_renamed = 0;
  uint32_t translated_waitcnt = 0;
  uint32_t translated_exec = 0;
  uint32_t unsupported_skipped = 0;
};

// ── TranspileCodeObject ──────────────────────────────────────────────────────

static amd_comgr_status_t
TranspileCodeObject(const void *elf_data, size_t elf_size,
                    const std::string &source_isa,
                    const std::string &target_isa,
                    void **out_data, size_t *out_size,
                    amd_comgr_hotswap_result_t *result) {
  TranspileStats stats;
  std::string src_cpu = ExtractCPU(source_isa);
  std::string tgt_cpu = ExtractCPU(target_isa);

  std::cerr << "hotswap: transpile: " << src_cpu << " → " << tgt_cpu << "\n";

  const uint8_t* elf = static_cast<const uint8_t*>(elf_data);
  size_t size = elf_size;

  ElfInfo elf_info;
  if (!ParseElfInfo(elf, size, elf_info)) {
    std::cerr << "hotswap: transpile: failed to parse ELF\n";
    return AMD_COMGR_STATUS_ERROR;
  }
  if (elf_info.text_size == 0) {
    std::cerr << "hotswap: transpile: empty .text section\n";
    uint8_t *buf = static_cast<uint8_t *>(std::malloc(elf_size));
    if (!buf) return AMD_COMGR_STATUS_ERROR;
    std::memcpy(buf, elf_data, elf_size);
    *out_data = buf;
    *out_size = elf_size;
    return AMD_COMGR_STATUS_SUCCESS;
  }

  LLVMState &src_state = InitLLVMCached(source_isa);
  if (!src_state.valid) {
    std::cerr << "hotswap: transpile: failed to init source ISA '" << source_isa << "'\n";
    return AMD_COMGR_STATUS_ERROR;
  }

  const uint8_t* text = elf + elf_info.text_offset;

  struct KernelInfo {
    uint64_t desc_offset;
    uint64_t code_offset;
  };
  std::vector<KernelInfo> kernels;
  for (uint64_t off = 0; off + 256 <= elf_info.text_size; off += 256) {
    uint64_t entry_offset;
    std::memcpy(&entry_offset, text + off + 16, 8);
    if (entry_offset == 256)
      kernels.push_back({off, off + 256});
  }

  if (kernels.empty()) {
    uint64_t code_offset_in_text = 0;
    uint64_t text_vaddr = 0;
    if (elf_info.text_idx >= 0)
      text_vaddr = elf_info.sections[elf_info.text_idx].addr;
    for (const auto& sec : elf_info.sections) {
      if (sec.name == ".rodata" && sec.size >= 64) {
        for (uint64_t off = 0; off + 64 <= sec.size; off += 64) {
          const uint8_t* desc = elf + sec.offset + off;
          uint64_t entry;
          std::memcpy(&entry, desc + 16, 8);
          if (entry > 0 && entry < 1000000) {
            uint64_t kd_vaddr = sec.addr + off;
            uint64_t code_vaddr = kd_vaddr + entry;
            if (code_vaddr >= text_vaddr)
              code_offset_in_text = code_vaddr - text_vaddr;
            break;
          }
        }
        break;
      }
    }
    std::cerr << "hotswap: transpile: no embedded descriptors in .text, "
              << "code at .text internal offset " << code_offset_in_text << "\n";
    kernels.push_back({code_offset_in_text, code_offset_in_text});
  } else {
    std::cerr << "hotswap: transpile: found " << kernels.size()
              << " embedded kernel descriptor(s)\n";
  }

  std::string translated_asm;
  translated_asm += ".text\n";

  for (size_t ki = 0; ki < kernels.size(); ++ki) {
    auto& kern = kernels[ki];

    uint64_t emit_end = kern.code_offset;
    uint64_t emit_start = (kern.desc_offset != kern.code_offset) ? kern.desc_offset : 0;
    for (uint64_t i = emit_start; i < emit_end; i += 4) {
      if (i + 4 > elf_info.text_size) break;
      uint32_t word;
      std::memcpy(&word, text + i, 4);
      std::ostringstream oss;
      oss << ".long 0x" << std::hex << word;
      translated_asm += oss.str() + "\n";
    }

    uint32_t num_vgprs12 = 8;
    uint32_t num_sgprs12 = 16;
    {
      uint32_t rsrc1_src = 0;
      if (kern.desc_offset != kern.code_offset &&
          kern.desc_offset + 52 <= elf_info.text_size)
        std::memcpy(&rsrc1_src, text + kern.desc_offset + 48, 4);
      else {
        for (const auto& sec : elf_info.sections) {
          if (sec.name == ".rodata" && sec.size >= 64) {
            for (uint64_t off = 0; off + 64 <= sec.size; off += 64) {
              const uint8_t* desc = elf + sec.offset + off;
              uint64_t entry;
              std::memcpy(&entry, desc + 16, 8);
              if (entry > 0 && entry < 1000000) { std::memcpy(&rsrc1_src, desc + 48, 4); break; }
            }
            break;
          }
        }
      }
      if (rsrc1_src) {
        num_vgprs12 = ((rsrc1_src & 0x3Fu) + 1u) * 12u;
        num_sgprs12 = (((rsrc1_src >> 6) & 0x3Fu) + 1u) * 16u;
        std::cerr << "hotswap: transpile: GFX12 RSRC1=0x" << std::hex
                  << rsrc1_src << std::dec << " → num_vgprs12=" << num_vgprs12
                  << " num_sgprs12=" << num_sgprs12 << "\n";
      }
    }
    if (num_vgprs12 < 8u) num_vgprs12 = 8u;
    if (num_sgprs12 < 16u) num_sgprs12 = 16u;

    // Scan MSGPACK for .sgpr_count
    {
      const char* key = ".sgpr_count";
      size_t key_len = 11;
      for (const auto& sec : elf_info.sections) {
        if (sec.name == ".note" && sec.size > key_len + 2) {
          for (size_t i = 0; i + key_len + 1 < sec.size; i++) {
            if (std::memcmp(elf + sec.offset + i, key, key_len) == 0) {
              uint8_t val = elf[sec.offset + i + key_len];
              uint32_t sgpr_count = 0;
              if (val <= 0x7F) sgpr_count = val;
              else if (val == 0xCC) sgpr_count = elf[sec.offset + i + key_len + 1];
              else if (val == 0xCD) {
                uint16_t v16;
                std::memcpy(&v16, elf + sec.offset + i + key_len + 1, 2);
                sgpr_count = (v16 >> 8) | ((v16 & 0xFF) << 8);
              }
              if (sgpr_count > num_sgprs12) num_sgprs12 = sgpr_count;
              break;
            }
          }
          break;
        }
      }
    }

    uint32_t save_vgpr_x = num_vgprs12;
    uint32_t save_vgpr_y = num_vgprs12 + 1u;
    uint32_t cmpx_temp_sgpr = num_sgprs12;
    const std::string sv_x = "v" + std::to_string(save_vgpr_x);
    const std::string sv_y = "v" + std::to_string(save_vgpr_y);

    uint64_t code_end = elf_info.text_size;
    if (ki + 1 < kernels.size())
      code_end = kernels[ki + 1].desc_offset;

    struct SourceInstr {
      std::string text;
      uint64_t pc_offset;
      uint32_t size;
      llvm::MCInst inst;
      bool valid_inst;
    };
    std::vector<SourceInstr> source_instrs;
    std::vector<std::string> source_lines;
    uint64_t pos = kern.code_offset;
    while (pos < code_end) {
      llvm::MCInst inst;
      uint64_t inst_size = 0;
      llvm::ArrayRef<uint8_t> bytes(text + pos, code_end - pos);
      auto status = src_state.disasm->getInstruction(inst, inst_size, bytes, pos, llvm::nulls());
      if (status == llvm::MCDisassembler::Fail) {
        if (pos + 4 <= code_end) {
          uint32_t word;
          std::memcpy(&word, text + pos, 4);
          std::ostringstream oss;
          oss << ".long 0x" << std::hex << word;
          source_instrs.push_back({oss.str(), pos, 4, llvm::MCInst(), false});
          source_lines.push_back(oss.str());
        }
        pos += 4;
        ++stats.total_instructions;
        continue;
      }
      std::string asm_text;
      if (src_state.printer) {
        llvm::raw_string_ostream rso(asm_text);
        src_state.printer->printInst(&inst, 0, "", *src_state.STI, rso);
        rso.flush();
      }
      size_t start = asm_text.find_first_not_of(" \t");
      if (start != std::string::npos && start > 0)
        asm_text = asm_text.substr(start);
      if (!asm_text.empty()) {
        source_instrs.push_back({asm_text, pos, static_cast<uint32_t>(inst_size), inst, true});
        source_lines.push_back(asm_text);
      }
      pos += inst_size;
      ++stats.total_instructions;
    }

    std::cerr << "hotswap: transpile: kernel " << ki << ": disassembled "
              << source_lines.size() << " instructions\n";

    // Branch label resolution
    std::map<uint64_t, std::string> branch_labels;
    int label_counter = 0;
    for (size_t i = 0; i < source_instrs.size(); ++i) {
      auto& info = source_instrs[i];
      std::string m = TranspileExtractMnemonic(info.text);
      bool is_branch = (m.find("s_branch") == 0 || m.find("s_cbranch_") == 0);
      if (!is_branch) continue;
      std::string ops = info.text.substr(info.text.find(m) + m.size());
      size_t s = ops.find_first_not_of(" \t");
      if (s == std::string::npos) continue;
      std::string offset_str = ops.substr(s);
      if (offset_str.find(".L_") == 0) continue;
      try {
        int64_t raw = std::stoll(offset_str, nullptr, 0);
        int64_t simm16 = static_cast<int16_t>(raw & 0xFFFF);
        uint64_t target_pc = info.pc_offset + 4 + simm16 * 4;
        uint64_t snapped_pc = target_pc;
        bool found = false;
        for (const auto& si : source_instrs) {
          if (si.pc_offset >= target_pc) { snapped_pc = si.pc_offset; found = true; break; }
        }
        if (!found && !source_instrs.empty())
          snapped_pc = source_instrs.back().pc_offset;
        if (branch_labels.find(snapped_pc) == branch_labels.end())
          branch_labels[snapped_pc] = ".L_br" + std::to_string(label_counter++);
      } catch (...) {}
    }

    // TTMP taint analysis
    std::vector<SourceInstrForTaint> taint_input;
    taint_input.reserve(source_instrs.size());
    for (const auto& si : source_instrs)
      taint_input.push_back({si.text, si.inst, si.valid_inst});
    auto taint_results = AnalyzeTTMPTaint(taint_input, *src_state.MCII, *src_state.MRI);

    for (auto& tr : taint_results) {
      if (tr.action == TaintAction::Replace) {
        if (tr.replace_src == "v5") tr.replace_src = sv_x;
        else if (tr.replace_src == "v4") tr.replace_src = sv_y;
      }
    }

    translated_asm += "v_mov_b32_e32 " + sv_x + ", s2 ; save workgroup_id_x\n";
    translated_asm += "v_mov_b32_e32 " + sv_y + ", s3 ; save workgroup_id_y\n";

    std::vector<std::pair<std::string, std::string>> replace_regs;
    for (auto& tr : taint_results) {
      if (tr.action == TaintAction::Replace && !tr.replace_dst.empty())
        replace_regs.emplace_back(tr.replace_dst, tr.replace_src);
    }

    int early_exit_after = 0;
    if (const char* ee = std::getenv("HSA_HOTSWAP_EARLY_EXIT"))
      early_exit_after = std::atoi(ee);
    int emitted_count = 0;
    bool early_exit_done = false;

    for (size_t ii = 0; ii < source_lines.size(); ++ii) {
      const auto& line = source_lines[ii];

      if (ii < source_instrs.size()) {
        auto lbl = branch_labels.find(source_instrs[ii].pc_offset);
        if (lbl != branch_labels.end())
          translated_asm += lbl->second + ":\n";
      }

      if (ii < taint_results.size()) {
        if (taint_results[ii].action == TaintAction::Skip) continue;
        if (taint_results[ii].action == TaintAction::Replace) {
          auto& tr = taint_results[ii];
          translated_asm += "v_readfirstlane_b32 " + tr.replace_dst + ", " + tr.replace_src + "\n";
          continue;
        }
      }

      auto translated_lines = TranslateInstruction(line, src_cpu, tgt_cpu,
                                                    save_vgpr_y + 1, cmpx_temp_sgpr, false);

      if (ii < source_instrs.size() && !branch_labels.empty()) {
        for (auto& t : translated_lines) {
          std::string tm = TranspileExtractMnemonic(t);
          if (tm.find("s_branch") == 0 || tm.find("s_cbranch_") == 0) {
            size_t op_pos = t.find(tm) + tm.size();
            std::string ops = t.substr(op_pos);
            size_t s = ops.find_first_not_of(" \t");
            if (s != std::string::npos) {
              std::string off_str = ops.substr(s);
              if (off_str.find(".L_") != 0) {
                try {
                  int64_t raw = std::stoll(off_str, nullptr, 0);
                  int64_t simm16 = static_cast<int16_t>(raw & 0xFFFF);
                  uint64_t target = source_instrs[ii].pc_offset + 4 + simm16 * 4;
                  uint64_t snapped = target;
                  for (const auto& si : source_instrs) {
                    if (si.pc_offset >= target) { snapped = si.pc_offset; break; }
                  }
                  auto lbl = branch_labels.find(snapped);
                  if (lbl != branch_labels.end()) t = tm + " " + lbl->second;
                } catch (...) {}
              }
            }
          }
        }
      }

      bool translated_had_saveexec = false;
      for (const auto& t : translated_lines) {
        if (t.empty()) continue;
        if (t.find("saveexec") != std::string::npos) translated_had_saveexec = true;
        std::string m = TranspileExtractMnemonic(line);
        if (t.find("UNSUPPORTED") != std::string::npos) ++stats.unsupported_skipped;
        else if (t != line) {
          std::string nm = TranspileExtractMnemonic(t);
          if (IsWaitInstruction(m)) ++stats.translated_waitcnt;
          else if (nm != m) ++stats.translated_renamed;
          else if (t.find("exec_hi") != std::string::npos) ++stats.translated_exec;
          else ++stats.translated_passthrough;
        } else {
          ++stats.translated_passthrough;
        }
        translated_asm += t + "\n";
      }
      ++emitted_count;

      if (early_exit_after > 0 && emitted_count >= early_exit_after && !early_exit_done
          && source_lines.size() > 400) {
        for (size_t jj = ii + 1; jj < source_instrs.size(); jj++) {
          auto lbl = branch_labels.find(source_instrs[jj].pc_offset);
          if (lbl != branch_labels.end()) translated_asm += lbl->second + ":\n";
        }
        translated_asm += ".L_exit:\ns_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)\n";
        translated_asm += "s_endpgm ; EARLY EXIT after " + std::to_string(emitted_count) + " instrs\n";
        early_exit_done = true;
        break;
      }

      if (ii < source_instrs.size()) {
        bool has_vcmpx = false;
        for (const auto& t : translated_lines)
          if (t.find("v_cmpx_") != std::string::npos) { has_vcmpx = true; break; }
        if (has_vcmpx) {
          if (source_lines.size() > 400) {
            translated_asm += "s_mov_b32 exec_hi, 0\n";
          } else {
            bool next_is_execz = false;
            for (size_t nxt = ii + 1; nxt < source_instrs.size(); nxt++) {
              std::string nm = TranspileExtractMnemonic(source_instrs[nxt].text);
              if (nm.find("s_delay") == 0 || nm.find("s_wait") == 0 || nm.find("s_nop") == 0 || nm.find("s_clause") == 0) continue;
              if (nm == "s_cbranch_execz") next_is_execz = true;
              break;
            }
            if (!next_is_execz) translated_asm += "s_mov_b32 exec_hi, 0\n";
          }
        }
      }

      if (translated_had_saveexec) {
        for (auto& r : replace_regs)
          translated_asm += "v_readfirstlane_b32 " + r.first + ", " + r.second + "\n";
      }
    }
  }

  std::cerr << "hotswap: transpile: translated "
            << stats.total_instructions << " instructions → "
            << stats.translated_passthrough << " passthrough, "
            << stats.translated_renamed << " renamed, "
            << stats.translated_waitcnt << " waitcnt, "
            << stats.translated_exec << " exec-widened, "
            << stats.unsupported_skipped << " unsupported\n";

  // Post-processing
  {
    auto replaceAll = [](std::string& s, const std::string& from, const std::string& to) {
      size_t pos = 0;
      while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
      }
    };
    // VCC branch fix
    {
      std::string tmp;
      std::istringstream vfix_iss(translated_asm);
      std::string vfix_line;
      while (std::getline(vfix_iss, vfix_line)) {
        if (vfix_line.find("s_cbranch_vccz") != std::string::npos ||
            vfix_line.find("s_cbranch_vccnz") != std::string::npos)
          tmp += "s_mov_b32 vcc_hi, 0\n";
        tmp += vfix_line + "\n";
      }
      translated_asm = tmp;
    }
    replaceAll(translated_asm, "v_add_nc_u32 ", "v_add_u32_e32 ");
    replaceAll(translated_asm, "v_sub_nc_u32 ", "v_sub_u32_e32 ");
    // Strip explicit VCC mask from v_cndmask_b32_e32
    {
      std::string tmp;
      std::istringstream vcc_iss(translated_asm);
      std::string vcc_line;
      while (std::getline(vcc_iss, vcc_line)) {
        if (vcc_line.find("v_cndmask_b32_e32") != std::string::npos) {
          size_t vcc_pos = vcc_line.rfind(", vcc_lo");
          if (vcc_pos == std::string::npos) vcc_pos = vcc_line.rfind(", vcc");
          if (vcc_pos != std::string::npos)
            vcc_line = vcc_line.substr(0, vcc_pos);
        }
        tmp += vcc_line + "\n";
      }
      translated_asm = tmp;
    }
    // Fix s_load from s[8:9]+0xc with saved kernarg ptr
    {
      if (translated_asm.find("s_load_dword s1, s[8:9], 0xc") != std::string::npos ||
          translated_asm.find("s_load_dword s0, s[8:9], 0xc") != std::string::npos) {
        std::string ka_pair = "s[30:31]";
        size_t ka_pos = translated_asm.find("; save kernarg ptr lo");
        if (ka_pos != std::string::npos) {
          size_t s_pos = translated_asm.rfind("s_mov_b32 s", ka_pos);
          if (s_pos != std::string::npos) {
            size_t n_start = s_pos + 11;
            size_t n_end = translated_asm.find(',', n_start);
            int lo = std::stoi(translated_asm.substr(n_start, n_end - n_start));
            ka_pair = "s[" + std::to_string(lo) + ":" + std::to_string(lo + 1) + "]";
          }
        } else {
          size_t insert_pos = translated_asm.find("; save workgroup_id_y\n");
          if (insert_pos != std::string::npos) {
            insert_pos = translated_asm.find('\n', insert_pos) + 1;
            translated_asm.insert(insert_pos,
              "s_mov_b32 s30, s0 ; save kernarg ptr lo\n"
              "s_mov_b32 s31, s1 ; save kernarg ptr hi\n");
          }
        }
        replaceAll(translated_asm, "s_load_dword s1, s[8:9], 0xc",
                   "s_load_dword s1, " + ka_pair + ", 0x3c");
        replaceAll(translated_asm, "s_load_dword s0, s[8:9], 0xc",
                   "s_load_dword s0, " + ka_pair + ", 0x3c");
      }
    }
  }

  if (std::getenv("HSA_HOTSWAP_DUMP")) {
    std::cerr << "hotswap: transpile: === TRANSLATED ASSEMBLY ===\n"
              << translated_asm
              << "hotswap: transpile: === END ASSEMBLY ===\n";
  }

  // Assemble translated text for target ISA
  LLVMState &tgt_state = InitLLVMCached(target_isa);
  if (!tgt_state.valid) {
    std::cerr << "hotswap: transpile: failed to init target ISA '" << target_isa << "'\n";
    return AMD_COMGR_STATUS_ERROR;
  }

  llvm::Triple tgt_triple("amdgcn-amd-amdhsa");
  llvm::MCTargetOptions mc_opts;

  tgt_state.Ctx->reset();

  llvm::StringRef asm_ref(translated_asm);
  auto buf = llvm::MemoryBuffer::getMemBuffer(asm_ref, "", false);
  llvm::SourceMgr src_mgr;
  src_mgr.AddNewSourceBuffer(std::move(buf), llvm::SMLoc());

  std::string data;
  auto data_stream = std::make_unique<llvm::raw_string_ostream>(data);
  auto bos = std::make_unique<llvm::buffer_ostream>(*data_stream);

#if LLVM_VERSION_MAJOR > 14
  llvm::MCCodeEmitter* ce = tgt_state.target->createMCCodeEmitter(*tgt_state.MCII, *tgt_state.Ctx);
#else
  llvm::MCCodeEmitter* ce = tgt_state.target->createMCCodeEmitter(*tgt_state.MCII, *tgt_state.MRI, *tgt_state.Ctx);
#endif
  llvm::MCAsmBackend* mab = tgt_state.target->createMCAsmBackend(*tgt_state.STI, *tgt_state.MRI, mc_opts);

  if (!ce || !mab) {
    std::cerr << "hotswap: transpile: failed to create code emitter/backend\n";
    return AMD_COMGR_STATUS_ERROR;
  }

#if LLVM_VERSION_MAJOR > 20
  auto streamer = std::unique_ptr<llvm::MCStreamer>(
      tgt_state.target->createMCObjectStreamer(
          tgt_triple, *tgt_state.Ctx,
          std::unique_ptr<llvm::MCAsmBackend>(mab),
          mab->createObjectWriter(*bos),
          std::unique_ptr<llvm::MCCodeEmitter>(ce), *tgt_state.STI));
#else
  auto streamer = std::unique_ptr<llvm::MCStreamer>(
      tgt_state.target->createMCObjectStreamer(
          tgt_triple, *tgt_state.Ctx,
          std::unique_ptr<llvm::MCAsmBackend>(mab),
          mab->createObjectWriter(*bos),
          std::unique_ptr<llvm::MCCodeEmitter>(ce), *tgt_state.STI,
          mc_opts.MCRelaxAll, mc_opts.MCIncrementalLinkerCompatible, false));
#endif

  if (!streamer) {
    std::cerr << "hotswap: transpile: failed to create MC streamer\n";
    return AMD_COMGR_STATUS_ERROR;
  }

  auto parser = std::unique_ptr<llvm::MCAsmParser>(
      llvm::createMCAsmParser(src_mgr, *tgt_state.Ctx, *streamer, *tgt_state.MAI));
  auto tap = std::unique_ptr<llvm::MCTargetAsmParser>(
      tgt_state.target->createMCAsmParser(*tgt_state.STI, *parser, *tgt_state.MCII, mc_opts));
  if (!tap) {
    std::cerr << "hotswap: transpile: failed to create target asm parser\n";
    return AMD_COMGR_STATUS_ERROR;
  }
  parser->setTargetParser(*tap);

  bool asm_failed = parser->Run(true);
  tap.reset();
  parser.reset();
  streamer.reset();
  bos.reset();
  data_stream->flush();

  if (asm_failed)
    std::cerr << "hotswap: transpile: assembly failed for " << tgt_cpu << "\n";

  if (data.size() < 64) {
    std::cerr << "hotswap: transpile: assembled output too small (" << data.size() << " bytes)\n";
    return AMD_COMGR_STATUS_ERROR;
  }

  // Extract .text from assembled ELF
  const uint8_t* asm_elf = reinterpret_cast<const uint8_t*>(data.data());
  ElfInfo asm_info;
  if (!ParseElfInfo(asm_elf, data.size(), asm_info)) {
    std::cerr << "hotswap: transpile: failed to parse assembled ELF\n";
    return AMD_COMGR_STATUS_ERROR;
  }

  const uint8_t* new_text = asm_elf + asm_info.text_offset;
  uint64_t new_text_size = asm_info.text_size;

  std::cerr << "hotswap: transpile: assembled " << new_text_size
            << " bytes (original: " << elf_info.text_size << ")\n";

  // Replace .text in a NEW writable ELF buffer
  {
    size_t new_elf_size = size;
    uint8_t* new_elf = static_cast<uint8_t*>(std::malloc(new_elf_size));
    if (!new_elf) return AMD_COMGR_STATUS_ERROR;
    std::memcpy(new_elf, elf, size);

    if (new_text_size <= elf_info.text_size) {
      std::memcpy(new_elf + elf_info.text_offset, new_text, new_text_size);
      uint8_t nop_bytes[] = {0x00, 0x00, 0x80, 0xBF};
      for (uint64_t i = new_text_size; i + 4 <= elf_info.text_size; i += 4)
        std::memcpy(new_elf + elf_info.text_offset + i, nop_bytes, 4);
    } else {
      uint64_t available = elf_info.text_size;
      uint64_t after_text = elf_info.text_offset + elf_info.text_size;
      uint64_t next_section_start = new_elf_size;
      uint16_t e_shentsize, e_shnum;
      std::memcpy(&e_shentsize, new_elf + 58, 2);
      std::memcpy(&e_shnum, new_elf + 60, 2);
      uint64_t e_shoff;
      std::memcpy(&e_shoff, new_elf + 40, 8);
      for (uint16_t i = 0; i < e_shnum; ++i) {
        uint64_t sh_off = e_shoff + i * e_shentsize;
        if (sh_off + e_shentsize > new_elf_size) break;
        uint64_t sec_offset, sec_size;
        std::memcpy(&sec_offset, new_elf + sh_off + 24, 8);
        std::memcpy(&sec_size, new_elf + sh_off + 32, 8);
        if (sec_offset > elf_info.text_offset && sec_offset < next_section_start && sec_size > 0)
          next_section_start = sec_offset;
      }
      available = next_section_start - elf_info.text_offset;

      if (new_text_size <= available) {
        std::memcpy(new_elf + elf_info.text_offset, new_text, new_text_size);
        for (uint16_t i = 0; i < e_shnum; ++i) {
          uint64_t sh_off = e_shoff + i * e_shentsize;
          if (static_cast<int>(i) == elf_info.text_idx) {
            std::memcpy(new_elf + sh_off + 32, &new_text_size, 8);
            break;
          }
        }
      } else {
        uint64_t text_end = elf_info.text_offset + elf_info.text_size;
        uint64_t delta = ((new_text_size - elf_info.text_size + 255u) / 256u) * 256u;
        uint64_t grown_size = new_elf_size + delta;
        uint8_t* grown = (uint8_t*)calloc(1, grown_size);
        std::memcpy(grown, new_elf, text_end);
        std::memcpy(grown + elf_info.text_offset, new_text, new_text_size);
        uint64_t new_sec_size = elf_info.text_size + delta;
        for (uint64_t p = new_text_size; p < new_sec_size; p += 4) {
          uint8_t nop[] = {0x00, 0x00, 0x80, 0xBF};
          std::memcpy(grown + elf_info.text_offset + p, nop, 4);
        }
        uint64_t tail = new_elf_size - text_end;
        if (tail > 0) std::memcpy(grown + text_end + delta, new_elf + text_end, tail);
        std::memcpy(&e_shoff, grown + 40, 8);
        e_shoff += delta;
        std::memcpy(grown + 40, &e_shoff, 8);
        for (uint16_t i = 0; i < e_shnum; ++i) {
          uint64_t sh_off = e_shoff + i * e_shentsize;
          uint64_t sec_offset;
          std::memcpy(&sec_offset, grown + sh_off + 24, 8);
          if (sec_offset > elf_info.text_offset) {
            sec_offset += delta;
            std::memcpy(grown + sh_off + 24, &sec_offset, 8);
          }
          if (static_cast<int>(i) == elf_info.text_idx)
            std::memcpy(grown + sh_off + 32, &new_sec_size, 8);
        }
        uint64_t e_phoff;
        uint16_t e_phentsize, e_phnum;
        std::memcpy(&e_phoff, grown + 32, 8);
        std::memcpy(&e_phentsize, grown + 54, 2);
        std::memcpy(&e_phnum, grown + 56, 2);
        for (uint16_t i = 0; i < e_phnum; ++i) {
          uint64_t ph_off = e_phoff + i * e_phentsize;
          if (ph_off + 56 > grown_size) break;
          uint64_t p_offset, p_filesz, p_memsz;
          std::memcpy(&p_offset, grown + ph_off + 8, 8);
          std::memcpy(&p_filesz, grown + ph_off + 32, 8);
          std::memcpy(&p_memsz, grown + ph_off + 40, 8);
          if (p_offset == elf_info.text_offset) {
            p_filesz += delta; p_memsz += delta;
            std::memcpy(grown + ph_off + 32, &p_filesz, 8);
            std::memcpy(grown + ph_off + 40, &p_memsz, 8);
          } else if (p_offset > elf_info.text_offset) {
            p_offset += delta;
            std::memcpy(grown + ph_off + 8, &p_offset, 8);
          }
        }
        free(new_elf);
        new_elf = grown;
        new_elf_size = grown_size;
      }
    }

    *out_data = new_elf;
    *out_size = new_elf_size;

    // Patch kernel descriptors for wave64
    ElfInfo updated_info;
    if (ParseElfInfo(new_elf, new_elf_size, updated_info)) {
      PatchKernelDescriptorsForWave64(new_elf, new_elf_size, updated_info);
      for (auto& sec : updated_info.sections) {
        if (sec.name == ".rodata" && sec.size >= 64) {
          for (uint64_t off = 0; off + 64 <= sec.size; off += 64) {
            uint8_t* desc = new_elf + sec.offset + off;
            uint64_t entry;
            std::memcpy(&entry, desc + 16, 8);
            if (entry == 0 || entry > 1000000) continue;
            uint32_t rsrc1;
            std::memcpy(&rsrc1, desc + 48, 4);
            rsrc1 &= 0x00FFFFFFu;
            rsrc1 |= (1u << 21) | (1u << 23);
            uint32_t vgpr_field12 = rsrc1 & 0x3Fu;
            uint32_t sgpr_field12_rd = (rsrc1 >> 6) & 0x3Fu;
            uint32_t num_vgprs = (vgpr_field12 + 1u) * 12u;
            if (num_vgprs < 8u) num_vgprs = 8u;
            num_vgprs += 4u;
            uint32_t gfx9_vgpr = (num_vgprs / 4u) - 1u;
            if (gfx9_vgpr > 62u) gfx9_vgpr = 62u;
            uint32_t rsrc1_vgpr_field = gfx9_vgpr;
            if (rsrc1_vgpr_field > 63u) rsrc1_vgpr_field = 63u;
            rsrc1 &= ~0xFFFu;
            rsrc1 |= (rsrc1_vgpr_field << 6u);
            {
              uint32_t num_sgprs_rd = (sgpr_field12_rd + 1u) * 16u + 8u;
              const char* sgpr_key = ".sgpr_count";
              for (size_t si = 0; si + 12 < new_elf_size; si++) {
                if (std::memcmp(new_elf + si, sgpr_key, 11) == 0) {
                  uint8_t val = new_elf[si + 11];
                  uint32_t sc = (val <= 0x7F) ? val : (val == 0xCC ? new_elf[si+12] : 0);
                  if (sc + 8 > num_sgprs_rd) num_sgprs_rd = sc + 8;
                  break;
                }
              }
              uint32_t gfx9_sgpr = (num_sgprs_rd / 8u) - 1u;
              if (gfx9_sgpr > 12u) gfx9_sgpr = 12u;
              rsrc1 |= gfx9_sgpr;
            }
            std::memcpy(desc + 48, &rsrc1, 4);
            uint32_t rsrc2;
            std::memcpy(&rsrc2, desc + 52, 4);
            rsrc2 |= (1u << 7) | (1u << 8) | (1u << 9);
            std::memcpy(desc + 52, &rsrc2, 4);
            uint16_t props;
            std::memcpy(&props, desc + 56, 2);
            props &= ~(1u << 10);
            std::memcpy(desc + 56, &props, 2);
            uint32_t rsrc3 = gfx9_vgpr;
            std::memcpy(desc + 44, &rsrc3, 4);
          }
        }
      }
    }

    // Patch ELF metadata
    PatchElfMetadata(new_elf, new_elf_size, tgt_cpu);
  }

  result->rules_matched = stats.translated_passthrough + stats.translated_renamed + stats.translated_waitcnt;

  std::cerr << "hotswap: transpile: complete (" << src_cpu << " → " << tgt_cpu << ")\n";

  if (auto* dump_path = std::getenv("HSA_HOTSWAP_DUMP_ELF")) {
    FILE* fp = fopen(dump_path, "wb");
    if (fp) {
      fwrite(*out_data, 1, *out_size, fp);
      fclose(fp);
      std::cerr << "hotswap: transpile: dumped patched ELF to " << dump_path << "\n";
    }
  }

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
