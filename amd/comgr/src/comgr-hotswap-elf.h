// comgr-hotswap-elf.h — ELF types, parsing, and binary helpers
// Included inside the anonymous namespace of comgr-hotswap.cpp.
// Not a standalone compilation unit.
//
// KEEP IN SYNC: ElfSection, ElfSymbol, ElfInfo, ParseElfInfo, ExtractCPU,
// and FindKernelAtOffset are duplicated in
// rocr-runtime/hotswap/hotswap_core.{hpp,cpp}.

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

    if (sec.offset + sec.size > elf_size) continue;

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
  if (inst_offset + inst_size > text_size) return false;
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
  if (tramp_total > SIZE_MAX - elf_size)
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
