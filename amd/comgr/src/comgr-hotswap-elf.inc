// comgr-hotswap-elf.inc — ELF types, parsing, and binary helpers
// Included inside the anonymous namespace of comgr-hotswap.cpp.
// Not a standalone compilation unit.
//
// NOTE: ElfSection, ElfSymbol, ElfInfo, and ExtractCPU are derived from the
// canonical shared header at:
//   rocr-runtime/hotswap/hotswap_shared_types.h
// This copy exists because comgr-hotswap-elf.inc is included inside an
// anonymous namespace, so it cannot directly #include a namespaced header.
// TODO: Unify via build-system include-path changes.

// ── MallocBuffer RAII wrapper ────────────────────────────────────────────────

struct MallocBuffer {
  uint8_t *data = nullptr;
  size_t size = 0;

  MallocBuffer() = default;
  MallocBuffer(size_t n) : data(static_cast<uint8_t *>(std::malloc(n))), size(n) {}
  ~MallocBuffer() { std::free(data); }

  MallocBuffer(MallocBuffer &&o) noexcept : data(o.data), size(o.size) {
    o.data = nullptr; o.size = 0;
  }
  MallocBuffer &operator=(MallocBuffer &&o) noexcept {
    if (this != &o) { std::free(data); data = o.data; size = o.size; o.data = nullptr; o.size = 0; }
    return *this;
  }

  MallocBuffer(const MallocBuffer &) = delete;
  MallocBuffer &operator=(const MallocBuffer &) = delete;

  explicit operator bool() const { return data != nullptr; }
  uint8_t *release() { uint8_t *p = data; data = nullptr; size = 0; return p; }
};

// ── Logging ──────────────────────────────────────────────────────────────────

enum class HotswapLogLevel : int { Silent = 0, Error = 1, Info = 2, Debug = 3 };

static HotswapLogLevel GetHotswapLogLevel() {
  static HotswapLogLevel level = []() {
    const char *env = std::getenv("HSA_HOTSWAP_LOG_LEVEL");
    if (env) {
      int v = std::atoi(env);
      if (v >= 0 && v <= 3) return static_cast<HotswapLogLevel>(v);
    }
    return HotswapLogLevel::Info;
  }();
  return level;
}

static std::ostream &HotswapLog(HotswapLogLevel level) {
  static std::ostream null_stream(nullptr);
  if (static_cast<int>(level) <= static_cast<int>(GetHotswapLogLevel()))
    return std::cerr;
  return null_stream;
}

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

[[nodiscard]] static bool EncodeSBranch(uint64_t from_offset, uint64_t to_offset,
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

[[nodiscard]] static bool ParseElfInfo(const uint8_t *elf, size_t elf_size, ElfInfo &info) {
  using ELFT = llvm::object::ELF64LE;
  auto elf_or_err = llvm::object::ELFFile<ELFT>::create(
      llvm::StringRef(reinterpret_cast<const char *>(elf), elf_size));
  if (!elf_or_err) {
    llvm::consumeError(elf_or_err.takeError());
    return false;
  }
  const auto &elf_file = *elf_or_err;

  auto sections_or_err = elf_file.sections();
  if (!sections_or_err) {
    llvm::consumeError(sections_or_err.takeError());
    return false;
  }
  auto shdrs = *sections_or_err;

  for (const auto &shdr : shdrs) {
    ElfSection sec;
    sec.type = shdr.sh_type;
    sec.offset = shdr.sh_offset;
    sec.size = shdr.sh_size;
    sec.addr = shdr.sh_addr;
    sec.name_idx = shdr.sh_name;

    auto name_or_err = elf_file.getSectionName(shdr);
    if (name_or_err)
      sec.name = name_or_err->str();
    else
      llvm::consumeError(name_or_err.takeError());

    if (sec.name == ".text" && sec.offset + sec.size <= elf_size) {
      info.text_section_idx = static_cast<int>(info.sections.size());
      info.text_idx = info.text_section_idx;
      info.text_offset = sec.offset;
      info.text_size = sec.size;
      info.text_addr = sec.addr;
    }

    info.sections.push_back(std::move(sec));
  }

  size_t num_sections = info.sections.size();
  for (size_t i = 0; i < num_sections; ++i) {
    if (info.sections[i].type != 2 && info.sections[i].type != 11)
      continue;

    const auto &sym_shdr = *(shdrs.begin() + i);

    auto syms_or_err = elf_file.symbols(&sym_shdr);
    if (!syms_or_err) {
      llvm::consumeError(syms_or_err.takeError());
      continue;
    }

    auto strtab_or_err = elf_file.getStringTableForSymtab(sym_shdr, shdrs);
    if (!strtab_or_err) {
      llvm::consumeError(strtab_or_err.takeError());
      continue;
    }

    for (const auto &sym : *syms_or_err) {
      ElfSymbol esym;
      esym.info = sym.st_info;
      esym.shndx = sym.st_shndx;
      esym.value = sym.st_value;
      esym.size = sym.st_size;

      auto sym_name_or_err = sym.getName(*strtab_or_err);
      if (sym_name_or_err)
        esym.name = sym_name_or_err->str();
      else
        llvm::consumeError(sym_name_or_err.takeError());

      info.symbols.push_back(std::move(esym));
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

[[nodiscard]] static bool ApplyByteReplace(const RewriteRule &rule, uint64_t inst_offset,
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
    if (sym.value < sec.addr) continue;
    uint64_t kd_file_offset = sec.offset + (sym.value - sec.addr);
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

static MallocBuffer GrowElfWithTrampolines(const uint8_t *elf, size_t elf_size,
                                            const ElfInfo &elf_info,
                                            const std::vector<Trampoline> &trampolines) {
  size_t tramp_total = 0;
  for (auto &t : trampolines)
    tramp_total += t.bytes.size();
  if (tramp_total == 0)
    return {};
  if (tramp_total > SIZE_MAX - elf_size)
    return {};

  size_t new_elf_size = elf_size + tramp_total;
  MallocBuffer buf(new_elf_size);
  if (!buf)
    return {};

  uint8_t *new_elf = buf.data;

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

  return buf;
}
