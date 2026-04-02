//===- comgr-hotswap-transpiler.cpp - Cross-family ISA transpile pipeline --===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-internal.h"

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
    props = static_cast<uint16_t>((static_cast<uint32_t>(props) & ~(1u << 10)) &
                                  0xFFFFu);
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
          HotswapLog(HotswapLogLevel::Info) << "hotswap: transpile: patched wavefront_size 32 → 64\n";
        }
      }
    }
  }
}

// ── Opcode-based MCInst-to-MCInst translation ────────────────────────────────

static bool TranslateViaOpcode(const llvm::MCInst &src_inst, unsigned src_opcode,
                                const OpcodeMapper &mapper, unsigned tgt_gen,
                                const llvm::MCInstrInfo &src_MCII,
                                const llvm::MCInstrInfo &tgt_MCII,
                                llvm::MCInst &out_inst) {
  unsigned pseudo = mapper.toPseudo(src_opcode);
  unsigned tgt_opcode = OpcodeMapper::toTarget(pseudo, tgt_gen);
  if (tgt_opcode == static_cast<unsigned>(-1))
    return false;

  const llvm::MCInstrDesc &tgt_desc = tgt_MCII.get(tgt_opcode);

  out_inst.setOpcode(tgt_opcode);

  unsigned num_ops = std::min(src_inst.getNumOperands(),
                               static_cast<unsigned>(tgt_desc.getNumOperands()));
  for (unsigned i = 0; i < num_ops; ++i)
    out_inst.addOperand(src_inst.getOperand(i));

  return true;
}

// ── Direct MCInst encoding ───────────────────────────────────────────────────

static std::vector<uint8_t> EncodeMCInst(const llvm::MCInst &inst,
                                          const LLVMState &state) {
  if (!state.CE) return {};
  llvm::SmallVector<char, 16> cb;
  llvm::SmallVector<llvm::MCFixup, 4> fixups;
  state.CE->encodeInstruction(inst, cb, fixups, *state.STI);
  return std::vector<uint8_t>(cb.begin(), cb.end());
}

// ── TranspileCodeObject ──────────────────────────────────────────────────────

amd_comgr_status_t
TranspileCodeObject(const void *elf_data, size_t elf_size,
                    const std::string &source_isa,
                    const std::string &target_isa,
                    void **out_data, size_t *out_size,
                    amd_comgr_hotswap_result_t *result) {
  TranspileStats stats;
  std::string src_cpu = ExtractCPU(source_isa);
  std::string tgt_cpu = ExtractCPU(target_isa);

  HotswapLog(HotswapLogLevel::Info) << "hotswap: transpile: " << src_cpu << " → " << tgt_cpu << "\n";

  const uint8_t* elf = static_cast<const uint8_t*>(elf_data);
  size_t size = elf_size;

  ElfInfo elf_info;
  if (!ParseElfInfo(elf, size, elf_info)) {
    HotswapLog(HotswapLogLevel::Error) << "hotswap: transpile: failed to parse ELF\n";
    return AMD_COMGR_STATUS_ERROR;
  }
  if (elf_info.text_size == 0) {
    HotswapLog(HotswapLogLevel::Info) << "hotswap: transpile: empty .text section\n";
    MallocBuffer copy(elf_size);
    if (!copy) return AMD_COMGR_STATUS_ERROR;
    std::memcpy(copy.data, elf_data, elf_size);
    *out_data = copy.release();
    *out_size = elf_size;
    return AMD_COMGR_STATUS_SUCCESS;
  }

  LLVMState src_state = InitLLVMCached(source_isa);
  if (!src_state.valid) {
    HotswapLog(HotswapLogLevel::Error) << "hotswap: transpile: failed to init source ISA '" << source_isa << "'\n";
    return AMD_COMGR_STATUS_ERROR;
  }

  LLVMState tgt_state = InitLLVMCached(target_isa);
  if (!tgt_state.valid) {
    HotswapLog(HotswapLogLevel::Error) << "hotswap: transpile: failed to init target ISA '" << target_isa << "'\n";
    return AMD_COMGR_STATUS_ERROR;
  }

  unsigned src_gen = GetEncodingFamily(src_cpu);
  unsigned tgt_gen = GetEncodingFamily(tgt_cpu);
  OpcodeMapper &mapper = GetOpcodeMapper(src_gen);

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
    HotswapLog(HotswapLogLevel::Info) << "hotswap: transpile: no embedded descriptors in .text, "
              << "code at .text internal offset " << code_offset_in_text << "\n";
    kernels.push_back({code_offset_in_text, code_offset_in_text});
  } else {
    HotswapLog(HotswapLogLevel::Info) << "hotswap: transpile: found " << kernels.size()
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
        HotswapLog(HotswapLogLevel::Debug) << "hotswap: transpile: GFX12 RSRC1=0x" << std::hex
                  << rsrc1_src << std::dec << " → num_vgprs12=" << num_vgprs12
                  << " num_sgprs12=" << num_sgprs12 << "\n";
      }
    }
    if (num_vgprs12 < 8u) num_vgprs12 = 8u;
    if (num_sgprs12 < 16u) num_sgprs12 = 16u;

    if (num_vgprs12 > 256u) {
      HotswapLog(HotswapLogLevel::Error) << "hotswap: transpile: kernel " << ki
                << " uses " << num_vgprs12 << " VGPRs (wave32), exceeds 256 "
                << "VGPR limit for wave64 target — cannot transpile\n";
      return AMD_COMGR_STATUS_ERROR;
    }

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

    HotswapLog(HotswapLogLevel::Info) << "hotswap: transpile: kernel " << ki << ": disassembled "
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
      {
        int64_t raw = 0;
        const char *fc_begin = offset_str.data();
        const char *fc_end = offset_str.data() + offset_str.size();
        int fc_base = 10;
        if (offset_str.size() > 2 && offset_str[0] == '0' &&
            (offset_str[1] == 'x' || offset_str[1] == 'X')) {
          fc_begin += 2;
          fc_base = 16;
        }
        auto [fc_p, fc_ec] = std::from_chars(fc_begin, fc_end, raw, fc_base);
        if (fc_ec == std::errc()) {
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
        }
      }
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

      // Phase 5: try opcode-based direct translation for non-control-flow
      if (ii < source_instrs.size() && source_instrs[ii].valid_inst) {
        const auto &si = source_instrs[ii];
        unsigned src_opc = si.inst.getOpcode();
        const llvm::MCInstrDesc &src_desc = src_state.MCII->get(src_opc);
        if (!src_desc.isBranch() && !src_desc.isCall() &&
            !src_desc.isTerminator() && !src_desc.isReturn()) {
          llvm::MCInst tgt_inst;
          if (TranslateViaOpcode(si.inst, src_opc, mapper, tgt_gen,
                                  *src_state.MCII, *tgt_state.MCII, tgt_inst)) {
            auto encoded = EncodeMCInst(tgt_inst, tgt_state);
            if (!encoded.empty()) {
              for (size_t b = 0; b + 4 <= encoded.size(); b += 4) {
                uint32_t word;
                std::memcpy(&word, encoded.data() + b, 4);
                std::ostringstream oss;
                oss << ".long 0x" << std::hex << word;
                translated_asm += oss.str() + "\n";
              }
              stats.translated_renamed++;
              ++emitted_count;
              continue;
            }
          }
        }
      }

      // Fall through to existing text-based translation (with opcode hint)
      unsigned src_opc = ~0u;
      const llvm::MCInstrInfo *src_mcii = nullptr;
      if (ii < source_instrs.size() && source_instrs[ii].valid_inst) {
        src_opc = source_instrs[ii].inst.getOpcode();
        src_mcii = src_state.MCII.get();
      }
      auto translated_lines = TranslateInstruction(line, src_cpu, tgt_cpu,
                                                    save_vgpr_y + 1, cmpx_temp_sgpr, false,
                                                    src_opc, src_mcii);

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
                {
                  int64_t raw = 0;
                  const char *fc_begin = off_str.data();
                  const char *fc_end = off_str.data() + off_str.size();
                  int fc_base = 10;
                  if (off_str.size() > 2 && off_str[0] == '0' &&
                      (off_str[1] == 'x' || off_str[1] == 'X')) {
                    fc_begin += 2;
                    fc_base = 16;
                  }
                  auto [fc_p, fc_ec] = std::from_chars(fc_begin, fc_end, raw, fc_base);
                  if (fc_ec == std::errc()) {
                    int64_t simm16 = static_cast<int16_t>(raw & 0xFFFF);
                    uint64_t target = source_instrs[ii].pc_offset + 4 + simm16 * 4;
                    uint64_t snapped = target;
                    for (const auto& si : source_instrs) {
                      if (si.pc_offset >= target) { snapped = si.pc_offset; break; }
                    }
                    auto lbl = branch_labels.find(snapped);
                    if (lbl != branch_labels.end()) t = tm + " " + lbl->second;
                  }
                }
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

  HotswapLog(HotswapLogLevel::Info) << "hotswap: transpile: translated "
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
    // Constant bus fix: VALU with two distinct SGPR sources
    {
      std::string tmp;
      std::istringstream cbus_iss(translated_asm);
      std::string cbus_line;
      const std::string vfix_reg = "v251";
      while (std::getline(cbus_iss, cbus_line)) {
        if (!cbus_line.empty() && cbus_line[0] == 'v' &&
            cbus_line.find("v_readfirstlane") != 0 &&
            cbus_line.find("v_writelane") != 0 &&
            cbus_line.find("v_readlane") != 0) {
          auto ops = ParseOperandList(cbus_line, TranspileExtractMnemonic(cbus_line));
          if (ops.size() >= 3) {
            std::string first_sgpr;
            size_t fix_idx = 0;
            for (size_t oi = 1; oi < ops.size(); ++oi) {
              std::string s = ops[oi];
              if (!s.empty() && s[0] == '-') s = s.substr(1);
              if (!s.empty() && s[0] == 's' && s.size() > 1 &&
                  (std::isdigit((unsigned char)s[1]) || s[1] == '[')) {
                if (first_sgpr.empty()) first_sgpr = s;
                else if (s != first_sgpr) { fix_idx = oi; break; }
              }
            }
            if (fix_idx > 0) {
              std::string op = ops[fix_idx];
              bool neg = !op.empty() && op[0] == '-';
              if (neg) op = op.substr(1);
              tmp += "v_mov_b32_e32 " + vfix_reg + ", " + op + "\n";
              ops[fix_idx] = (neg ? "-" : "") + vfix_reg;
              std::string mnem = TranspileExtractMnemonic(cbus_line);
              std::string fixed = mnem + " " + ops[0];
              for (size_t oi = 1; oi < ops.size(); ++oi) fixed += ", " + ops[oi];
              cbus_line = fixed;
            }
          }
        }
        tmp += cbus_line + "\n";
      }
      translated_asm = tmp;
    }
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
            int lo = 0;
            std::from_chars(translated_asm.data() + n_start, translated_asm.data() + n_end, lo);
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
    HotswapLog(HotswapLogLevel::Debug) << "hotswap: transpile: === TRANSLATED ASSEMBLY ===\n"
              << translated_asm
              << "hotswap: transpile: === END ASSEMBLY ===\n";
  }

  // Assemble translated text for target ISA
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

  llvm::MCCodeEmitter* ce = tgt_state.target->createMCCodeEmitter(*tgt_state.MCII, *tgt_state.Ctx);
  llvm::MCAsmBackend* mab = tgt_state.target->createMCAsmBackend(*tgt_state.STI, *tgt_state.MRI, mc_opts);

  if (!ce || !mab) {
    HotswapLog(HotswapLogLevel::Error) << "hotswap: transpile: failed to create code emitter/backend\n";
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
    HotswapLog(HotswapLogLevel::Error) << "hotswap: transpile: failed to create MC streamer\n";
    return AMD_COMGR_STATUS_ERROR;
  }

  auto parser = std::unique_ptr<llvm::MCAsmParser>(
      llvm::createMCAsmParser(src_mgr, *tgt_state.Ctx, *streamer, *tgt_state.MAI));
  auto tap = std::unique_ptr<llvm::MCTargetAsmParser>(
      tgt_state.target->createMCAsmParser(*tgt_state.STI, *parser, *tgt_state.MCII, mc_opts));
  if (!tap) {
    HotswapLog(HotswapLogLevel::Error) << "hotswap: transpile: failed to create target asm parser\n";
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
    HotswapLog(HotswapLogLevel::Error) << "hotswap: transpile: assembly failed for " << tgt_cpu << "\n";

  if (data.size() < 64) {
    HotswapLog(HotswapLogLevel::Error) << "hotswap: transpile: assembled output too small (" << data.size() << " bytes)\n";
    return AMD_COMGR_STATUS_ERROR;
  }

  // Extract .text from assembled ELF
  const uint8_t* asm_elf = reinterpret_cast<const uint8_t*>(data.data());
  ElfInfo asm_info;
  if (!ParseElfInfo(asm_elf, data.size(), asm_info)) {
    HotswapLog(HotswapLogLevel::Error) << "hotswap: transpile: failed to parse assembled ELF\n";
    return AMD_COMGR_STATUS_ERROR;
  }

  const uint8_t* new_text = asm_elf + asm_info.text_offset;
  uint64_t new_text_size = asm_info.text_size;

  HotswapLog(HotswapLogLevel::Info) << "hotswap: transpile: assembled " << new_text_size
            << " bytes (original: " << elf_info.text_size << ")\n";

  // Replace .text in a NEW writable ELF buffer
  {
    size_t new_elf_size = size;
    MallocBuffer new_buf(new_elf_size);
    if (!new_buf) return AMD_COMGR_STATUS_ERROR;
    uint8_t *new_elf = new_buf.data;
    std::memcpy(new_elf, elf, size);

    if (new_text_size <= elf_info.text_size) {
      std::memcpy(new_elf + elf_info.text_offset, new_text, new_text_size);
      uint8_t nop_bytes[] = {0x00, 0x00, 0x80, 0xBF};
      for (uint64_t i = new_text_size; i + 4 <= elf_info.text_size; i += 4)
        std::memcpy(new_elf + elf_info.text_offset + i, nop_bytes, 4);
    } else {
      uint64_t available = elf_info.text_size;
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
        MallocBuffer grown_buf(grown_size);
        if (!grown_buf) return AMD_COMGR_STATUS_ERROR;
        std::memset(grown_buf.data, 0, grown_size);
        uint8_t *grown = grown_buf.data;
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
        new_buf = std::move(grown_buf);
        new_elf = new_buf.data;
        new_elf_size = grown_size;
      }
    }

    *out_data = new_buf.release();
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
            props = static_cast<uint16_t>(
                (static_cast<uint32_t>(props) & ~(1u << 10)) & 0xFFFFu);
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

  HotswapLog(HotswapLogLevel::Info) << "hotswap: transpile: complete (" << src_cpu << " → " << tgt_cpu << ")\n";

  if (auto* dump_path = std::getenv("HSA_HOTSWAP_DUMP_ELF")) {
    FILE* fp = fopen(dump_path, "wb");
    if (fp) {
      fwrite(*out_data, 1, *out_size, fp);
      fclose(fp);
      HotswapLog(HotswapLogLevel::Debug) << "hotswap: transpile: dumped patched ELF to " << dump_path << "\n";
    }
  }

  return AMD_COMGR_STATUS_SUCCESS;
}
