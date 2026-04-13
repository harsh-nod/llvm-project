//===- comgr-hotswap-llvm.cpp - LLVM MC infrastructure, decode/encode -----===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-internal.h"

namespace {
std::once_flag g_llvm_init_flag;
std::mutex g_target_cache_mutex;
const llvm::Target *g_cached_target = nullptr;
} // namespace

static void InitLLVMTargets() {
  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUTargetMC();
  LLVMInitializeAMDGPUAsmParser();
  LLVMInitializeAMDGPUDisassembler();
}

LLVMState InitLLVMImpl(const std::string &isa_name,
                              const llvm::Target *cached_target) {
  std::call_once(g_llvm_init_flag, InitLLVMTargets);

  LLVMState state;
  state.cpu = ExtractCPU(isa_name);
  if (state.cpu.empty()) return state;

  llvm::Triple triple("amdgcn-amd-amdhsa");

  if (cached_target) {
    state.target = cached_target;
  } else {
    std::string error;
    state.target = llvm::TargetRegistry::lookupTarget("amdgcn", triple, error);
  }
  if (!state.target) return state;

  state.MRI.reset(
      state.target->createMCRegInfo(llvm::Triple("amdgcn-amd-amdhsa")));
  if (!state.MRI) return state;

  llvm::MCTargetOptions mc_opts;
  state.MAI.reset(state.target->createMCAsmInfo(
      *state.MRI, llvm::Triple("amdgcn-amd-amdhsa"), mc_opts));
  if (!state.MAI) return state;

  state.MCII.reset(state.target->createMCInstrInfo());
  if (!state.MCII) return state;

  state.STI.reset(state.target->createMCSubtargetInfo(
      llvm::Triple("amdgcn-amd-amdhsa"), state.cpu, ""));
  if (!state.STI || !state.STI->isCPUStringValid(state.cpu)) return state;

  state.Ctx = std::make_unique<llvm::MCContext>(triple, state.MAI.get(),
                                                state.MRI.get(),
                                                state.STI.get());
  state.MOFI = std::make_unique<llvm::MCObjectFileInfo>();
  state.MOFI->initMCObjectFileInfo(*state.Ctx, false);
  state.Ctx->setObjectFileInfo(state.MOFI.get());

  state.disasm.reset(
      state.target->createMCDisassembler(*state.STI, *state.Ctx));
  if (!state.disasm) return state;

  unsigned asm_variant = state.MAI->getAssemblerDialect();
  state.printer.reset(state.target->createMCInstPrinter(
      triple, asm_variant, *state.MAI, *state.MCII, *state.MRI));

  state.CE.reset(state.target->createMCCodeEmitter(*state.MCII, *state.Ctx));

  state.valid = true;
  return state;
}

LLVMState InitLLVMCached(const std::string &isa_name) {
  std::call_once(g_llvm_init_flag, InitLLVMTargets);

  const llvm::Target *tgt;
  {
    std::lock_guard<std::mutex> lock(g_target_cache_mutex);
    if (!g_cached_target) {
      std::string error;
      llvm::Triple triple("amdgcn-amd-amdhsa");
      g_cached_target =
          llvm::TargetRegistry::lookupTarget("amdgcn", triple, error);
    }
    tgt = g_cached_target;
  }

  return InitLLVMImpl(isa_name, tgt);
}

// ── Instruction decode ───────────────────────────────────────────────────────

[[nodiscard]] bool DecodeTextSection(const uint8_t *text, uint64_t text_size,
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

std::vector<uint8_t> AssembleSingleInst(const std::string &asm_str,
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

  llvm::MCCodeEmitter *ce =
      llvm_state.target->createMCCodeEmitter(*llvm_state.MCII, *llvm_state.Ctx);
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

[[nodiscard]] bool ApplyMnemonicSwap(const RewriteRule &rule,
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

Trampoline BuildTrampoline(const std::vector<std::string> &asm_lines,
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
  bool is_gfx12 = !(cpu.find("gfx9") == 0 || cpu.find("gfx10") == 0);
  if (!EncodeSBranch(branch_back_from, branch_back_to, branch_bytes,
                     is_gfx12)) {
    result.bytes.clear();
    return result;
  }

  result.bytes.insert(result.bytes.end(), branch_bytes, branch_bytes + 4);
  return result;
}

// ── VGPR introspection ───────────────────────────────────────────────────────

int GetVgprNum(unsigned reg, const llvm::MCRegisterInfo &MRI) {
  const char *name = MRI.getName(reg);
  if (!name) return -1;
  std::string rname(name);
  if (rname.find("VGPR") == 0) {
    size_t numstart = 4;
    size_t underscore = rname.find('_', numstart);
    std::string numstr = rname.substr(
        numstart, underscore == std::string::npos ? std::string::npos
                                                  : underscore - numstart);
    int val = -1;
    std::from_chars(numstr.data(), numstr.data() + numstr.size(), val);
    return val;
  }
  return -1;
}

std::pair<int, int> GetVgprRange(unsigned reg,
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
  auto [p, ec] = std::from_chars(numstr.data(), numstr.data() + numstr.size(), base);
  if (ec != std::errc())
    return {-1, 0};
  return {base, count};
}

std::pair<int, int>
GetOperandVgprRange(const llvm::MCInst &inst, unsigned op_idx,
                    const llvm::MCRegisterInfo &MRI) {
  if (op_idx >= inst.getNumOperands()) return {-1, 0};
  const auto &op = inst.getOperand(op_idx);
  if (!op.isReg()) return {-1, 0};
  return GetVgprRange(op.getReg(), MRI);
}

std::string PrintInst(const InternalDecodedInst &di,
                              const LLVMState &llvm_state) {
  std::string inst_str;
  if (llvm_state.printer) {
    llvm::raw_string_ostream rso(inst_str);
    llvm_state.printer->printInst(&di.inst, 0, "", *llvm_state.STI, rso);
    rso.flush();
  }
  return inst_str;
}

bool RangesOverlap(int base1, int count1, int base2, int count2) {
  if (base1 < 0 || base2 < 0) return false;
  return base1 < base2 + count2 && base2 < base1 + count1;
}

// ── WMMA co-execution hazard overlap check ──────────────────────────────────

bool CheckVgprOverlap(const llvm::MCInst &wmma_inst,
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
