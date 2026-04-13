//===- comgr-hotswap-internal.h - HotSwap internal types and declarations -===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Internal header for the HotSwap ISA rewriting subsystem. Shared by all
/// comgr-hotswap-*.cpp compilation units. Not part of the public COMGR API.
///
/// Module structure:
///   comgr-hotswap-elf.cpp       — ELF parsing, binary helpers, trampoline growth
///   comgr-hotswap-dwarf.cpp     — DWARF debug section patching
///   comgr-hotswap-llvm.cpp      — LLVM MC infrastructure (disasm/asm/encode)
///   comgr-hotswap-liveness.cpp  — CFG, backward liveness, scratch allocator
///   comgr-hotswap-b0a0.cpp      — GFX1250 B0-to-A0 silicon stepping patches
///   comgr-hotswap.cpp           — Public C API entry points
///
//===----------------------------------------------------------------------===//

#ifndef COMGR_HOTSWAP_INTERNAL_H
#define COMGR_HOTSWAP_INTERNAL_H

#include "amd_comgr.h"

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "llvm/Config/llvm-config.h"
#include "llvm/ADT/BitVector.h"
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
#include "llvm/Support/LEB128.h"
#include "llvm/Object/ELF.h"
#include "llvm/Object/ELFTypes.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/MC/TargetRegistry.h"

// ═══════════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════════

// ── MallocBuffer RAII wrapper ────────────────────────────────────────────────

struct MallocBuffer {
  uint8_t *data = nullptr;
  size_t size = 0;

  MallocBuffer() = default;
  MallocBuffer(size_t n)
      : data(static_cast<uint8_t *>(std::malloc(n))), size(data ? n : 0) {}
  ~MallocBuffer() { std::free(data); }

  MallocBuffer(MallocBuffer &&o) noexcept : data(o.data), size(o.size) {
    o.data = nullptr;
    o.size = 0;
  }
  MallocBuffer &operator=(MallocBuffer &&o) noexcept {
    if (this != &o) {
      std::free(data);
      data = o.data;
      size = o.size;
      o.data = nullptr;
      o.size = 0;
    }
    return *this;
  }

  MallocBuffer(const MallocBuffer &) = delete;
  MallocBuffer &operator=(const MallocBuffer &) = delete;

  explicit operator bool() const { return data != nullptr; }
  uint8_t *release() {
    uint8_t *p = data;
    data = nullptr;
    size = 0;
    return p;
  }
};

// ── Logging ──────────────────────────────────────────────────────────────────

enum class HotswapLogLevel : int { Silent = 0, Error = 1, Info = 2, Debug = 3 };

inline HotswapLogLevel GetHotswapLogLevel() {
  static HotswapLogLevel level = []() {
    const char *env = std::getenv("HSA_HOTSWAP_LOG_LEVEL");
    if (env) {
      int v = std::atoi(env);
      if (v >= 0 && v <= 3)
        return static_cast<HotswapLogLevel>(v);
    }
    return HotswapLogLevel::Info;
  }();
  return level;
}

inline std::ostream &HotswapLog(HotswapLogLevel level) {
  class NullBuf : public std::streambuf {
  protected:
    int overflow(int c) override { return c; }
  };
  static NullBuf null_buf;
  static std::ostream null_stream(&null_buf);
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

// ── Rewrite-rule types (used by ApplyByteReplace / ApplyMnemonicSwap) ────────

struct RewriteRule {
  std::string replace_mnemonic;
  bool preserve_operands = true;
  std::vector<uint8_t> replace_bytes;
};

// ── s_branch / s_nop constants ───────────────────────────────────────────────

inline constexpr uint32_t S_BRANCH_GFX9 = 0xBF820000u;
inline constexpr uint32_t S_BRANCH_GFX12 = 0xBFA00000u;
inline constexpr uint32_t S_NOP_OPCODE = 0xBF800000u;

// ── AMDGPU Kernel Descriptor RSRC1 bit fields ───────────────────────────────

static constexpr uint32_t KD_RSRC1_VGPR_MASK = 0x3Fu;
static constexpr uint32_t KD_RSRC1_SGPR_SHIFT = 6;
static constexpr uint32_t KD_RSRC1_SGPR_MASK = 0xFu;

// ── DWARF types ──────────────────────────────────────────────────────────────

struct DebugLineRow {
  uint64_t address;
  uint32_t file;
  int32_t line;
};

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
  std::unique_ptr<llvm::MCCodeEmitter> CE;
  std::string cpu;
  bool valid = false;
};

// ── Decoded instruction with MCInst ──────────────────────────────────────────

struct InternalDecodedInst {
  uint64_t offset;
  uint32_t size;
  llvm::MCInst inst;
  std::string mnemonic;
};

// ── Per-point VGPR liveness types ────────────────────────────────────────────

struct RegDefUse {
  llvm::BitVector defs{256};
  llvm::BitVector uses{256};
};

struct BasicBlock {
  uint64_t start_offset = 0;
  uint64_t end_offset = 0;
  std::vector<size_t> inst_indices;
  std::vector<int> successors;
  std::vector<int> predecessors;
};

struct CFG {
  std::vector<BasicBlock> blocks;
  std::unordered_map<uint64_t, int> offset_to_block;
};

struct LivenessInfo {
  std::vector<llvm::BitVector> live_before;
  std::vector<llvm::BitVector> live_after;
  bool converged = false;
};

struct ScratchAllocator {
  llvm::BitVector live_at_point;
  int kd_allocated_vgprs;
  int next_above_kd;
  int extra_allocated = 0;

  ScratchAllocator(const llvm::BitVector &live, int kd_vgprs)
      : live_at_point(live), kd_allocated_vgprs(kd_vgprs),
        next_above_kd(kd_vgprs) {}

  int Alloc() {
    for (int v = kd_allocated_vgprs - 1; v >= 0; --v) {
      if (!live_at_point.test(v)) {
        live_at_point.set(v);
        return v;
      }
    }
    if (next_above_kd >= 256)
      return -1;
    int v = next_above_kd++;
    extra_allocated++;
    live_at_point.set(v);
    return v;
  }

  int ExtraVgprsNeeded() const { return extra_allocated; }
};

struct ScratchPatchInfo {
  uint64_t offset;
  llvm::BitVector scratch_regs{256};
};

// ── B0-to-A0 types ──────────────────────────────────────────────────────────

struct WmmaNopReq {
  int b0_nops;
  int a0_nops;
};

struct WmmaHazard {
  size_t wmma_idx;
  size_t valu_idx;
  int existing_nops;
  int needed_nops;
  int deficit;
};

struct KernelPatchStats {
  int extra_vgprs = 0;
  int scratch_reused = 0;
  int scratch_above_kd = 0;
};

struct PatchContext {
  std::vector<InternalDecodedInst> &decoded;
  uint8_t *text;
  uint64_t text_size;
  const LLVMState &llvm_state;
  std::vector<Trampoline> &out_trampolines;
  std::vector<NopSled> &nop_sleds;
  uint8_t *elf_data;
  size_t elf_size;
  const ElfInfo &elf_info;
  const LivenessInfo &liveness;
  std::unordered_map<std::string, KernelPatchStats> &kernel_stats;
  std::vector<ScratchPatchInfo> &out_scratch_patches;
};

// ═══════════════════════════════════════════════════════════════════════════════
// Function declarations (cross-file)
// ═══════════════════════════════════════════════════════════════════════════════

// ── elf ──────────────────────────────────────────────────────────────────────

[[nodiscard]] bool EncodeSBranch(uint64_t from_offset, uint64_t to_offset,
                                 uint8_t out_bytes[4], bool gfx12 = false);
void EncodeSNop(uint8_t out_bytes[4]);
std::string ExtractCPU(const std::string &isa_name);
[[nodiscard]] bool ParseElfInfo(const uint8_t *elf, size_t elf_size,
                                ElfInfo &info);
std::string FindKernelAtOffset(const ElfInfo &elf_info, uint64_t text_offset);
[[nodiscard]] bool ApplyByteReplace(const RewriteRule &rule,
                                    uint64_t inst_offset, uint32_t inst_size,
                                    uint8_t *text, uint64_t text_size);
void UpdateKernelDescriptor(uint8_t *elf_data, size_t elf_size,
                            const ElfInfo &elf_info,
                            const std::string &kernel_name,
                            int32_t extra_vgprs, int32_t extra_sgprs);
NopSled *FindNearestSled(std::vector<NopSled> &sleds, uint64_t offset,
                         uint64_t needed);
MallocBuffer GrowElfWithTrampolines(const uint8_t *elf, size_t elf_size,
                                    const ElfInfo &elf_info,
                                    const std::vector<Trampoline> &trampolines);
bool PatchElfIsa(uint8_t *elf, size_t elf_size, const std::string &target_cpu);
int GetKernelVgprCount(const uint8_t *elf_data, size_t elf_size,
                       const ElfInfo &elf_info,
                       const std::string &kernel_name);

// ── dwarf ────────────────────────────────────────────────────────────────────

uint8_t *FindSectionHeader(uint8_t *elf, size_t elf_size, const char *name,
                           int *out_idx = nullptr);
[[nodiscard]] bool AddTrampolineSymbols(
    MallocBuffer &elf_buf, const std::vector<Trampoline> &trampolines,
    uint64_t text_size_before, int text_section_idx);
[[nodiscard]] bool PatchDebugLine(MallocBuffer &elf_buf,
                                  const std::vector<Trampoline> &trampolines,
                                  uint64_t text_size_before,
                                  uint64_t text_addr);
void PatchDebugRanges(uint8_t *elf, size_t elf_size, uint64_t text_addr,
                      uint64_t text_size_before, uint64_t tramp_total);
void PatchDebugInfo(uint8_t *elf, size_t elf_size, uint64_t text_addr,
                    uint64_t text_size_before, uint64_t tramp_total);
void PatchDebugFrame(uint8_t *elf, size_t elf_size, uint64_t text_addr,
                     uint64_t text_size_before, uint64_t tramp_total);

// ── llvm ─────────────────────────────────────────────────────────────────────

LLVMState InitLLVMImpl(const std::string &isa_name,
                       const llvm::Target *cached_target = nullptr);
LLVMState InitLLVMCached(const std::string &isa_name);
[[nodiscard]] bool DecodeTextSection(const uint8_t *text, uint64_t text_size,
                                     const LLVMState &llvm_state,
                                     std::vector<InternalDecodedInst> &decoded);
std::vector<uint8_t> AssembleSingleInst(const std::string &asm_str,
                                        const LLVMState &llvm_state);
[[nodiscard]] bool ApplyMnemonicSwap(const RewriteRule &rule,
                                     InternalDecodedInst &inst, uint8_t *text,
                                     const LLVMState &llvm_state);
Trampoline BuildTrampoline(const std::vector<std::string> &asm_lines,
                           uint64_t original_offset, uint32_t original_size,
                           uint64_t trampoline_text_offset,
                           const std::string &cpu,
                           const LLVMState &llvm_state);
std::string PrintInst(const InternalDecodedInst &di,
                      const LLVMState &llvm_state);
int GetVgprNum(unsigned reg, const llvm::MCRegisterInfo &MRI);
std::pair<int, int> GetVgprRange(unsigned reg,
                                 const llvm::MCRegisterInfo &MRI);
std::pair<int, int> GetOperandVgprRange(const llvm::MCInst &inst,
                                        unsigned op_idx,
                                        const llvm::MCRegisterInfo &MRI);
bool RangesOverlap(int base1, int count1, int base2, int count2);
bool CheckVgprOverlap(const llvm::MCInst &wmma_inst,
                      const llvm::MCInst &valu_inst,
                      const llvm::MCRegisterInfo &MRI);

// ── liveness ─────────────────────────────────────────────────────────────────

RegDefUse GetInstRegDefUse(const llvm::MCInst &inst,
                           const llvm::MCInstrInfo &MCII,
                           const llvm::MCRegisterInfo &MRI);
int64_t GetBranchImm(const llvm::MCInst &inst);
CFG BuildCFG(const std::vector<InternalDecodedInst> &decoded,
             const llvm::MCInstrInfo &MCII);
LivenessInfo ComputeLiveness(const std::vector<InternalDecodedInst> &decoded,
                             const CFG &cfg, const llvm::MCInstrInfo &MCII,
                             const llvm::MCRegisterInfo &MRI);
[[nodiscard]] bool VerifyPatchCorrectness(
    const uint8_t *text, uint64_t text_size, const LLVMState &llvm_state,
    const std::vector<ScratchPatchInfo> &scratch_patches);

// ── b0a0 — dispatcher and entry point ────────────────────────────────────────

amd_comgr_status_t RetargetCodeObjectB0A0(const void *elf_data,
                                          size_t elf_size, void **out_data,
                                          size_t *out_size);

// ── b0a0 — patch entry points (weak stubs in b0a0.cpp) ──────────────────────
//
// Each group is defined as a weak symbol in comgr-hotswap-b0a0.cpp returning 0.
// Patch .cpp files provide strong definitions that override the stubs at link
// time, allowing patches to land as independent PRs.

uint32_t ApplyInPlacePatches(PatchContext &ctx, size_t idx);
uint32_t ApplyTrampolinePatches(PatchContext &ctx, size_t idx);
uint32_t ApplyWmmaHazardPatch(PatchContext &ctx);
uint32_t ApplyWmmaSplitPatches(PatchContext &ctx, size_t idx);
uint32_t ApplyScratchPatches(PatchContext &ctx, size_t idx);

#endif // COMGR_HOTSWAP_INTERNAL_H
