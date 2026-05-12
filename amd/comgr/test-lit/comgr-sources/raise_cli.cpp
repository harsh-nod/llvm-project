// Per-file raiser CLI — two modes.
//
// Usage:
//   raise_cli <code-object.co|.hsaco> [--isa=<arch>] [--target-isa=<arch>]
//   raise_cli <code-object.co|.hsaco> --emit-ir[=<kernel>] [--isa=<arch>]
//                                     [--target-isa=<arch>]
//
// Default (kerneldex-coverage) mode. For each kernel in the code object,
// forks a child that runs raiseToIR so that a fatal error
// (report_fatal_error / asan trap / ...) in one kernel doesn't poison
// the whole file.  Emits one line per kernel on stdout:
//
//   OK   <kernel-name> (<lifted>/<total>)
//   FAIL <kernel-name> -> <mnemonic> [<format>]
//
// Kernels that crashed in the child (signal, non-zero exit, incomplete
// shm) are reported as a FAIL with mnemonic ``__crash__`` and format
// ``signal_<N>`` so they still land in the kerneldex worklist instead of
// being silently dropped.
//
// Exits 0 iff every kernel succeeded; otherwise 1.  ISA is auto-detected
// from the filename (look for ``gfx<digits>[a-z]?``) when ``--isa=`` is
// not passed.
//
// --emit-ir mode. Designed for lit tests. Runs raiseToIR in-process (no
// fork), dumps the raised LLVM IR for a single kernel on stdout, and
// leaves stderr alone so FileCheck can match warnings / abort-gate
// diagnostics. Selects the only kernel when the code object has one, or
// requires the ``=<kernel>`` form when there are multiple. Exits 0 iff
// the kernel raised successfully; non-zero otherwise.
//
// --target-isa=<arch>. Optional. Controls the target ISA the raiser
// lowers for; defaults to the source ISA (same-wave translation). Use
// to exercise cross-wave paths from a single CO (e.g. a gfx1250 CO
// compiled for a wave64 target).
//
// --enable-writelane-rewrite / --disable-writelane-rewrite. Default
// **on** (post-Triton-corpus graduation; see raiser.hpp for the full
// rationale).  Controls the post-raise rewrite of cross-widen-divergent
// `v_writelane_b32` / `v_readlane_b32` sites into per-source-wave
// `select` / `ds_bpermute` primitives — see
// `rewrite_cross_lane_divergent.{hpp,cpp}` and
// hotswap/docs/wave-size-translation.md §5.6.3.
//
// `--enable-writelane-rewrite` is accepted for backward compatibility
// (the canonical flag name used by existing lit fixtures) and is a
// no-op since the default is already on; `--disable-writelane-rewrite`
// forces the pre-rewrite path and is used by the `REFUSE` / `UNCHANGED`
// sibling RUN lines in the writelane/readlane regression fixtures to
// pin the pre-rewrite contract.  Later-wins between the two flags is
// by command-line order (last occurrence decides).
//
// --enable-wave-native / --disable-wave-native. Default **on** as
// of the WaveNative graduation. Selects `WaveNativeProjection`
// instead of `ModuloReplicationProjection` for wave32 source →
// wave64 target cross-widening. Under wave-native the kernel entry
// emits `@llvm.amdgcn.init_whole_wave` so hardware EXEC = -1 for
// the body, which:
//   * makes the WMMA → MFMA pipeline in `wmma-lowering.cpp`
//     correct on the upper half of the Wave64 target (the original
//     design motivation — see wave-size-translation.md §5.6.1);
//   * projects kernels with `num_warps > 1` correctly by giving
//     each target lane its own modeled-EXEC bit (fixes the
//     `swiglu_fp32` / `corpus_layernorm_fp32` class documented in
//     hotswap/docs/modrep-predicate-chain.md §4.3 sub-case 1);
//   * renders the C5 classifier's MODREP-specific refusal
//     rationale inapplicable — target lanes have their own
//     modeled-EXEC bits rather than sharing source wave 0's. The
//     classifier's `waveNative` gate suppresses refusal on this
//     path. For `canary_bpermute_scan_fp32`, the underlying
//     miscompile that would otherwise surface is closed by the
//     VOPD-cndmask SGPR-condition fix
//     (modrep-predicate-chain.md §6.4) rather than by the
//     projection choice itself.
//
// `--disable-wave-native` opts back into `ModuloReplicationProjection`
// for the narrow class of pointwise / independent-half kernels where
// MODREP's "replicas of source wave 0" model is correct AND where
// the C5 refusal under MODREP is the desired loud-fail signal.
// No env-var override exists; `HSA_HOTSWAP_WAVE_NATIVE` was a
// transient test hook during the graduation sweep and has been
// removed so the opt-out path isn't silently bypassed.
//
// (The earlier `--enable-permlane16-xor3-partner` /
// `--enable-permlane16-swap-selfpreserve` flags were removed along
// with their rewrite passes once the asymmetric
// `v_permlane16_swap_b32` lift landed — see
// `handle-valu-cross-lane.cpp::emitPermLaneSwapEmulation` and
// matrix-translation.md §12.4.7.)

#include "comgr-metadata.h"
#include "comgr.h"
#include "hotswap/code-object-utils.h"
#include "hotswap/pipeline.h"
#include "hotswap/raiser.h"

// raiser.hpp forward-declares llvm::LLVMContext and llvm::Module but
// RaiseResult holds them by unique_ptr, so the destructor synthesized in
// main() needs the complete types.
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

// Shared-memory block handed from each per-kernel child back to the parent.
// Using a fixed-size POD struct keeps the IPC trivially safe across fork().
struct KernelRaiseStats {
  bool done;
  bool success;
  int lifted;
  int total;
  char FailMnemonic[128];
  char FailFormat[64];
};

std::string autoDetectIsa(llvm::StringRef path) {
  // Look for ``gfx<digits>[a-z]?`` anywhere in the filename.
  for (size_t i = 0; i + 3 < path.size(); ++i) {
    if (path[i] == 'g' && path[i + 1] == 'f' && path[i + 2] == 'x') {
      size_t j = i + 3;
      while (j < path.size() &&
             std::isdigit(static_cast<unsigned char>(path[j])))
        ++j;
      if (j > i + 3) {
        if (j < path.size() && path[j] >= 'a' && path[j] <= 'z')
          ++j;
        return path.substr(i, j - i).str();
      }
    }
  }
  return {};
}

int usage() {
  llvm::errs()
      << "usage:\n"
         "  raise_cli <code-object.co|.hsaco> [--isa=<arch>] "
         "[--target-isa=<arch>] [--disable-writelane-rewrite] "
         "[--disable-wave-native]\n"
         "  raise_cli <code-object.co|.hsaco> --emit-ir[=<kernel>] "
         "[--isa=<arch>] [--target-isa=<arch>] "
         "[--disable-writelane-rewrite] [--disable-wave-native]\n"
         "  raise_cli <code-object.co|.hsaco> --write-hsaco=<path> "
         "[--kernel=<name>] [--isa=<arch>] [--target-isa=<arch>] "
         "[--disable-writelane-rewrite] [--disable-wave-native]\n"
         "\n"
         "Default mode: emits per-kernel OK/FAIL lines on stdout in the "
         "format\n"
         "  kerneldex coverage expects. Exits 0 iff every kernel raises.\n"
         "--emit-ir mode: dumps raised LLVM IR for a single kernel on "
         "stdout.\n"
         "  No fork; stderr left alone for FileCheck.\n"
         "--write-hsaco mode: runs the full pipeline (raise + llc + lld)\n"
         "  for a single kernel and writes the produced HSACO to <path>.\n"
         "  Intended for post-rewrite disassembly triage (see\n"
         "  hotswap/docs/wave-size-translation.md \u00a75.6.3).\n"
         "--target-isa: overrides the target ISA (default: same as --isa).\n"
         "--enable-writelane-rewrite / --disable-writelane-rewrite: controls\n"
         "  the cross-widen-divergent writelane/readlane rewrite (default "
         "on;\n"
         "  see wave-size-translation.md \u00a75.6.3). The `--enable-` form "
         "is\n"
         "  kept for backward compatibility (existing REWRITE lit RUN "
         "lines);\n"
         "  `--disable-` pins the pre-rewrite REFUSE / UNCHANGED path for "
         "the\n"
         "  sibling RUN lines. Later-wins on the command line.\n"
         "--enable-wave-native / --disable-wave-native: select between\n"
         "  WaveNativeProjection (post-graduation default) and\n"
         "  ModuloReplicationProjection for wave32 source \u2192 wave64\n"
         "  target cross-widening. The `--enable-` form is kept for\n"
         "  backward compatibility; `--disable-` pins the MODREP path\n"
         "  for lit fixtures and for kernels outside WaveNative's\n"
         "  class coverage (see wave-size-translation.md \u00a7\u00a72.2 / "
         "5.6.1\n"
         "  and modrep-predicate-chain.md \u00a76 for the graduation\n"
         "  rationale). Later-wins on the command line.\n"
         "ISA is inferred from the filename when --isa is not given.\n";
  return 2;
}

} // namespace

int main(int argc, char **argv) {
  std::string coPath;
  std::string isa;
  std::string targetIsa;
  bool emitIr = false;
  // Default on as of the Triton-corpus graduation (see this file's
  // top-of-file comment and raiser.hpp for the rationale).  The
  // `--disable-writelane-rewrite` flag (parsed below) forces the
  // pre-rewrite path for the lit fixtures that pin the
  // REFUSE / UNCHANGED sibling contracts.
  bool EnableWritelaneRewrite = true;
  bool EnableWaveNative = true;
  std::string emitIrKernel;
  std::string writeHsacoPath;
  std::string writeHsacoKernel;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a.rfind("--isa=", 0) == 0) {
      isa = a.substr(6);
    } else if (a == "--isa") {
      if (i + 1 >= argc)
        return usage();
      isa = argv[++i];
    } else if (a.rfind("--target-isa=", 0) == 0) {
      targetIsa = a.substr(13);
    } else if (a == "--target-isa") {
      if (i + 1 >= argc)
        return usage();
      targetIsa = argv[++i];
    } else if (a == "--emit-ir") {
      emitIr = true;
    } else if (a.rfind("--emit-ir=", 0) == 0) {
      emitIr = true;
      emitIrKernel = a.substr(10);
    } else if (a.rfind("--write-hsaco=", 0) == 0) {
      writeHsacoPath = a.substr(14);
    } else if (a.rfind("--kernel=", 0) == 0) {
      writeHsacoKernel = a.substr(9);
    } else if (a == "--enable-writelane-rewrite") {
      EnableWritelaneRewrite = true;
    } else if (a == "--disable-writelane-rewrite") {
      // Later-wins on the command line: the last occurrence of an
      // --enable- / --disable- pair decides the effective value.  This
      // matches the behaviour every lit fixture's REFUSE / REWRITE RUN
      // lines implicitly rely on (one flag per RUN line).
      EnableWritelaneRewrite = false;
    } else if (a == "--enable-wave-native") {
      EnableWaveNative = true;
    } else if (a == "--disable-wave-native") {
      // Later-wins on the command line, symmetric with
      // --enable-/--disable-writelane-rewrite. Post-graduation the
      // default is on; --disable-wave-native is the opt-out path for
      // lit fixtures that pin MODREP-specific IR shapes (the
      // `cross_wave_warn` warn-only contract, the narrow-O1 C5
      // refusal siblings) and for producer flows that want MODREP's
      // "independent halves" throughput on pointwise kernels. See
      // this file's top-of-file comment.
      EnableWaveNative = false;
    } else if (!a.empty() && a[0] == '-') {
      llvm::errs() << "raise_cli: unknown flag: " << a << "\n";
      return usage();
    } else if (coPath.empty()) {
      coPath = a;
    } else {
      llvm::errs() << "raise_cli: unexpected positional arg: " << a << "\n";
      return usage();
    }
  }
  if (coPath.empty())
    return usage();

  // Read the file up-front so we can fall back to the ELF e_flags
  // ISA when the filename heuristic fails (kerneldex corpora often
  // store kernels under hashed names with no `gfx*` substring; the
  // ELF MACH field is the only deterministic source).
  auto coBufOrErr = llvm::MemoryBuffer::getFile(coPath, /*IsText=*/false);
  if (!coBufOrErr) {
    llvm::errs() << "raise_cli: cannot read " << coPath << ": "
                 << coBufOrErr.getError().message() << "\n";
    return 2;
  }
  llvm::MemoryBufferRef coData = (*coBufOrErr)->getMemBufferRef();

  if (isa.empty()) {
    isa = autoDetectIsa(coPath);
    if (isa.empty()) {
      std::string elfIsa;
      if (COMGR::metadata::getElfIsaName(coData, elfIsa) ==
          AMD_COMGR_STATUS_SUCCESS)
        isa = std::move(elfIsa);
    }
    if (isa.empty()) {
      llvm::errs() << "raise_cli: could not infer ISA from " << coPath
                   << "; pass --isa=<arch>\n";
      return 2;
    }
  }

  auto kernelNamesOrErr = COMGR::hotswap::listKernelNames(coData);
  if (!kernelNamesOrErr) {
    llvm::errs() << "raise_cli: no kernels in " << coPath << ": "
                 << llvm::toString(kernelNamesOrErr.takeError()) << "\n";
    return 2;
  }
  llvm::SmallVector<std::string> kernelNames = std::move(*kernelNamesOrErr);
  if (kernelNames.empty()) {
    llvm::errs() << "raise_cli: no kernels in " << coPath << "\n";
    return 2;
  }

  auto textOrErr = COMGR::hotswap::extractTextSection(coData);
  if (!textOrErr) {
    llvm::errs() << "raise_cli: could not extract .text from " << coPath
                 << ": " << llvm::toString(textOrErr.takeError()) << "\n";
    return 2;
  }
  COMGR::hotswap::TextSection text = std::move(*textOrErr);

  // --emit-ir path — no fork, no stderr redirect. Used by lit tests that
  // FileCheck the raised IR on stdout and the raiser diagnostics on
  // stderr. One kernel per invocation.
  if (emitIr) {
    std::string target;
    if (emitIrKernel.empty()) {
      if (kernelNames.size() != 1) {
        llvm::errs() << "raise_cli: --emit-ir requires =<kernel> when the "
                        "code object has "
                     << kernelNames.size() << " kernels\n";
        return 2;
      }
      target = kernelNames.front();
    } else {
      bool found = false;
      for (const auto &kn : kernelNames)
        if (kn == emitIrKernel) {
          target = kn;
          found = true;
          break;
        }
      if (!found) {
        llvm::errs() << "raise_cli: kernel '" << emitIrKernel
                     << "' not found in " << coPath << "\n";
        return 2;
      }
    }
    auto metaOrErr = COMGR::hotswap::extractKernelMeta(coData, target);
    if (!metaOrErr) {
      llvm::errs() << "raise_cli: kernel '" << target << "' metadata: "
                   << llvm::toString(metaOrErr.takeError()) << "\n";
      return 1;
    }
    COMGR::hotswap::KernelMeta meta = std::move(*metaOrErr);
    auto kernelOffsetOrErr = COMGR::hotswap::findKernelSymbolOffset(coData, target);
    if (!kernelOffsetOrErr) {
      std::string err = llvm::toString(kernelOffsetOrErr.takeError());
      llvm::errs() << "raise_cli: kernel '" << target
                   << "' offset lookup failed: " << err << "\n";
      return 1;
    }
    uint64_t kernelOffset = *kernelOffsetOrErr;
    auto raised = COMGR::hotswap::raiseToIR(text.Bytes, isa, target, meta,
                                        kernelOffset, targetIsa,
                                        EnableWritelaneRewrite,
                                        EnableWaveNative);
    if (!raised.Success) {
      // Contract: raiseToIR only populates RaiseResult::IrText on the
      // success path (the last write before setting `success = true`),
      // so we cannot dump partial IR here. Callers that need stderr
      // diagnostics (abort-gate lit tests, etc.) FileCheck the raiser's
      // stderr — we leave that untouched.
      llvm::errs() << "raise_cli: kernel '" << target << "' failed to raise: "
                   << (raised.Failure.Mnemonic.empty()
                           ? "unknown"
                           : raised.Failure.Mnemonic.c_str())
                   << " ["
                   << (raised.Failure.Format.empty()
                           ? "unknown"
                           : raised.Failure.Format.c_str())
                   << "] @offset=0x";
      llvm::errs().write_hex(raised.Failure.Offset);
      if (!raised.Failure.Detail.empty())
        llvm::errs() << " :: " << raised.Failure.Detail;
      llvm::errs() << "\n";
      return 1;
    }
    llvm::outs().write(raised.IrText.data(), raised.IrText.size());
    return 0;
  }

  // --write-hsaco path — runs the full pipeline (raise + llc + lld)
  // for a single kernel and writes the resulting HSACO to disk.
  // Triage-mode only: lets downstream tools (llvm-objdump) inspect the
  // exact bytes the gtest harness would launch, so we can walk the
  // Phase 6.5 rewrite end-to-end through the final ISA.
  if (!writeHsacoPath.empty()) {
    std::string target;
    if (writeHsacoKernel.empty()) {
      if (kernelNames.size() != 1) {
        llvm::errs() << "raise_cli: --write-hsaco requires --kernel=<name> "
                        "when the code object has "
                     << kernelNames.size() << " kernels\n";
        return 2;
      }
      target = kernelNames.front();
    } else {
      bool found = false;
      for (const auto &kn : kernelNames)
        if (kn == writeHsacoKernel) {
          target = kn;
          found = true;
          break;
        }
      if (!found) {
        llvm::errs() << "raise_cli: kernel '" << writeHsacoKernel
                     << "' not found in " << coPath << "\n";
        return 2;
      }
    }
    std::string effectiveTargetIsa = targetIsa.empty() ? isa : targetIsa;
    COMGR::hotswap::PipelineOptions pipelineOptions;
    pipelineOptions.EnableWritelaneRewrite = EnableWritelaneRewrite;
    pipelineOptions.EnableWaveNative = EnableWaveNative;
    auto pipe = COMGR::hotswap::runPipeline(coData, isa, effectiveTargetIsa,
                                        target, pipelineOptions);
    if (!pipe.Success) {
      llvm::errs() << "raise_cli: pipeline failed for kernel '" << target
                   << "' (lifted=" << pipe.LiftedCount << "/"
                   << pipe.TotalCount << ", FailMnemonic='"
                   << pipe.FailMnemonic << "')\n";
      return 1;
    }
    FILE *fp = std::fopen(writeHsacoPath.c_str(), "wb");
    if (!fp) {
      llvm::errs() << "raise_cli: cannot open " << writeHsacoPath
                   << " for writing\n";
      return 2;
    }
    size_t hsacoSize = pipe.Hsaco ? pipe.Hsaco->getBufferSize() : 0;
    const char *hsacoData = pipe.Hsaco ? pipe.Hsaco->getBufferStart() : nullptr;
    size_t wrote = std::fwrite(hsacoData, 1, hsacoSize, fp);
    std::fclose(fp);
    if (wrote != hsacoSize) {
      llvm::errs() << "raise_cli: short write to " << writeHsacoPath << " ("
                   << wrote << " of " << hsacoSize << " bytes)\n";
      return 2;
    }
    llvm::errs() << "raise_cli: wrote " << hsacoSize << " byte HSACO for "
                 << "kernel '" << target << "' to " << writeHsacoPath
                 << " (lifted " << pipe.LiftedCount << "/" << pipe.TotalCount
                 << ")\n";
    return 0;
  }

  int totalKernels = 0, okKernels = 0, failKernels = 0, crashKernels = 0;

  for (auto &kName : kernelNames) {
    ++totalKernels;
    auto *shm = static_cast<KernelRaiseStats *>(
        mmap(nullptr, sizeof(KernelRaiseStats), PROT_READ | PROT_WRITE,
             MAP_SHARED | MAP_ANONYMOUS, -1, 0));
    if (shm == MAP_FAILED) {
      llvm::errs() << "raise_cli: mmap failed\n";
      return 3;
    }
    std::memset(shm, 0, sizeof(KernelRaiseStats));

    auto kernelOffsetOrErr = COMGR::hotswap::findKernelSymbolOffset(coData, kName);
    if (!kernelOffsetOrErr) {
      std::string err = llvm::toString(kernelOffsetOrErr.takeError());
      llvm::errs() << "raise_cli: kernel '" << kName
                   << "' offset lookup failed: " << err << "\n";
      ++failKernels;
      llvm::outs() << "FAIL " << kName
                   << " -> __kernel_offset__ "
                      "[KernelSymbolOffsetLookupFailed]\n";
      munmap(shm, sizeof(KernelRaiseStats));
      continue;
    }
    uint64_t kernelOffset = *kernelOffsetOrErr;

    // Flush stdout so the child doesn't inherit pending bytes that
    // would re-emit after fork().
    llvm::outs().flush();

    pid_t pid = fork();
    if (pid == 0) {
      // Silence the child's stderr: LLVM chatters a lot, and kerneldex
      // only cares about OK/FAIL on stdout plus the last stderr line
      // when the process as a whole crashes.
      int devnull = open("/dev/null", O_WRONLY);
      if (devnull >= 0) {
        dup2(devnull, STDERR_FILENO);
        close(devnull);
      }
      auto metaOrErr = COMGR::hotswap::extractKernelMeta(coData, kName);
      COMGR::hotswap::KernelMeta meta;
      if (metaOrErr) {
        meta = std::move(*metaOrErr);
      } else {
        llvm::consumeError(metaOrErr.takeError());
      }
      auto raised = COMGR::hotswap::raiseToIR(text.Bytes, isa, kName, meta,
                                          kernelOffset, targetIsa,
                                          EnableWritelaneRewrite,
                                          EnableWaveNative);
      shm->done = true;
      shm->success = raised.Success;
      shm->lifted = raised.LiftedCount;
      shm->total = raised.TotalCount;
      if (!raised.Success) {
        const char *mn = raised.Failure.Mnemonic.empty()
                             ? "unknown"
                             : raised.Failure.Mnemonic.c_str();
        const char *fmt = raised.Failure.Format.empty()
                              ? "unknown"
                              : raised.Failure.Format.c_str();
        std::strncpy(shm->FailMnemonic, mn, sizeof(shm->FailMnemonic) - 1);
        std::strncpy(shm->FailFormat, fmt, sizeof(shm->FailFormat) - 1);
      }
      _exit(0);
    }

    int st = 0;
    waitpid(pid, &st, 0);

    if (!shm->done || WIFSIGNALED(st) || (WIFEXITED(st) && WEXITSTATUS(st) != 0)) {
      // Child never wrote the shm marker, or died by signal, or exited
      // with a nonzero status: surface this as a FAIL row with a
      // synthetic mnemonic so kerneldex still counts the kernel.
      ++crashKernels;
      int sig = WIFSIGNALED(st) ? WTERMSIG(st) : 0;
      llvm::outs() << "FAIL " << kName << " -> __crash__ [signal_" << sig
                   << "]\n";
    } else if (shm->success) {
      ++okKernels;
      llvm::outs() << "OK " << kName << " (" << shm->lifted << "/"
                   << shm->total << ")\n";
    } else {
      ++failKernels;
      llvm::outs() << "FAIL " << kName << " -> " << shm->FailMnemonic
                   << " [" << shm->FailFormat << "]\n";
    }

    munmap(shm, sizeof(KernelRaiseStats));
  }

  llvm::errs() << "raise_cli: " << totalKernels << " kernels, " << okKernels
               << " ok, " << failKernels << " fail, " << crashKernels
               << " crash (" << coPath << ")\n";

  return (failKernels + crashKernels) == 0 ? 0 : 1;
}
