//===- raise-failure.cpp - Structured raise-failure values ----------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "raise-failure.h"

#include "decoded-inst.h"

namespace COMGR::hotswap {

const char *reasonString(RaiseFailureReason R) {
  switch (R) {
  case RaiseFailureReason::None:                    return "None";
  case RaiseFailureReason::BadInput:                return "BadInput";
  case RaiseFailureReason::UnsupportedOpcode:       return "UnsupportedOpcode";
  case RaiseFailureReason::UnsupportedShape:        return "UnsupportedShape";
  case RaiseFailureReason::SPEUnsafeExecWriter:
    return "SPE-unmodeled-EXEC-writer";
  case RaiseFailureReason::TargetMachineCreationFailed:
    return "TargetMachineCreationFailed";
  case RaiseFailureReason::IRVerificationFailed:
    return "IRVerificationFailed";
  case RaiseFailureReason::CrossWaveLaneIdLeak:
    return "cross-wave-lane-id-leak";
  case RaiseFailureReason::CrossWaveUnrewritableShuffle:
    return "cross-wave-unrewritable-shuffle";
  case RaiseFailureReason::CrossWaveShuffleRewritePending:
    return "cross-wave-shuffle-rewrite-pending";
  case RaiseFailureReason::CrossWaveReplicaRace:
    return "cross-wave-replica-race";
  case RaiseFailureReason::CrossWaveLanePredicatedExec:
    return "cross-wave-lane-predicated-exec";
  case RaiseFailureReason::CrossWavePredicateChain:
    return "cross-wave-predicate-chain";
  case RaiseFailureReason::StrictUnsafeLowering:
    return "strict-unsafe-lowering";
  case RaiseFailureReason::MissingKernelDescriptor:
    return "missing-kernel-descriptor";
  case RaiseFailureReason::UserSgprLayoutMismatch:
    return "user-sgpr-layout-mismatch";
  }
  llvm_unreachable("unhandled RaiseFailureReason");
}

RaiseFailure RaiseFailure::unsupportedShape(const DecodedInst &Di,
                                             llvm::StringRef Format,
                                             const llvm::Twine &Detail) {
  RaiseFailure F;
  F.Reason = RaiseFailureReason::UnsupportedShape;
  F.Mnemonic = Di.Mnemonic;
  F.Format = Format.str();
  F.Offset = Di.Offset;
  F.Detail = Detail.str();
  return F;
}

RaiseFailure RaiseFailure::unsupportedOpcode(const DecodedInst &Di,
                                              llvm::StringRef Format) {
  RaiseFailure F;
  F.Reason = RaiseFailureReason::UnsupportedOpcode;
  F.Mnemonic = Di.Mnemonic;
  F.Format = Format.str();
  F.Offset = Di.Offset;
  return F;
}

RaiseFailure RaiseFailure::speUnsafeExecWriter(const DecodedInst &Di) {
  RaiseFailure F;
  F.Reason = RaiseFailureReason::SPEUnsafeExecWriter;
  F.Mnemonic = Di.Mnemonic;
  F.Format = "SPE-unmodeled-EXEC-writer";
  F.Offset = Di.Offset;
  return F;
}

RaiseFailure RaiseFailure::targetMachineCreationFailed() {
  RaiseFailure F;
  F.Reason = RaiseFailureReason::TargetMachineCreationFailed;
  F.Format = reasonString(RaiseFailureReason::TargetMachineCreationFailed);
  F.Detail = "createTargetMachine returned null";
  return F;
}

RaiseFailure RaiseFailure::irVerificationFailed(const llvm::Twine &Err) {
  RaiseFailure F;
  F.Reason = RaiseFailureReason::IRVerificationFailed;
  F.Format = reasonString(RaiseFailureReason::IRVerificationFailed);
  F.Detail = Err.str();
  return F;
}

// ----------------------------------------------------------------------------
// Phase 1.4.5 wave-size-obstruction factories. All share the same
// structure: take the refused instruction for mnemonic / offset, and
// a kind-specific detail string for the `detail` field.
// ----------------------------------------------------------------------------

namespace {

RaiseFailure makeCrossWaveFailure(RaiseFailureReason Reason,
                                   const DecodedInst &Di,
                                   const llvm::Twine &KindDetail) {
  RaiseFailure F;
  F.Reason = Reason;
  F.Mnemonic = Di.Mnemonic;
  F.Format = reasonString(Reason);
  F.Offset = Di.Offset;
  F.Detail = KindDetail.str();
  return F;
}

} // namespace

RaiseFailure RaiseFailure::crossWaveLaneIdLeak(const DecodedInst &Di,
                                                const llvm::Twine &KindDetail) {
  return makeCrossWaveFailure(RaiseFailureReason::CrossWaveLaneIdLeak, Di,
                               KindDetail);
}

RaiseFailure RaiseFailure::crossWaveUnrewritableShuffle(
    const DecodedInst &Di, const llvm::Twine &KindDetail) {
  return makeCrossWaveFailure(
      RaiseFailureReason::CrossWaveUnrewritableShuffle, Di, KindDetail);
}

RaiseFailure RaiseFailure::crossWaveShuffleRewritePending(
    const DecodedInst &Di, const llvm::Twine &KindDetail) {
  return makeCrossWaveFailure(
      RaiseFailureReason::CrossWaveShuffleRewritePending, Di, KindDetail);
}

RaiseFailure RaiseFailure::crossWaveReplicaRace(const DecodedInst &Di,
                                                 const llvm::Twine &KindDetail) {
  return makeCrossWaveFailure(RaiseFailureReason::CrossWaveReplicaRace, Di,
                               KindDetail);
}

RaiseFailure RaiseFailure::crossWaveLanePredicatedExec(
    const DecodedInst &Di, const llvm::Twine &KindDetail) {
  return makeCrossWaveFailure(
      RaiseFailureReason::CrossWaveLanePredicatedExec, Di, KindDetail);
}

// see hotswap/docs/modrep-predicate-chain.md §5 (narrow-O1)
RaiseFailure RaiseFailure::crossWavePredicateChain(
    llvm::StringRef KernelName, const llvm::Twine &Detail) {
  RaiseFailure F;
  F.Reason = RaiseFailureReason::CrossWavePredicateChain;
  F.Mnemonic = "workitem.id.x-predicate-chain-classifier";
  F.Format = reasonString(RaiseFailureReason::CrossWavePredicateChain);
  F.Offset = 0;
  F.Detail = ("kernel '" + KernelName + "': " + Detail).str();
  return F;
}

RaiseFailure RaiseFailure::strictUnsafeLowering(const DecodedInst &Di,
                                                  llvm::StringRef Site,
                                                  const llvm::Twine &Detail) {
  RaiseFailure F;
  F.Reason = RaiseFailureReason::StrictUnsafeLowering;
  F.Mnemonic = Di.Mnemonic;
  F.Format = Site.str();
  F.Offset = Di.Offset;
  F.Detail = Detail.str();
  return F;
}

RaiseFailure RaiseFailure::missingKernelDescriptor(llvm::StringRef KernelName) {
  RaiseFailure F;
  F.Reason = RaiseFailureReason::MissingKernelDescriptor;
  F.Mnemonic = "<kernel-descriptor>";
  F.Format = reasonString(RaiseFailureReason::MissingKernelDescriptor);
  F.Offset = 0;
  F.Detail = ("kernel '" + KernelName + "': .kd symbol not parsed").str();
  return F;
}

RaiseFailure RaiseFailure::userSgprLayoutMismatch(
    llvm::StringRef KernelName, const llvm::Twine &Detail) {
  RaiseFailure F;
  F.Reason = RaiseFailureReason::UserSgprLayoutMismatch;
  F.Mnemonic = "<user-sgpr-layout>";
  F.Format = reasonString(RaiseFailureReason::UserSgprLayoutMismatch);
  F.Offset = 0;
  F.Detail = ("kernel '" + KernelName + "': " + Detail).str();
  return F;
}

RaiseFailure RaiseFailure::crossWaveRewriteOracleDisagreement(
    llvm::StringRef KernelName, const llvm::Twine &Detail) {
  RaiseFailure F;
  F.Reason = RaiseFailureReason::CrossWaveLaneIdLeak;
  F.Mnemonic = "writelane/readlane-post-raise-safety-net";
  F.Format = reasonString(RaiseFailureReason::CrossWaveLaneIdLeak);
  F.Offset = 0;
  F.Detail = ("kernel '" + KernelName + "': " + Detail).str();
  return F;
}

} // namespace COMGR::hotswap
