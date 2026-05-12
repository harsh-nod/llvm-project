//===- c5_predicate_chain_test.cpp - c5_predicate_chain unit tests --------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Unit tests for the narrow-O1 Class-5 predicate-chain classifier.
// See hotswap/docs/modrep-predicate-chain.md §5 (O1) and
// transpiler/c5_predicate_chain_classifier.{hpp,cpp}.
//
// The tests synthesise small LLVM Function IR modules directly (no
// code-object round-trip) so the classifier is audited in isolation
// from the raiser's MC-level pipeline. This matches the approach
// AGENTS.md (Missing targeted tests) recommends for
// `WaveProjection` / `ModuloReplicationProjection` primitives:
// build a tiny IR module, run the primitive, assert the verdict.
//
// Coverage matrix (each test listed corresponds to one row):
//
//   * TidDirectSmallConstRefuses — baseline refusal path: direct
//     `tid → icmp ult tid, 15 → br → side effect`. Constant K=15
//     is within (0, W_s-1=31]. Classifier must refuse.
//   * TidMaskedBeforeCmpAccepts — non-refusal path: `tid → and tid,
//     31 → icmp ult masked, 15 → br → side effect`. The mask
//     collapses replica-1 onto `[0, W_s)` before the predicate,
//     so the classifier must not refuse.
//   * TidSmallConstZeroAccepts — the `tid != 0` / `tid == 0` null-
//     check idiom. K=0 is NOT in (0, W_s-1], so classifier must
//     not refuse (zero is a mask-extraction / null-check shape,
//     not a lane-position gate).
//   * TidLargeConstAccepts — `icmp ult tid, 64` with K=64 > W_s-1.
//     Classifier must treat this as a bounds check and not refuse.
//     Pins the baseline-non-refusal contract for kernels whose
//     bounds check compares against a constant ≥ W_t.
//   * TidDynamicCmpAccepts — `icmp ult tid, %kernarg_N`. The
//     non-tid operand is a runtime SGPR value, not a
//     compile-time constant. Pins the baseline-non-refusal
//     contract for the vecadd_f16 / rope_fp32 shape (bounds
//     check vs. a kernarg).
//   * SameWaveDirectionGate — same-wave (source = target) MUST
//     return `!refused && visitedCalls == 0`. Pins the
//     direction gate.
//   * NarrowingDirectionGate — wave64 → wave32 (narrowing)
//     MUST return `!refused && visitedCalls == 0`. Pins the
//     direction gate.
//   * WaveNativeProjectionGate — same refusal-shaped kernel as
//     TidDirectSmallConstRefuses, but invoked with
//     `PredicateChainProjection::WaveNative`. MUST return `!refused`
//     while STILL populating `observedSites` (the walk runs so
//     raiser.cpp can emit `LLVM_DEBUG` attribution breadcrumbs — the
//     classifier's projection gate SUPPRESSES refusal, it does not
//     skip the walk). Pins the structural projection gate:
//     under WaveNativeProjection each target lane is its own
//     source lane, so the MODREP replica-1 rationale does not
//     apply, but we still need the site list for debug
//     attribution. Protects the `enableWaveNative` thread-
//     through in raiser.cpp Phase 6.6 (and the LLVM_DEBUG
//     emission under WaveNative) from a future refactor that
//     accidentally drops the parameter or the walk.
//   * CrossSubtreeMaskedVsUnmaskedAccepts — `icmp f(tid)_unmasked,
//     g(tid)_masked` where both operands are tid-derived via
//     different subtrees: unmasked `tid+1` vs masked `tid & 15`.
//     Pins the FALSIFIED cross-subtree-refusal theory: the
//     classifier used to refuse this shape in an earlier
//     iteration, but compare_correctness (2026-04-21) showed
//     Triton's bounds-check idiom
//     (`icmp sgt tid-unmasked, tid-masked`) is in the baseline-
//     MATCH set under MODREP for `ult`/`ugt`/`slt`/`sgt`. Only
//     `eq`/`ne` variants would actually diverge (a different
//     per-predicate rule the classifier does not implement
//     today). The test pins the non-refusal so a future
//     iteration re-introducing the cross-subtree rule without
//     per-predicate reasoning fails here.
//   * IntrinsicPropagatorRefuses — `icmp @llvm.umin(tid, 100), 15`:
//     tid flows through a `umin` numeric intrinsic (not a
//     BinaryOperator). Pins the #8 intrinsic-propagator audit —
//     without explicit enumeration of numeric intrinsics as
//     propagators, the walk would stop at the `umin` call and
//     miss the downstream C5 site.
//   * NoCallsIsNoOp — function with no `workitem.id.x` intrinsic
//     call returns `!refused && visitedCalls == 0`.
//   * PhiPropagatesTidDerivation — tid flows through a phi whose
//     OTHER incoming is undef / constant; the masked arm MUST
//     still refuse through the phi chain. Pins that phi is a
//     pure propagator in the walk and doesn't break tid-
//     derivation tracking.
//   * MaskedPhiThroughUnmaskedArmRefuses — tid flows through a
//     phi with one UNMASKED arm (`%tid` directly) and one
//     MASKED arm (`%vand = and %tid, 31`). The classifier must
//     still refuse because the unmasked arm exists on the chain
//     — the `isSourceWaveMaskAnd` gate only stops propagation
//     through the AND user itself, not through phi unions with
//     the AND's consumer. This matches the shape observed in
//     `swiglu_fp32`'s IR (phi-arm asymmetry); the fixture
//     is not refused end-to-end because its icmp constant is
//     dynamic, which is caught by the `TidDynamicCmpAccepts`
//     test above.
//
// The tests use LLVM's IRBuilder + in-memory Module rather than
// `parseAssemblyString` to avoid a MemoryBuffer / LLVM-text-IR
// dependency. Each test constructs a tiny kernel-like function,
// calls `classifyPredicateChain`, and asserts the boolean verdict
// + the `visitedCalls` counter.

#include "hotswap/c5-predicate-chain-classifier.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

using namespace llvm;
using COMGR::hotswap::classifyPredicateChain;
using COMGR::hotswap::PredicateChainClassifierReport;
using COMGR::hotswap::PredicateChainProjection;

namespace {

// Canonical source / target wave sizes for all positive-direction
// tests: wave32 source → wave64 target, matching the gfx1250 →
// gfx942/gfx950 cross-widening this class exists to catch.
constexpr unsigned kSrcWs = 32;
constexpr unsigned kTgtWs = 64;

// Build a minimal module + function skeleton. Caller fills in the
// entry block via the returned IRBuilder. The function signature
// is `void kernel(i32* out, i32 bound)` — two args cover the
// common shapes the tests exercise (a store pointer and a dynamic
// bound).
struct Harness {
  LLVMContext Ctx;
  std::unique_ptr<Module> M;
  Function *F = nullptr;
  IRBuilder<> B;
  BasicBlock *Entry = nullptr;

  Harness() : B(Ctx) {
    M = std::make_unique<Module>("c5_predicate_chain_test", Ctx);
    auto *I32Ty = Type::getInt32Ty(Ctx);
    auto *PtrI32 = PointerType::get(Ctx, /*AS=*/1);
    auto *FnTy = FunctionType::get(Type::getVoidTy(Ctx),
                                   {PtrI32, I32Ty},
                                   /*isVarArg=*/false);
    F = Function::Create(FnTy, Function::ExternalLinkage, "kernel",
                          M.get());
    Entry = BasicBlock::Create(Ctx, "entry", F);
    B.SetInsertPoint(Entry);
  }

  // Emit a `call i32 @llvm.amdgcn.workitem.id.x()` into the current
  // insert block.
  CallInst *emitTid() {
    Function *Intrin = Intrinsic::getOrInsertDeclaration(
        M.get(), Intrinsic::amdgcn_workitem_id_x);
    return B.CreateCall(Intrin, {}, "tid");
  }

  // Emit a store that gates off a branch `br i1 %cond, spe_do, spe_skip`,
  // returning a handle on the original insert block's terminator-free
  // continuation so the caller can inspect the resulting function.
  // Wiring: we create `spe_do` and `spe_skip`, put a `store` in
  // `spe_do`, branch from the current block on `%cond`, and leave
  // the insert point at `spe_skip`.
  void emitStoreGate(Value *Cond) {
    BasicBlock *SpeDo = BasicBlock::Create(Ctx, "spe_do", F);
    BasicBlock *SpeSkip = BasicBlock::Create(Ctx, "spe_skip", F);
    B.CreateCondBr(Cond, SpeDo, SpeSkip);
    B.SetInsertPoint(SpeDo);
    auto *I32Ty = Type::getInt32Ty(Ctx);
    auto *PtrI32 = PointerType::get(Ctx, /*AS=*/1);
    Value *OutPtr = F->getArg(0);
    (void)PtrI32;
    B.CreateStore(ConstantInt::get(I32Ty, 42), OutPtr);
    B.CreateBr(SpeSkip);
    B.SetInsertPoint(SpeSkip);
  }

  // Close the function with a `ret void`.
  void finish() {
    if (B.GetInsertBlock()->empty() ||
        !B.GetInsertBlock()->getTerminator())
      B.CreateRetVoid();
  }
};

} // namespace

// ---------------------------------------------------------------------
// Refusal path: direct `tid → icmp ult tid, 15 → br → store`.
// ---------------------------------------------------------------------
TEST(C5PredicateChain, TidDirectSmallConstRefuses) {
  Harness H;
  Value *Tid = H.emitTid();
  Value *Cmp = H.B.CreateICmpULT(
      Tid, ConstantInt::get(Type::getInt32Ty(H.Ctx), 15), "c5_cmp");
  H.emitStoreGate(Cmp);
  H.finish();

  auto Report = classifyPredicateChain(*H.F, kSrcWs, kTgtWs);
  EXPECT_TRUE(Report.Refused);
  EXPECT_EQ(Report.VisitedCalls, 1u);
  EXPECT_NE(Report.RefusalDetail.find("compile-time constant 15"),
            std::string::npos)
      << "refusalDetail='" << Report.RefusalDetail << "'";
}

// ---------------------------------------------------------------------
// Non-refusal: `tid → and tid, 31 → icmp ult masked, 15 → br → store`.
// The mask collapses replica-1 onto [0, W_s); classifier stops walking.
// ---------------------------------------------------------------------
TEST(C5PredicateChain, TidMaskedBeforeCmpAccepts) {
  Harness H;
  Value *Tid = H.emitTid();
  auto *I32Ty = Type::getInt32Ty(H.Ctx);
  Value *Masked = H.B.CreateAnd(Tid, ConstantInt::get(I32Ty, 31),
                                 "tid_masked");
  Value *Cmp = H.B.CreateICmpULT(
      Masked, ConstantInt::get(I32Ty, 15), "c5_cmp_masked");
  H.emitStoreGate(Cmp);
  H.finish();

  auto Report = classifyPredicateChain(*H.F, kSrcWs, kTgtWs);
  EXPECT_FALSE(Report.Refused);
  EXPECT_EQ(Report.VisitedCalls, 1u);
  EXPECT_TRUE(Report.RefusalDetail.empty());
}

// ---------------------------------------------------------------------
// Non-refusal: `icmp eq tid, 0` (null-check / mask-extract idiom).
// K=0 is structurally a different shape from a lane-position gate
// and the classifier intentionally does not refuse it.
// ---------------------------------------------------------------------
TEST(C5PredicateChain, TidSmallConstZeroAccepts) {
  Harness H;
  Value *Tid = H.emitTid();
  auto *I32Ty = Type::getInt32Ty(H.Ctx);
  Value *Cmp = H.B.CreateICmpEQ(
      Tid, ConstantInt::get(I32Ty, 0), "c5_cmp_zero");
  H.emitStoreGate(Cmp);
  H.finish();

  auto Report = classifyPredicateChain(*H.F, kSrcWs, kTgtWs);
  EXPECT_FALSE(Report.Refused);
  EXPECT_EQ(Report.VisitedCalls, 1u);
}

// ---------------------------------------------------------------------
// Non-refusal: `icmp ult tid, 64` (K=64 > W_s-1=31). The classifier
// treats this as a bounds check rather than a lane-position gate.
// ---------------------------------------------------------------------
TEST(C5PredicateChain, TidLargeConstAccepts) {
  Harness H;
  Value *Tid = H.emitTid();
  auto *I32Ty = Type::getInt32Ty(H.Ctx);
  Value *Cmp = H.B.CreateICmpULT(
      Tid, ConstantInt::get(I32Ty, 64), "c5_cmp_large");
  H.emitStoreGate(Cmp);
  H.finish();

  auto Report = classifyPredicateChain(*H.F, kSrcWs, kTgtWs);
  EXPECT_FALSE(Report.Refused);
  EXPECT_EQ(Report.VisitedCalls, 1u);
}

// ---------------------------------------------------------------------
// Non-refusal: `icmp ult tid, %arg1` (dynamic kernarg bound). Matches
// the vecadd_f16 / rope_fp32 baseline shape from compare_correctness.
// ---------------------------------------------------------------------
TEST(C5PredicateChain, TidDynamicCmpAccepts) {
  Harness H;
  Value *Tid = H.emitTid();
  Value *Bound = H.F->getArg(1); // dynamic i32 kernarg
  Value *Cmp = H.B.CreateICmpULT(Tid, Bound, "c5_cmp_dynamic");
  H.emitStoreGate(Cmp);
  H.finish();

  auto Report = classifyPredicateChain(*H.F, kSrcWs, kTgtWs);
  EXPECT_FALSE(Report.Refused);
  EXPECT_EQ(Report.VisitedCalls, 1u);
}

// ---------------------------------------------------------------------
// Direction gate: source == target. Must no-op.
// ---------------------------------------------------------------------
TEST(C5PredicateChain, SameWaveDirectionGate) {
  Harness H;
  Value *Tid = H.emitTid();
  auto *I32Ty = Type::getInt32Ty(H.Ctx);
  Value *Cmp = H.B.CreateICmpULT(
      Tid, ConstantInt::get(I32Ty, 15), "c5_cmp_same_wave");
  H.emitStoreGate(Cmp);
  H.finish();

  // Same-wave (wave32 → wave32): direction gate triggers the early-
  // return in classifyPredicateChain BEFORE any tid-chain walk. The
  // kernel IR still contains the lane-position-scoped predicate,
  // but the classifier must stay quiet because modulo-replication
  // is a no-op when src == tgt.
  auto Report = classifyPredicateChain(*H.F, kSrcWs, /*tgt=*/kSrcWs);
  EXPECT_FALSE(Report.Refused);
  EXPECT_EQ(Report.VisitedCalls, 0u);
}

// ---------------------------------------------------------------------
// Direction gate: narrowing (wave64 source → wave32 target). Must no-op.
// ---------------------------------------------------------------------
TEST(C5PredicateChain, NarrowingDirectionGate) {
  Harness H;
  Value *Tid = H.emitTid();
  auto *I32Ty = Type::getInt32Ty(H.Ctx);
  Value *Cmp = H.B.CreateICmpULT(
      Tid, ConstantInt::get(I32Ty, 15), "c5_cmp_narrow");
  H.emitStoreGate(Cmp);
  H.finish();

  // Narrowing (wave64 → wave32): same reason as same-wave — MODREP
  // has no replica-1, so the predicate evaluates consistently
  // across every target lane. Classifier must not touch it.
  auto Report = classifyPredicateChain(*H.F, /*src=*/kTgtWs,
                                        /*tgt=*/kSrcWs);
  EXPECT_FALSE(Report.Refused);
  EXPECT_EQ(Report.VisitedCalls, 0u);
}

// ---------------------------------------------------------------------
// Projection gate: WaveNativeProjection MUST suppress the refusal,
// even on the exact IR shape that the default MODREP path would
// refuse. Pins the structural projection gate documented on the
// `waveNative` parameter in the header — regressions that accidentally
// drop the plumbing from loader/executable.cpp or raiser.cpp Phase 6.6
// fail this test.
// ---------------------------------------------------------------------
TEST(C5PredicateChain, WaveNativeProjectionGate) {
  Harness H;
  Value *Tid = H.emitTid();
  auto *I32Ty = Type::getInt32Ty(H.Ctx);
  Value *Cmp = H.B.CreateICmpULT(
      Tid, ConstantInt::get(I32Ty, 15), "c5_cmp_wave_native");
  H.emitStoreGate(Cmp);
  H.finish();

  // Sanity: under MODREP with unknown workgroup metadata this exact IR
  // refuses. Narrows the WaveNative assertion to "the projection is
  // what turns the refusal off", not "the IR happens to be safe".
  auto ModrepReport = classifyPredicateChain(
      *H.F, kSrcWs, kTgtWs, PredicateChainProjection::ModuloReplication);
  EXPECT_TRUE(ModrepReport.Refused);
  EXPECT_FALSE(ModrepReport.WaveNativePhantomRefusal);

  // WaveNative + no maxFlatWorkgroupSize given (default 0 == unknown
  // to the classifier). Suppression arm kicks in, walk still runs.
  auto WaveNativeReport = classifyPredicateChain(
      *H.F, kSrcWs, kTgtWs, PredicateChainProjection::WaveNative);
  EXPECT_FALSE(WaveNativeReport.Refused);
  EXPECT_FALSE(WaveNativeReport.WaveNativePhantomRefusal);
  // Walk still runs — `visitedCalls` reflects the tid call count,
  // and `observedSites` names the C5 shape so raiser.cpp can emit
  // LLVM_DEBUG attribution. Only the refusal itself is suppressed.
  EXPECT_EQ(WaveNativeReport.VisitedCalls, 1u);
  EXPECT_EQ(WaveNativeReport.ObservedSites.size(), 1u);
  EXPECT_TRUE(WaveNativeReport.RefusalDetail.empty());
  EXPECT_NE(WaveNativeReport.SuppressionReason.find("WaveNativeProjection"),
            std::string::npos);

  // WaveNative + maxFlatWorkgroupSize >= targetWaveSize (no phantom
  // lanes). Suppression arm kicks in — same shape as the default-0
  // case above.
  auto NoPhantomReport = classifyPredicateChain(
      *H.F, kSrcWs, kTgtWs, PredicateChainProjection::WaveNative,
      /*maxFlatWorkgroupSize=*/kTgtWs);
  EXPECT_FALSE(NoPhantomReport.Refused);
  EXPECT_FALSE(NoPhantomReport.WaveNativePhantomRefusal);
  EXPECT_EQ(NoPhantomReport.ObservedSites.size(), 1u);
  EXPECT_NE(NoPhantomReport.SuppressionReason.find("target wave size"),
            std::string::npos);
}

TEST(C5PredicateChain, WaveNativeEqualityPredicateUsesMaskShadow) {
  Harness H;
  Value *Tid = H.emitTid();
  auto *I32Ty = Type::getInt32Ty(H.Ctx);
  Value *Cmp = H.B.CreateICmpEQ(
      Tid, ConstantInt::get(I32Ty, 16), "c5_cmp_wave_native_eq");
  H.emitStoreGate(Cmp);
  H.finish();

  auto Report = classifyPredicateChain(
      *H.F, kSrcWs, kTgtWs, PredicateChainProjection::WaveNative,
      /*maxFlatWorkgroupSize=*/kTgtWs);
  EXPECT_FALSE(Report.Refused);
  EXPECT_FALSE(Report.WaveNativePhantomRefusal);
  EXPECT_TRUE(Report.WaveNativeEqualityObserved);
  EXPECT_EQ(Report.ObservedSites.size(), 1u);
  EXPECT_TRUE(Report.RefusalDetail.empty());
  EXPECT_NE(Report.SuppressionReason.find("mask-shadow"),
            std::string::npos);
}

TEST(C5PredicateChain, ThreadLoopProjectionGate) {
  Harness H;
  Value *Tid = H.emitTid();
  auto *I32Ty = Type::getInt32Ty(H.Ctx);
  Value *Cmp = H.B.CreateICmpULT(
      Tid, ConstantInt::get(I32Ty, 15), "c5_cmp_thread_loop");
  H.emitStoreGate(Cmp);
  H.finish();

  auto ModrepReport = classifyPredicateChain(
      *H.F, kSrcWs, kTgtWs, PredicateChainProjection::ModuloReplication);
  EXPECT_TRUE(ModrepReport.Refused);

  auto ThreadLoopReport = classifyPredicateChain(
      *H.F, kSrcWs, kTgtWs, PredicateChainProjection::ThreadLoop,
      /*maxFlatWorkgroupSize=*/kSrcWs, /*suppressThreadLoopC5=*/true);
  EXPECT_FALSE(ThreadLoopReport.Refused);
  EXPECT_FALSE(ThreadLoopReport.WaveNativePhantomRefusal);
  EXPECT_EQ(ThreadLoopReport.VisitedCalls, 1u);
  EXPECT_EQ(ThreadLoopReport.ObservedSites.size(), 1u);
  EXPECT_TRUE(ThreadLoopReport.RefusalDetail.empty());
  EXPECT_NE(ThreadLoopReport.SuppressionReason.find("ThreadLoopProjection"),
            std::string::npos);
}

// ---------------------------------------------------------------------
// WaveNative + phantom-lane regime (maxFlatWorkgroupSize < targetWaveSize):
// the classifier tightens back to refusal. Pins the
// `canary_bitmatrix_composite` contract — the docstring-named
// soundness gap of the permissive WaveNative default — and its
// complement (the suppression must STILL apply when the WG honours
// the target wavefront width).
// ---------------------------------------------------------------------
TEST(C5PredicateChain, WaveNativePhantomLaneRegimeRefuses) {
  Harness H;
  Value *Tid = H.emitTid();
  auto *I32Ty = Type::getInt32Ty(H.Ctx);
  Value *Cmp = H.B.CreateICmpULT(
      Tid, ConstantInt::get(I32Ty, 15), "c5_cmp_phantom_lane");
  H.emitStoreGate(Cmp);
  H.finish();

  // Phantom-lane guaranteed: max_flat_workgroup_size = source wave
  // size (32) is BELOW the target wave size (64), so a direct WaveNative
  // caller would make lanes outside the source lane space active via
  // init_whole_wave. The classifier must refuse under WaveNative with
  // `waveNativePhantomRefusal == true` so downstream attribution
  // distinguishes this defensive arm from the baseline MODREP refusal.
  auto PhantomReport =
      classifyPredicateChain(*H.F, kSrcWs, kTgtWs,
                              PredicateChainProjection::WaveNative,
                              /*maxFlatWorkgroupSize=*/kSrcWs);
  EXPECT_TRUE(PhantomReport.Refused);
  EXPECT_TRUE(PhantomReport.WaveNativePhantomRefusal);
  EXPECT_EQ(PhantomReport.ObservedSites.size(), 1u);
  // Diagnostic names the phantom-lane rationale so operators can
  // distinguish the two refusal arms at triage time. Match on a
  // stable substring — the full wording is longer and the lit
  // fixtures pin the exact shape.
  EXPECT_NE(PhantomReport.RefusalDetail.find("phantom-lane regime"),
            std::string::npos);
  EXPECT_NE(PhantomReport.RefusalDetail.find("max_flat_workgroup_size"),
            std::string::npos);
  // The base C5-shape diagnostic is still present (the phantom-lane
  // detail is a prefix on the underlying refusal reason, not a
  // replacement).
  EXPECT_NE(PhantomReport.RefusalDetail.find("compile-time constant 15"),
            std::string::npos);
}

// ---------------------------------------------------------------------
// The phantom-lane tightening must NOT fire for the no-evidence case
// (maxFlatWorkgroupSize = 0 meaning "caller doesn't know"). Conservative
// default: preserve the historical WaveNative suppression rather than
// refuse every kernel whose caller omits the metadata. Pins the
// `> 0` guard in the classifier.
// ---------------------------------------------------------------------
TEST(C5PredicateChain, WaveNativeUnknownWorkgroupSizeKeepsSuppression) {
  Harness H;
  Value *Tid = H.emitTid();
  auto *I32Ty = Type::getInt32Ty(H.Ctx);
  Value *Cmp = H.B.CreateICmpULT(
      Tid, ConstantInt::get(I32Ty, 15), "c5_cmp_unknown_wg");
  H.emitStoreGate(Cmp);
  H.finish();

  // Explicitly pass `maxFlatWorkgroupSize = 0` to assert the
  // "unknown" sentinel does NOT trigger phantom-lane refusal.
  auto Report =
      classifyPredicateChain(*H.F, kSrcWs, kTgtWs,
                              PredicateChainProjection::WaveNative,
                              /*maxFlatWorkgroupSize=*/0u);
  EXPECT_FALSE(Report.Refused);
  EXPECT_FALSE(Report.WaveNativePhantomRefusal);
  EXPECT_NE(Report.SuppressionReason.find("unknown max_flat_workgroup_size"),
            std::string::npos);
}

// ---------------------------------------------------------------------
// The phantom-lane rule MUST be WaveNative-only — under MODREP the
// MODREP refusal arm already fires unconditionally on any C5 site, so
// the phantom-lane bit is an orthogonal signal the classifier tracks
// but the MODREP path should not flip on. Pins the "MODREP arm is
// independent of maxFlatWorkgroupSize" contract.
// ---------------------------------------------------------------------
TEST(C5PredicateChain, ModrepSingleSourceWaveAccepts) {
  Harness H;
  Value *Tid = H.emitTid();
  auto *I32Ty = Type::getInt32Ty(H.Ctx);
  Value *Cmp = H.B.CreateICmpULT(
      Tid, ConstantInt::get(I32Ty, 15), "c5_cmp_modrep_wg");
  H.emitStoreGate(Cmp);
  H.finish();

  // MODREP + a statically single-source-wave workgroup has no active
  // target replica lanes: lanes >= W_s remain hardware-inactive for the
  // whole kernel, so the C5 replica-divergence obligation does not apply.
  for (unsigned Wg : {1u, kSrcWs - 1, kSrcWs}) {
    auto Report = classifyPredicateChain(
        *H.F, kSrcWs, kTgtWs, PredicateChainProjection::ModuloReplication,
        /*maxFlatWorkgroupSize=*/Wg);
    EXPECT_FALSE(Report.Refused) << "wg=" << Wg;
    EXPECT_FALSE(Report.WaveNativePhantomRefusal) << "wg=" << Wg;
    EXPECT_EQ(Report.ObservedSites.size(), 1u) << "wg=" << Wg;
    EXPECT_TRUE(Report.RefusalDetail.empty()) << "wg=" << Wg;
    EXPECT_NE(Report.SuppressionReason.find("no active target replica lanes"),
              std::string::npos)
        << "wg=" << Wg << " reason='" << Report.SuppressionReason << "'";
  }
}

TEST(C5PredicateChain, ModrepUnknownOrActiveReplicaRefuses) {
  Harness H;
  Value *Tid = H.emitTid();
  auto *I32Ty = Type::getInt32Ty(H.Ctx);
  Value *Cmp = H.B.CreateICmpULT(
      Tid, ConstantInt::get(I32Ty, 15), "c5_cmp_modrep_active_replica");
  H.emitStoreGate(Cmp);
  H.finish();

  // Unknown metadata fails conservative. Any workgroup bound above W_s can
  // activate at least one target lane outside the source-wave lane domain,
  // so the original MODREP C5 refusal still fires.
  for (unsigned Wg : {0u, kSrcWs + 1, kTgtWs, 4u * kTgtWs}) {
    auto Report = classifyPredicateChain(
        *H.F, kSrcWs, kTgtWs, PredicateChainProjection::ModuloReplication,
        /*maxFlatWorkgroupSize=*/Wg);
    EXPECT_TRUE(Report.Refused) << "wg=" << Wg;
    EXPECT_FALSE(Report.WaveNativePhantomRefusal) << "wg=" << Wg;
    EXPECT_NE(Report.RefusalDetail.find("compile-time constant 15"),
              std::string::npos)
        << "wg=" << Wg << " detail='" << Report.RefusalDetail << "'";
  }
}

// ---------------------------------------------------------------------
// Functions with no `workitem.id.x()` call must return the empty
// report. Pins the initial site collection's zero-case.
// ---------------------------------------------------------------------
TEST(C5PredicateChain, NoCallsIsNoOp) {
  Harness H;
  // No tid read — just a constant-gated store. Nothing for the
  // classifier to walk.
  auto *I32Ty = Type::getInt32Ty(H.Ctx);
  Value *Cmp = H.B.CreateICmpULT(
      H.F->getArg(1), ConstantInt::get(I32Ty, 15), "not_a_tid");
  H.emitStoreGate(Cmp);
  H.finish();

  auto Report = classifyPredicateChain(*H.F, kSrcWs, kTgtWs);
  EXPECT_FALSE(Report.Refused);
  EXPECT_EQ(Report.VisitedCalls, 0u);
}

// ---------------------------------------------------------------------
// Refusal path through a phi node: `tid → phi [tid, entry] [tid, other]
// → icmp ult %phi, 15 → br → store`. Phi is a pure propagator per the
// classifier's walk rules; the refusal must fire through it.
// ---------------------------------------------------------------------
TEST(C5PredicateChain, PhiPropagatesTidDerivation) {
  Harness H;
  Value *Tid = H.emitTid();

  // Synthesise a two-predecessor phi: branch unconditionally from
  // entry to `join`, but also create a dummy `other` block that
  // merges in. This gives us a real phi whose only incoming is
  // `tid` from both sides.
  BasicBlock *Other = BasicBlock::Create(H.Ctx, "other", H.F);
  BasicBlock *Join = BasicBlock::Create(H.Ctx, "join", H.F);
  H.B.CreateBr(Join);

  H.B.SetInsertPoint(Other);
  H.B.CreateBr(Join);

  H.B.SetInsertPoint(Join);
  auto *Phi = H.B.CreatePHI(Tid->getType(), 2, "tid_phi");
  Phi->addIncoming(Tid, H.Entry);
  Phi->addIncoming(Tid, Other);

  auto *I32Ty = Type::getInt32Ty(H.Ctx);
  Value *Cmp = H.B.CreateICmpULT(
      Phi, ConstantInt::get(I32Ty, 15), "c5_cmp_phi");
  H.emitStoreGate(Cmp);
  H.finish();

  auto Report = classifyPredicateChain(*H.F, kSrcWs, kTgtWs);
  EXPECT_TRUE(Report.Refused);
  EXPECT_EQ(Report.VisitedCalls, 1u);
}

// ---------------------------------------------------------------------
// Refusal through a phi with one MASKED and one UNMASKED arm: the
// classifier must still refuse because the unmasked arm reaches the
// icmp. This pins the behaviour observed in `swiglu_fp32`'s IR
// (phi-arm asymmetry from Triton's SPE diamond): the SPE
// diamond produces `phi [vand, spe_do], [tid, entry]` where the
// entry arm is unmasked, and the downstream icmp against the phi
// result is reachable from an unmasked tid path. (The swiglu end-
// to-end kernel is NOT refused because its icmp constant is
// dynamic; `TidDynamicCmpAccepts` pins that. This test pins the
// phi-arm asymmetry in isolation on a compile-time K so the
// refusal-through-phi semantics stay stable under future walker
// refactors.)
// ---------------------------------------------------------------------
TEST(C5PredicateChain, MaskedPhiThroughUnmaskedArmRefuses) {
  Harness H;
  Value *Tid = H.emitTid();
  auto *I32Ty = Type::getInt32Ty(H.Ctx);

  BasicBlock *MaskedArm = BasicBlock::Create(H.Ctx, "masked_arm", H.F);
  BasicBlock *Join = BasicBlock::Create(H.Ctx, "join", H.F);

  // Split on an arbitrary predicate that is NOT tid-derived, so the
  // branch itself is not a C5 site — only the phi at `join` is what
  // the classifier must audit. We use `arg1 != 0` for this.
  Value *SplitCond = H.B.CreateICmpNE(
      H.F->getArg(1), ConstantInt::get(I32Ty, 0), "split_cond");
  H.B.CreateCondBr(SplitCond, MaskedArm, Join);

  H.B.SetInsertPoint(MaskedArm);
  Value *Vand = H.B.CreateAnd(Tid, ConstantInt::get(I32Ty, 31),
                               "vand");
  H.B.CreateBr(Join);

  H.B.SetInsertPoint(Join);
  auto *Phi = H.B.CreatePHI(Tid->getType(), 2, "tid_phi");
  Phi->addIncoming(Tid, H.Entry);       // unmasked arm
  Phi->addIncoming(Vand, MaskedArm);    // masked arm

  Value *Cmp = H.B.CreateICmpULT(
      Phi, ConstantInt::get(I32Ty, 15), "c5_cmp_phi_mixed");
  H.emitStoreGate(Cmp);
  H.finish();

  auto Report = classifyPredicateChain(*H.F, kSrcWs, kTgtWs);
  EXPECT_TRUE(Report.Refused);
  EXPECT_EQ(Report.VisitedCalls, 1u);
}

// ---------------------------------------------------------------------
// Cross-subtree non-refusal: `icmp (tid+1)_unmasked, (tid&15)_masked`.
// Both icmp operands are tid-derived via different walks: operand 0
// is `add tid, 1` (unmasked), operand 1 is `and tid, 15` (masked).
// An earlier iteration of the classifier refused this shape on the
// theory that replica-0's `(L+1, L&15)` and replica-1's
// `(L+33, L&15)` evaluate the predicate differently. For
// `ult`/`ugt`/`slt`/`sgt` the specific value ranges make the
// predicate evaluate identically across replicas — e.g.
// `icmp ult (L+1), (L&15)` is false for every L in [0, 31] and
// `icmp ult (L+33), (L&15)` is also false for every L in [0, 31],
// so both replicas agree. Only `eq`/`ne` variants genuinely
// diverge (`icmp eq tid, (tid&15)` is true iff `L < 16`, which
// differs between replicas); that per-predicate tightening is not
// implemented today. compare_correctness 2026-04-21 confirmed
// `vecadd_f16`, `corpus_add_fp32`, `corpus_asin_fp32`, and
// `canary_dpp_reduce_fp32` all emit this shape under Triton's
// `icmp sgt tid_unmasked, tid_masked` bounds-check idiom and pass
// MATCH end-to-end under MODREP.
//
// This test pins the non-refusal so a future iteration that
// re-introduces the cross-subtree rule without per-predicate
// reasoning fails here.
// ---------------------------------------------------------------------
TEST(C5PredicateChain, CrossSubtreeMaskedVsUnmaskedAccepts) {
  Harness H;
  Value *Tid = H.emitTid();
  auto *I32Ty = Type::getInt32Ty(H.Ctx);
  Value *Plus1 = H.B.CreateAdd(Tid, ConstantInt::get(I32Ty, 1),
                                "tid_plus_1");
  Value *Masked = H.B.CreateAnd(Tid, ConstantInt::get(I32Ty, 15),
                                 "tid_masked_15");
  Value *Cmp = H.B.CreateICmpULT(Plus1, Masked, "c5_cmp_cross_subtree");
  H.emitStoreGate(Cmp);
  H.finish();

  auto Report = classifyPredicateChain(*H.F, kSrcWs, kTgtWs);
  EXPECT_FALSE(Report.Refused);
  EXPECT_EQ(Report.VisitedCalls, 1u);
  EXPECT_TRUE(Report.RefusalDetail.empty());
  EXPECT_TRUE(Report.ObservedSites.empty());
}

// ---------------------------------------------------------------------
// Intrinsic-propagator refusal: tid flows through `@llvm.umin(tid,
// 100)` and then into an `icmp ult %umin, 15`. Pins #8 — without
// explicit enumeration of numeric intrinsics as propagators the
// walk would stop at the `umin` call and the downstream C5 icmp
// would be missed.
//
// Semantic check: `umin(tid, 100)` is NOT a mask (it clamps at 100,
// not at W_s-1=31). For replica-0 L in [0,31]: result = L. For
// replica-1 L+32 in [32, 63]: result = L+32 (since both < 100). So
// the umin output still carries per-replica divergence — classifier
// must treat it as unmasked-propagator.
// ---------------------------------------------------------------------
TEST(C5PredicateChain, IntrinsicPropagatorRefuses) {
  Harness H;
  Value *Tid = H.emitTid();
  auto *I32Ty = Type::getInt32Ty(H.Ctx);
  Function *UminDecl = Intrinsic::getOrInsertDeclaration(
      H.M.get(), Intrinsic::umin, {I32Ty});
  Value *Umin = H.B.CreateCall(
      UminDecl, {Tid, ConstantInt::get(I32Ty, 100)}, "umin_tid_100");
  Value *Cmp = H.B.CreateICmpULT(
      Umin, ConstantInt::get(I32Ty, 15), "c5_cmp_through_umin");
  H.emitStoreGate(Cmp);
  H.finish();

  auto Report = classifyPredicateChain(*H.F, kSrcWs, kTgtWs);
  EXPECT_TRUE(Report.Refused);
  EXPECT_EQ(Report.VisitedCalls, 1u);
  EXPECT_NE(Report.RefusalDetail.find("compile-time constant 15"),
            std::string::npos)
      << "refusalDetail='" << Report.RefusalDetail << "'";
}
