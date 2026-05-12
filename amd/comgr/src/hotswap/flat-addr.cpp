//===- flat-addr.cpp - Hotswap transpiler ---------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flat-addr.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

using namespace llvm;

namespace COMGR::hotswap {

namespace {

// Scan the operand tail (at and after `immStart`) for the first immediate
// and return its value. Any later imms are encoding flags (cpol, th,
// scope) and are ignored. Signed 13-bit offset; already sign-extended
// by the MC layer.
int64_t firstImmOffset(const DecodedInst &Di, OpResolver &Op,
                       unsigned ImmStart) {
  for (unsigned K = ImmStart; K < Op.nSrcs(); ++K) {
    if (Di.isImm(Op.srcIdx(K)))
      return Di.getImm(Op.srcIdx(K));
  }
  return 0;
}

// Coerce an integer address into a global-AS pointer and apply a signed
// byte offset via a plain (non-inbounds) GEP. The ISA's signed offset
// can legitimately leave the base allocation (e.g. compiler-scheduled
// prefetches, negative strides); `inbounds` would turn that into UB.
Value *toGlobalPtr(RaiseContext &Ctx, Value *Addr, int64_t MemOffset) {
  if (Addr->getType() != Ctx.PtrGlobalTy)
    Addr = Ctx.B.CreateIntToPtr(Addr, Ctx.PtrGlobalTy);
  if (MemOffset != 0)
    Addr = Ctx.B.CreateGEP(Ctx.I8Ty, Addr, Ctx.B.getInt64(MemOffset));
  return Addr;
}

} // namespace

FlatAddr decodeGlobalLoadAddr(RaiseContext &Ctx, const DecodedInst &Di,
                               OpResolver &Op, int ElemBytes,
                               StringRef DiagLabel) {
  FlatAddr Out;
  Value *Addr = nullptr;

  // SADDR form: saddr(SGPR64), vaddr(VGPR32), ... -- LLVM MC places the
  // SGPR first in the decoded operand order.
  if (Op.nSrcs() >= 2 && Op.isSrcReg(0) && Op.isSrcReg(1) &&
      Op.srcReg(0).RegKind == ParsedReg::SGPR &&
      Op.srcReg(1).RegKind == ParsedReg::VGPR) {
    Out.HasSaddr = true;
    Value *Saddr = Ctx.Regs.readReg64(Ctx.B, Op.srcReg(0));
    Value *Vaddr = Ctx.B.CreateSExt(Ctx.Regs.readReg32(Ctx.B, Op.srcReg(1)),
                                    Ctx.I64Ty, "voff_sext");
    if (Di.HasScaleOffset)
      Vaddr = Ctx.B.CreateMul(Vaddr, ConstantInt::get(Ctx.I64Ty, ElemBytes),
                              "scaled_voff");
    Addr = Ctx.B.CreateAdd(Saddr, Vaddr, "saddr_vaddr");
  } else if (Op.nSrcs() >= 1 && Op.isSrcReg(0) &&
             Op.srcReg(0).RegKind == ParsedReg::VGPR) {
    // Plain form: VGPR64 holds the full per-lane address. Do NOT gate on
    // width -- parseReg currently reports tuple VGPRs (e.g. VGPR2_VGPR3)
    // with width=1 on some subtargets; readReg64 walks the sub0/sub1
    // graph itself, so trust the SGPR-vs-VGPR discriminator above and
    // let readReg64 enforce the 64-bit shape.
    Addr = Ctx.Regs.readReg64(Ctx.B, Op.srcReg(0));
  } else {
    std::string Msg;
    raw_string_ostream Os(Msg);
    Os << "transpiler: unrecognized " << DiagLabel
       << " operand shape (expected plain VGPR64 or SADDR SGPR64+VGPR32): \""
       << Di.FullText << "\" (mnemonic=" << Di.RawMnemonic << ")";
    report_fatal_error(StringRef(Os.str()));
  }

  Out.MemOffset = firstImmOffset(Di, Op, Out.HasSaddr ? 2 : 1);
  Out.Ptr = toGlobalPtr(Ctx, Addr, Out.MemOffset);
  return Out;
}

FlatAddr decodeGlobalStoreAddr(RaiseContext &Ctx, const DecodedInst &Di,
                                OpResolver &Op, int ElemBytes,
                                StringRef DiagLabel) {
  FlatAddr Out;
  Value *Addr = nullptr;

  // SADDR form: vaddr(VGPR32), vdata(VGPR*), saddr(SGPR64), ...
  if (Op.nSrcs() >= 3 && Op.isSrcReg(0) && Op.isSrcReg(1) && Op.isSrcReg(2) &&
      Op.srcReg(0).RegKind == ParsedReg::VGPR &&
      Op.srcReg(1).RegKind == ParsedReg::VGPR &&
      Op.srcReg(2).RegKind == ParsedReg::SGPR) {
    Out.HasSaddr = true;
    Value *Saddr = Ctx.Regs.readReg64(Ctx.B, Op.srcReg(2));
    Value *Vaddr = Ctx.B.CreateSExt(Ctx.Regs.readReg32(Ctx.B, Op.srcReg(0)),
                                    Ctx.I64Ty, "st_voff_sext");
    if (Di.HasScaleOffset)
      Vaddr = Ctx.B.CreateMul(Vaddr, ConstantInt::get(Ctx.I64Ty, ElemBytes),
                              "st_scaled_voff");
    Addr = Ctx.B.CreateAdd(Saddr, Vaddr, "st_saddr_vaddr");
    Out.StData = Op.srcReg(1);
  } else if (Op.nSrcs() >= 2 && Op.isSrcReg(0) && Op.isSrcReg(1) &&
             Op.srcReg(0).RegKind == ParsedReg::VGPR &&
             Op.srcReg(1).RegKind == ParsedReg::VGPR) {
    Addr = Ctx.Regs.readReg64(Ctx.B, Op.srcReg(0));
    Out.StData = Op.srcReg(1);
  } else {
    std::string Msg;
    raw_string_ostream Os(Msg);
    Os << "transpiler: unrecognized " << DiagLabel
       << " operand shape (expected plain VGPR+VGPR or SADDR VGPR+VGPR+SGPR): \""
       << Di.FullText << "\" (mnemonic=" << Di.RawMnemonic << ")";
    report_fatal_error(StringRef(Os.str()));
  }

  Out.MemOffset = firstImmOffset(Di, Op, Out.HasSaddr ? 3 : 2);
  Out.Ptr = toGlobalPtr(Ctx, Addr, Out.MemOffset);
  return Out;
}

} // namespace COMGR::hotswap
