//===- source-hidden-args.h - Hotswap transpiler --------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_SOURCE_HIDDEN_ARGS_H
#define HOTSWAP_TRANSPILER_SOURCE_HIDDEN_ARGS_H

#include "code-object-utils.h"
#include "kernarg-layout.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Twine.h"

#include <string>

namespace llvm {
class Function;
class IRBuilderBase;
class LLVMContext;
class Module;
class Type;
class Value;
} // namespace llvm

namespace COMGR::hotswap {

struct SourceHiddenArgContext {
  llvm::LLVMContext &C;
  llvm::Module &M;
  llvm::IRBuilderBase &B;
  llvm::Type *I8Ty;
  llvm::Type *I32Ty;
  llvm::Type *I64Ty;
  llvm::ArrayRef<KernelArgMeta> Args;
};

struct SourceHiddenArgValue {
  // True when ByteOffset maps to a source metadata hidden_* field.
  bool Matched = false;
  // Non-null when a matched hidden field was lowered successfully.
  llvm::Value *Value = nullptr;
  // Non-empty when Matched is true and Value is null.
  std::string FailureDetail;
};

SourceHiddenArgValue emitSourceHiddenDword(SourceHiddenArgContext &Ctx,
                                           int ByteOffset);
SourceHiddenArgValue emitSourceHiddenInteger(SourceHiddenArgContext &Ctx,
                                             int ByteOffset,
                                             unsigned ByteWidth,
                                             bool IsSigned);

} // namespace COMGR::hotswap

#endif
