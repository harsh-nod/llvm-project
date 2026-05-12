cmake_minimum_required(VERSION 3.20)
project(hotswap-transpiler LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- LLVM ---------------------------------------------------------------------
# The MC stack uses AMDGPU target-private headers (SIDefines.h, SIInstrInfo.h,
# AMDGPUBaseInfo.h) which transitively include TableGen-generated
# AMDGPUGen*.inc files. Those .inc files are produced by llvm-tblgen during
# the LLVM build and are not copied by the `install` target, so we require
# an LLVM *build tree* rather than an install tree.
if(TARGET LLVMSupport AND DEFINED LLVM_MAIN_SRC_DIR)
  set(LLVM_BUILD_MAIN_SRC_DIR "${LLVM_MAIN_SRC_DIR}")
else()
  find_package(LLVM REQUIRED CONFIG)
endif()
if(NOT COMMAND llvm_update_compile_flags)
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  include(AddLLVM)
endif()

if(NOT DEFINED LLVM_BUILD_MAIN_SRC_DIR OR NOT LLVM_BUILD_MAIN_SRC_DIR OR
   NOT DEFINED LLVM_BINARY_DIR OR NOT LLVM_BINARY_DIR)
  message(FATAL_ERROR
    "hotswap-transpiler requires an LLVM build tree (not an install tree). "
    "The install tree does not expose AMDGPU target-private headers "
    "(SIDefines.h, AMDGPUBaseInfo.h, AMDGPUGen*.inc). "
    "Pass -DLLVM_DIR=<llvm-build>/lib/cmake/llvm.")
endif()

include_directories(${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})
add_definitions(${LLVM_DEFINITIONS})

# --- TDM emulation runtime helper (HIP -> bitcode -> embedded array) --------
# When `hipcc` is available we build `runtime/tdm.hip` to LLVM bitcode for
# gfx942 and embed the bytes via `cmake/EmbedBinary.cmake`. The transpiler
# parses and link-merges the bitcode into each raised IR module that uses
# the cross-target VIMAGE TENSOR helper (see `tdm_runtime.{h,cpp}` and the
# raiser's cross-target VIMAGE handling). When hipcc is missing the embedded
# blob is empty (one-byte stub in tdm-runtime.cpp), `tdmRuntimeAvailable()`
# returns false, and the cross-target VIMAGE handler keeps its loud refusal.
# This preserves the no-hipcc build behaviour exactly.
find_program(HIPCC hipcc)
set(HOTSWAP_TDM_RUNTIME_GENERATED_CPP "")
if(HIPCC)
  set(_tdm_src "${CMAKE_CURRENT_SOURCE_DIR}/runtime/tdm.hip")
  set(_tdm_bc  "${CMAKE_CURRENT_BINARY_DIR}/tdm_runtime.gfx942.bc")
  set(_tdm_cpp "${CMAKE_CURRENT_BINARY_DIR}/tdm_runtime_blob.cpp")

  add_custom_command(
    OUTPUT ${_tdm_bc}
    COMMAND ${HIPCC}
            --offload-arch=gfx942
            -O2 -emit-llvm -c
            -fgpu-rdc
            --cuda-device-only
            -o ${_tdm_bc} ${_tdm_src}
    DEPENDS ${_tdm_src}
    COMMENT "Building TDM emulation runtime bitcode (gfx942)"
    VERBATIM)

  add_custom_command(
    OUTPUT ${_tdm_cpp}
    COMMAND ${CMAKE_COMMAND}
            -DINPUT=${_tdm_bc}
            -DOUTPUT=${_tdm_cpp}
            -DSYMBOL=tdm_runtime_bc
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedBinary.cmake
    DEPENDS ${_tdm_bc} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedBinary.cmake
    COMMENT "Embedding TDM runtime bitcode into C++ blob"
    VERBATIM)

  set(HOTSWAP_TDM_RUNTIME_GENERATED_CPP ${_tdm_cpp})
  set(HOTSWAP_HAVE_TDM_RUNTIME 1)
  message(STATUS "TDM runtime helper enabled (hipcc=${HIPCC})")
else()
  set(HOTSWAP_HAVE_TDM_RUNTIME 0)
  message(STATUS "TDM runtime helper DISABLED (hipcc not found); "
                 "cross-target gfx1250->gfx942 TENSOR ops will keep refusing")
endif()

# Surface the flag to other subdirectories (e.g. test-lit, which gates the
# `tdm-runtime` lit feature on it so the tensor_*_lds.s fixtures only run
# when hipcc was present at configure time).
set(HOTSWAP_HAVE_TDM_RUNTIME ${HOTSWAP_HAVE_TDM_RUNTIME} CACHE INTERNAL
    "Hotswap built with TDM emulation runtime (hipcc available at configure time)")

# --- Library ------------------------------------------------------------------
# OBJECT library so its translation units land directly in amd_comgr.so when
# `target_link_libraries(amd_comgr PRIVATE hotswap::transpiler)` runs (gated
# on COMGR_ENABLE_HOTSWAP_TRANSPILE in the parent CMakeLists.txt). This puts
# the hotswap raiser and the rest of comgr in the same TU set so hotswap
# files can call comgr-metadata helpers directly without a layering inversion
# or a separate static archive.
add_library(hotswap-transpiler OBJECT
  raiser.cpp
  code-object-utils.cpp
  mc-state.cpp
  hotswap-error.cpp
  canonical-op.cpp
  opcode-map.cpp
  decode.cpp
  raise-failure.cpp
  raise-context.cpp
  reg-file.cpp
  kernarg-layout.cpp
  user-sgpr-layout.cpp
  wave-projection.cpp
  setpc-analysis.cpp
  handle-sopp.cpp
  handle-sopc.cpp
  handle-sop1.cpp
  handle-sop2.cpp
  handle-sopk.cpp
  canonical-op-attrs.cpp
  handle-smem.cpp
  source-hidden-args.cpp
  handle-flat.cpp
  flat-addr.cpp
  handle-mubuf.cpp
  mubuf-addr.cpp
  handle-ds.cpp
  handle-valu.cpp
  handle-valu-output-mods.cpp
  handle-valu-small-ops.cpp
  handle-valu-cross-lane.cpp
  handle-valu-vcmp.cpp
  handle-valu-vop3p.cpp
  wmma-lowering.cpp
  mxfp4-dequant.cpp
  handle-mfma.cpp
  handle-vopd.cpp
  handle-vimage.cpp
  tdm-runtime.cpp
  ${HOTSWAP-TDM-RUNTIME-GENERATED-CPP}
  wave-size-obstruction.cpp
)

if(NOT TARGET hotswap::transpiler)
  add_library(hotswap::transpiler ALIAS hotswap-transpiler)
endif()

target_compile_definitions(hotswap-transpiler PRIVATE
  HOTSWAP_HAVE_TDM_RUNTIME=${HOTSWAP_HAVE_TDM_RUNTIME}
)

# Match LLVM's compile flags (no-rtti, no-exceptions) — every LLVM type we
# consume relies on LLVM's hand-rolled RTTI rather than C++ typeid; mixing
# RTTI-on TUs with RTTI-off LLVM libs produces undefined-reference errors
# for `typeinfo` symbols on any class with a virtual method.
llvm_update_compile_flags(hotswap-transpiler)

set_target_properties(hotswap-transpiler PROPERTIES POSITION_INDEPENDENT_CODE ON)

# AMDGPU target-private headers (not exposed by the LLVM install tree).
target_include_directories(hotswap-transpiler PRIVATE
  "${LLVM_BUILD_MAIN_SRC_DIR}/lib/Target/AMDGPU"
  "${LLVM_BUILD_MAIN_SRC_DIR}/lib/Target/AMDGPU/Utils"
  "${LLVM_BINARY_DIR}/lib/Target/AMDGPU"
)

# Public include root so consumers can `#include "hotswap/raiser.hpp"`.
target_include_directories(hotswap-transpiler PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
)

target_link_libraries(hotswap-transpiler
  PUBLIC
    # Core LLVM components, sorted alphabetically.
    LLVMBinaryFormat
    LLVMCore
    LLVMMC
    LLVMMCDisassembler
    LLVMMCParser
    LLVMObject
    LLVMSupport
    LLVMTarget
    LLVMTargetParser
    # AMDGPU target components, sorted alphabetically.
    LLVMAMDGPUAsmParser
    LLVMAMDGPUCodeGen
    LLVMAMDGPUDesc
    LLVMAMDGPUDisassembler
    LLVMAMDGPUInfo
    LLVMAMDGPUUtils
)

# Per-file gtests live under amd/comgr/test-unit/ and follow the 3-tier
# llvm_gtest / find_package(GTest CONFIG) / LLVM-installed fallback discovery
# there. No standalone gtest target lives in this subdirectory.
