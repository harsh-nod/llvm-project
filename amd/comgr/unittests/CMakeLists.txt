set(LLVM_LINK_COMPONENTS
  MC
  MCDisassembler
  MCParser
  Object
  Support
  TargetParser
  AllTargetsDescs
  AllTargetsDisassemblers
  AllTargetsInfos
  AllTargetsMCAs
  AMDGPUUtils
  )

add_unittest(check-comgr HotswapTests
  HotswapTest.cpp
  $<TARGET_OBJECTS:comgr_hotswap_obj>)

target_include_directories(HotswapTests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../src
  ${LLVM_MAIN_SRC_DIR}/lib/Target/AMDGPU
  ${LLVM_BINARY_DIR}/lib/Target/AMDGPU
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../include>)
