; RUN: %llvm_mc -mcpu=gfx942 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=v_fma_f16_basic_kernel 2>/dev/null \
; RUN:   | %FileCheck %s --check-prefix=BASIC
; RUN: %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=v_fma_f16_opsel_kernel 2>/dev/null \
; RUN:   | %FileCheck %s --check-prefix=OPSEL
; RUN: %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=v_fma_f16_dsthi_kernel 2>/dev/null \
; RUN:   | %FileCheck %s --check-prefix=DSTHI
; RUN: %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=v_fma_f16_neg_kernel 2>/dev/null \
; RUN:   | %FileCheck %s --check-prefix=NEG
;
; Pins VOP3 V_FMA_F16 (gfx9+ V_FMA_F16_gfx9_e64 pseudo). Verifies:
;   * Default op_sel ([0,0,0,0]) reads the low 16 bits of each source
;     and writes the result into the low 16 bits of the destination
;     while preserving the destination's high half (BASIC).
;   * Source-half op_sel bits route the high 16 bits of the relevant
;     source through `lshr i32 ..., 16` before the trunc/bitcast (OPSEL).
;   * Destination op_sel bit 3 toggles the writeback to the high half
;     and preserves the destination's low half (DSTHI).
;   * VOP3 neg/abs source modifiers lower to `fneg half` / `llvm.fabs.f16`
;     before the fused multiply-add (NEG).
;
; All four kernels assemble the gfx9+ V_FMA_F16 pseudo
; (V_FMA_F16_gfx9_e64 after t16/fake16 collapse) and raise to the same
; ISA so the test exercises the in-target round-trip for the f16 fused
; multiply-add lowering in handle-valu.cpp.

; BASIC-LABEL: define amdgpu_kernel void @v_fma_f16_basic_kernel(
; BASIC-DAG: trunc i32 {{.*}} to i16
; BASIC-DAG: bitcast i16 {{.*}} to half
; BASIC: %fma_f16 = call half @llvm.fma.f16(
; BASIC: bitcast half %fma_f16 to i16
; BASIC: zext i16 {{.*}} to i32
; BASIC: and i32 {{.*}}, -65536
; BASIC: %f16_merge_lo = or i32
; BASIC-NOT: unsupported instruction

; OPSEL-LABEL: define amdgpu_kernel void @v_fma_f16_opsel_kernel(
; OPSEL-DAG: %f16_src_hi = lshr i32 {{.*}}, 16
; OPSEL-DAG: %f16_src_hi{{[0-9]+}} = lshr i32 {{.*}}, 16
; OPSEL: %fma_f16 = call half @llvm.fma.f16(
; OPSEL: %f16_merge_lo = or i32
; OPSEL-NOT: unsupported instruction

; DSTHI-LABEL: define amdgpu_kernel void @v_fma_f16_dsthi_kernel(
; DSTHI: %fma_f16 = call half @llvm.fma.f16(
; DSTHI: bitcast half %fma_f16 to i16
; DSTHI: zext i16 {{.*}} to i32
; DSTHI: and i32 {{.*}}, 65535
; DSTHI: shl i32 {{.*}}, 16
; DSTHI: %f16_merge_hi = or i32
; DSTHI-NOT: unsupported instruction

; NEG-LABEL: define amdgpu_kernel void @v_fma_f16_neg_kernel(
; NEG: %neg_f16 = fneg half
; NEG: %abs_f16 = call half @llvm.fabs.f16(half
; NEG: %fma_f16 = call half @llvm.fma.f16(half %neg_f16, half {{.*}}, half %abs_f16)
; NEG-NOT: unsupported instruction

	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	v_fma_f16_basic_kernel
	.p2align	8
	.type	v_fma_f16_basic_kernel,@function
v_fma_f16_basic_kernel:
	v_fma_f16 v0, v1, v2, v3
	s_endpgm

	.globl	v_fma_f16_opsel_kernel
	.p2align	8
	.type	v_fma_f16_opsel_kernel,@function
v_fma_f16_opsel_kernel:
	v_fma_f16 v0, v1, v2, v3 op_sel:[1,0,1,0]
	s_endpgm

	.globl	v_fma_f16_dsthi_kernel
	.p2align	8
	.type	v_fma_f16_dsthi_kernel,@function
v_fma_f16_dsthi_kernel:
	v_fma_f16 v0, v1, v2, v3 op_sel:[0,0,0,1]
	s_endpgm

	.globl	v_fma_f16_neg_kernel
	.p2align	8
	.type	v_fma_f16_neg_kernel,@function
v_fma_f16_neg_kernel:
	v_fma_f16 v0, -v1, v2, |v3|
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_fma_f16_basic_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_accum_offset 4
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel v_fma_f16_opsel_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_accum_offset 4
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel v_fma_f16_dsthi_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_accum_offset 4
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel v_fma_f16_neg_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_accum_offset 4
	.end_amdhsa_kernel
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           v_fma_f16_basic_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     0
    .symbol:         v_fma_f16_basic_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 64
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           v_fma_f16_opsel_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     0
    .symbol:         v_fma_f16_opsel_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 64
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           v_fma_f16_dsthi_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     0
    .symbol:         v_fma_f16_dsthi_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 64
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           v_fma_f16_neg_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     0
    .symbol:         v_fma_f16_neg_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
