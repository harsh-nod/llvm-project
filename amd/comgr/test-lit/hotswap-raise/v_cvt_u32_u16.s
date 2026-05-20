; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=v_cvt_u32_u16_lo_kernel 2>/dev/null | %FileCheck %s --check-prefix=LO
; RUN: %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=v_cvt_u32_u16_hi_kernel 2>/dev/null | %FileCheck %s --check-prefix=HI
; RUN: %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=v_cvt_u32_u16_e64_hi_kernel 2>/dev/null | %FileCheck %s --check-prefix=E64HI
; RUN: %raise_cli %t.hsaco --target-isa=gfx942 --write-hsaco=%t.out --kernel=v_cvt_u32_u16_lo_kernel 2>&1 | %FileCheck %s --check-prefix=PIPE
;
; gfx11+ true16 u16 -> u32 zero-extend. The source half is selected by
; src0's op_sel bit in one of two places, depending on encoding form:
;   * `_e32` (the form llvm-mc emits for plain `v_cvt_u32_u16 v, vN.h`):
;     the MCInst src0 slot holds a `_LO16` / `_HI16` subreg of the parent
;     VGPR. There is no src0_modifiers operand.
;   * `_e64` (forced by the explicit `_e64` mnemonic): the disassembler
;     prints `v0, v1.h op_sel:[1,0]` -- both the subreg and the
;     `src0_modifiers[OP_SEL_0]` bit are set. This kernel pins the
;     modifier-decode path (handler must accept OP_SEL_0 in its allowed
;     modifier mask and OR it with the subreg signal).
;
; The handler in handle-valu-small-ops.cpp lifts each form as
;   half = trunc i32 [src0 >> (src0_hi ? 16 : 0)] to i16
;   dst  = zext i16 half to i32
; The `CHECK-NOT: sext` lines pin the unsigned (zero-extend) conversion
; contract -- a future drift to `sext` would silently sign-extend i16
; sources with the MSB set.

; LO-LABEL: define amdgpu_kernel void @v_cvt_u32_u16_lo_kernel(
; LO: trunc i32 {{.*}} to i16
; LO: %cvt_u32_u16{{.*}} = zext i16 {{.*}} to i32
; LO-NOT: lshr i32 {{.*}}, 16
; LO-NOT: sext

; HI-LABEL: define amdgpu_kernel void @v_cvt_u32_u16_hi_kernel(
; HI: lshr i32 {{.*}}, 16
; HI: trunc i32 {{.*}} to i16
; HI: %cvt_u32_u16{{.*}} = zext i16 {{.*}} to i32
; HI-NOT: sext

; E64HI-LABEL: define amdgpu_kernel void @v_cvt_u32_u16_e64_hi_kernel(
; E64HI: lshr i32 {{.*}}, 16
; E64HI: trunc i32 {{.*}} to i16
; E64HI: %cvt_u32_u16{{.*}} = zext i16 {{.*}} to i32
; E64HI-NOT: sext
; E64HI-NOT: unsupported source modifiers

; PIPE: raise_cli: wrote
; PIPE-SAME: v_cvt_u32_u16_lo_kernel

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	v_cvt_u32_u16_lo_kernel
	.p2align	8
	.type	v_cvt_u32_u16_lo_kernel,@function
v_cvt_u32_u16_lo_kernel:
	v_cvt_u32_u16 v0, v1.l
	s_endpgm

	.globl	v_cvt_u32_u16_hi_kernel
	.p2align	8
	.type	v_cvt_u32_u16_hi_kernel,@function
v_cvt_u32_u16_hi_kernel:
	v_cvt_u32_u16 v0, v1.h
	s_endpgm

	.globl	v_cvt_u32_u16_e64_hi_kernel
	.p2align	8
	.type	v_cvt_u32_u16_e64_hi_kernel,@function
v_cvt_u32_u16_e64_hi_kernel:
	v_cvt_u32_u16_e64 v0, v1.h
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_cvt_u32_u16_lo_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 8
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel v_cvt_u32_u16_hi_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 8
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel v_cvt_u32_u16_e64_hi_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 8
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           v_cvt_u32_u16_lo_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_cvt_u32_u16_lo_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           v_cvt_u32_u16_hi_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_cvt_u32_u16_hi_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           v_cvt_u32_u16_e64_hi_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_cvt_u32_u16_e64_hi_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
