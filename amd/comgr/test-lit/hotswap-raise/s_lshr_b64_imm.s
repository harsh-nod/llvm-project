; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=s_lshr_b64_imm_kernel 2>/dev/null | %FileCheck %s
;
; Lift test for the SOP2 64-bit logical right shift (`s_lshr_b64`) with
; an immediate shift count. src0 is the 64-bit value, src1 a single
; 32-bit SGPR shift count (SOP2_64_32). AMDGPU masks the count to the
; low 6 bits, so the handler zext's it to i64 and ANDs with 63 before
; the shift; the mask constant-folds against an immediate src1. All
; CHECK lines sit next to the asm they pin, below.

; CHECK-LABEL: define amdgpu_kernel void @s_lshr_b64_imm_kernel(
; The i64 source operand is carried directly (no by_value split), the
; corpus shape for shifting a kernarg-derived i64 by an immediate.
; CHECK-SAME: ptr addrspace(4) byref([272 x i8]) align 16 %kargs

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	s_lshr_b64_imm_kernel
	.p2align	8
	.type	s_lshr_b64_imm_kernel,@function
s_lshr_b64_imm_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_clause 0x1
	s_load_b32 s2, s[0:1], 0x1c
	s_load_b128 s[4:7], s[0:1], 0x0
	s_wait_xcnt 0x0
	s_bfe_u32 s0, ttmp6, 0x4000c
	s_and_b32 s1, ttmp6, 15
	s_add_co_i32 s0, s0, 1
	s_getreg_b32 s3, hwreg(HW_REG_IB_STS2, 6, 4)
	s_mul_i32 s0, ttmp9, s0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s1, s1, s0
	s_wait_kmcnt 0x0
	s_and_b32 s2, s2, 0xffff
	s_cmp_eq_u32 s3, 0
	s_cselect_b32 s0, ttmp9, s1
	v_mad_u32 v2, s0, s2, v0
	;;#ASMSTART
; In-range immediate 16: 16 & 63 == 16, the mask folds away. `lshr64`
; is the canonical value-name the handler emits (cf. `shl64`/`ashr64`).
; The shift stays 64-bit (a pair-of-i32 regression would read
; `lshr64 = lshr i32`) and the count is masked with `and`, never `urem`.
; CHECK-NOT: lshr64 = lshr i32
; CHECK-NOT: urem
; CHECK: %lshr64 = lshr i64 %{{[^,]+}}, 16
	s_lshr_b64 s[0:1], s[6:7], 16
; 64 & 63 == 0 (without the mask LLVM would emit a poison `lshr .., 64`).
; CHECK: lshr i64 %{{[^,]+}}, 0
	s_lshr_b64 s[0:1], s[6:7], 64
; 65 & 63 == 1. 65 is outside the inline-constant range, so it is
; encoded as a 32-bit literal and folds through the same path.
; CHECK: lshr i64 %{{[^,]+}}, 1
	s_lshr_b64 s[0:1], s[6:7], 65
; 127 & 63 == 63 (upper-bound 7-bit immediate).
; CHECK: lshr i64 %{{[^,]+}}, 63
	s_lshr_b64 s[0:1], s[6:7], 127
	;;#ASMEND
	v_mov_b64_e32 v[0:1], s[0:1]
	global_store_b64 v2, v[0:1], s[4:5] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel s_lshr_b64_imm_kernel
		.amdhsa_kernarg_size 272
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 3
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
  - .args:
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
      - { .offset:         8, .size:           8, .value_kind:     by_value }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 272
    .max_flat_workgroup_size: 1024
    .name:           s_lshr_b64_imm_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         s_lshr_b64_imm_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
