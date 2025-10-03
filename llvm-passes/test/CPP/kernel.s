	.section	__TEXT,__text,regular,pure_instructions
	.globl	_mat4x4_16acc_kernel            ; -- Begin function mat4x4_16acc_kernel
	.p2align	2
_mat4x4_16acc_kernel:                   ; @mat4x4_16acc_kernel
	.cfi_startproc
; %bb.0:                                ; %entry
	mov	x8, #0                          ; =0x0
	movi.2d	v1, #0000000000000000
	movi.2d	v3, #0000000000000000
	mov	w9, #4                          ; =0x4
	movi.2d	v2, #0000000000000000
	movi.2d	v0, #0000000000000000
	ld4.4s	{ v4, v5, v6, v7 }, [x0]
	cbz	x9, LBB0_2
LBB0_1:                                 ; %for.inc
                                        ; =>This Inner Loop Header: Depth=1
	ldp	q16, q17, [x1]
	ldp	q18, q19, [x1, #32]
	fmla.4s	v1, v4, v16[0]
	fmla.4s	v3, v4, v16[1]
	fmla.4s	v2, v4, v16[2]
	fmla.4s	v0, v4, v16[3]
	fmla.4s	v1, v5, v17[0]
	fmla.4s	v3, v5, v17[1]
	fmla.4s	v2, v5, v17[2]
	fmla.4s	v0, v5, v17[3]
	fmla.4s	v1, v6, v18[0]
	fmla.4s	v3, v6, v18[1]
	fmla.4s	v2, v6, v18[2]
	fmla.4s	v0, v6, v18[3]
	fmla.4s	v1, v7, v19[0]
	fmla.4s	v3, v7, v19[1]
	fmla.4s	v2, v7, v19[2]
	fmla.4s	v0, v7, v19[3]
	add	x8, x8, #4
	sub	x9, x9, #4
	cbnz	x9, LBB0_1
LBB0_2:                                 ; %for.end
	mov	s4, v1[1]
	mov	s5, v1[2]
	mov	s6, v1[3]
	stp	s1, s3, [x2]
	mov	s1, v3[1]
	mov	s7, v3[2]
	mov	s3, v3[3]
	stp	s2, s0, [x2, #8]
	mov	s16, v2[1]
	mov	s17, v2[2]
	mov	s2, v2[3]
	stp	s4, s1, [x2, #16]
	mov	s1, v0[1]
	mov	s4, v0[2]
	mov	s0, v0[3]
	stp	s16, s1, [x2, #24]
	stp	s5, s7, [x2, #32]
	stp	s17, s4, [x2, #40]
	stp	s6, s3, [x2, #48]
	stp	s2, s0, [x2, #56]
	ret
	.cfi_endproc
                                        ; -- End function
	.globl	_run_case                       ; -- Begin function run_case
	.p2align	2
_run_case:                              ; @run_case
	.cfi_startproc
; %bb.0:                                ; %entry
	sub	sp, sp, #240
	stp	d9, d8, [sp, #160]              ; 16-byte Folded Spill
	stp	x24, x23, [sp, #176]            ; 16-byte Folded Spill
	stp	x22, x21, [sp, #192]            ; 16-byte Folded Spill
	stp	x20, x19, [sp, #208]            ; 16-byte Folded Spill
	stp	x29, x30, [sp, #224]            ; 16-byte Folded Spill
	add	x29, sp, #224
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	.cfi_offset w23, -56
	.cfi_offset w24, -64
	.cfi_offset b8, -72
	.cfi_offset b9, -80
	mov	x21, x3
	mov	x19, x2
	mov	x20, x1
	mov	x22, x0
	movi.2d	v0, #0000000000000000
	stp	q0, q0, [sp, #128]
	stp	q0, q0, [sp, #96]
	stp	q0, q0, [sp, #32]
	stp	q0, q0, [sp, #64]
	mov	x0, x1
	bl	__ZNKSt3__15arrayIfLm16EE4dataB8se210000Ev
	mov	x23, x0
	mov	x0, x19
	bl	__ZNKSt3__15arrayIfLm16EE4dataB8se210000Ev
	mov	x24, x0
	add	x0, sp, #96
	bl	__ZNSt3__15arrayIfLm16EE4dataB8se210000Ev
	mov	x2, x0
	mov	x0, x23
	mov	x1, x24
	bl	__ZL17mat4x4_ref_triplePKfS0_Pf
	mov	x0, x20
	bl	__ZNKSt3__15arrayIfLm16EE4dataB8se210000Ev
	mov	x23, x0
	mov	x0, x19
	bl	__ZNKSt3__15arrayIfLm16EE4dataB8se210000Ev
	mov	x24, x0
	add	x0, sp, #32
	bl	__ZNSt3__15arrayIfLm16EE4dataB8se210000Ev
	mov	x2, x0
	mov	x0, x23
	mov	x1, x24
	bl	_mat4x4_16acc_kernel
	add	x0, sp, #96
	bl	__ZNSt3__15arrayIfLm16EE4dataB8se210000Ev
	mov	x23, x0
	add	x0, sp, #32
	bl	__ZNSt3__15arrayIfLm16EE4dataB8se210000Ev
	mov	x1, x0
	mov	x0, x23
	bl	__ZL12max_abs_diffPKfS0_
	fmov	s8, s0
	add	x0, sp, #96
	bl	__ZNSt3__15arrayIfLm16EE4dataB8se210000Ev
	mov	x23, x0
	add	x0, sp, #32
	bl	__ZNSt3__15arrayIfLm16EE4dataB8se210000Ev
	mov	x1, x0
	mov	w8, #50604                      ; =0xc5ac
	movk	w8, #14119, lsl #16
	fmov	s0, w8
	mov	x0, x23
	bl	__ZL16nearly_equal_matPKfS0_f
	mov	x23, x0
	fcvt	d0, s8
Lloh0:
	adrp	x8, l_.str.2@PAGE
Lloh1:
	add	x8, x8, l_.str.2@PAGEOFF
Lloh2:
	adrp	x9, l_.str.1@PAGE
Lloh3:
	add	x9, x9, l_.str.1@PAGEOFF
	cmp	w0, #0
	csel	x8, x9, x8, ne
	str	x8, [sp, #16]
	str	d0, [sp, #8]
	str	x22, [sp]
Lloh4:
	adrp	x0, l_.str@PAGE
Lloh5:
	add	x0, x0, l_.str@PAGEOFF
	bl	_printf
	tbnz	w21, #0, LBB1_2
; %bb.1:                                ; %entry
	cbnz	w23, LBB1_3
LBB1_2:                                 ; %if.then
	mov	x0, x20
	bl	__ZNKSt3__15arrayIfLm16EE4dataB8se210000Ev
	mov	x1, x0
Lloh6:
	adrp	x0, l_.str.3@PAGE
Lloh7:
	add	x0, x0, l_.str.3@PAGEOFF
	bl	__ZL9print_matPKcPKf
	mov	x0, x19
	bl	__ZNKSt3__15arrayIfLm16EE4dataB8se210000Ev
	mov	x1, x0
Lloh8:
	adrp	x0, l_.str.4@PAGE
Lloh9:
	add	x0, x0, l_.str.4@PAGEOFF
	bl	__ZL9print_matPKcPKf
	add	x0, sp, #96
	bl	__ZNSt3__15arrayIfLm16EE4dataB8se210000Ev
	mov	x1, x0
Lloh10:
	adrp	x0, l_.str.5@PAGE
Lloh11:
	add	x0, x0, l_.str.5@PAGEOFF
	bl	__ZL9print_matPKcPKf
	add	x0, sp, #32
	bl	__ZNSt3__15arrayIfLm16EE4dataB8se210000Ev
	mov	x1, x0
Lloh12:
	adrp	x0, l_.str.6@PAGE
Lloh13:
	add	x0, x0, l_.str.6@PAGEOFF
	bl	__ZL9print_matPKcPKf
LBB1_3:                                 ; %if.end
	tbz	w23, #0, LBB1_5
; %bb.4:                                ; %if.end22
	ldp	x29, x30, [sp, #224]            ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #208]            ; 16-byte Folded Reload
	ldp	x22, x21, [sp, #192]            ; 16-byte Folded Reload
	ldp	x24, x23, [sp, #176]            ; 16-byte Folded Reload
	ldp	d9, d8, [sp, #160]              ; 16-byte Folded Reload
	add	sp, sp, #240
	ret
LBB1_5:                                 ; %if.then21
	mov	w0, #1                          ; =0x1
	bl	_exit
	.loh AdrpAdd	Lloh4, Lloh5
	.loh AdrpAdd	Lloh2, Lloh3
	.loh AdrpAdd	Lloh0, Lloh1
	.loh AdrpAdd	Lloh12, Lloh13
	.loh AdrpAdd	Lloh10, Lloh11
	.loh AdrpAdd	Lloh8, Lloh9
	.loh AdrpAdd	Lloh6, Lloh7
	.cfi_endproc
                                        ; -- End function
	.p2align	2                               ; -- Begin function _ZL17mat4x4_ref_triplePKfS0_Pf
__ZL17mat4x4_ref_triplePKfS0_Pf:        ; @_ZL17mat4x4_ref_triplePKfS0_Pf
	.cfi_startproc
; %bb.0:                                ; %entry
	mov	x8, #0                          ; =0x0
	b	LBB2_2
LBB2_1:                                 ; %for.inc20
                                        ;   in Loop: Header=BB2_2 Depth=1
	add	x8, x8, #1
	add	x0, x0, #16
LBB2_2:                                 ; %for.cond
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB2_5 Depth 2
                                        ;       Child Loop BB2_7 Depth 3
	cmp	x8, #4
	b.eq	LBB2_8
; %bb.3:                                ; %for.cond1.preheader
                                        ;   in Loop: Header=BB2_2 Depth=1
	mov	x9, #0                          ; =0x0
	lsl	x10, x8, #4
	mov	x11, x1
	b	LBB2_5
LBB2_4:                                 ; %for.end
                                        ;   in Loop: Header=BB2_5 Depth=2
	orr	x12, x10, x9, lsl #2
	str	s0, [x2, x12]
	add	x9, x9, #1
	add	x11, x11, #4
LBB2_5:                                 ; %for.cond1
                                        ;   Parent Loop BB2_2 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB2_7 Depth 3
	cmp	x9, #4
	b.eq	LBB2_1
; %bb.6:                                ; %for.cond4.preheader
                                        ;   in Loop: Header=BB2_5 Depth=2
	mov	x12, #0                         ; =0x0
	movi	d0, #0000000000000000
	cmp	x12, #16
	b.eq	LBB2_4
LBB2_7:                                 ; %for.inc
                                        ;   Parent Loop BB2_2 Depth=1
                                        ;     Parent Loop BB2_5 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	ldr	s1, [x0, x12]
	ldr	s2, [x11, x12, lsl #2]
	fmul	s1, s1, s2
	fadd	s0, s0, s1
	add	x12, x12, #4
	cmp	x12, #16
	b.ne	LBB2_7
	b	LBB2_4
LBB2_8:                                 ; %for.end22
	ret
	.cfi_endproc
                                        ; -- End function
	.private_extern	__ZNKSt3__15arrayIfLm16EE4dataB8se210000Ev ; -- Begin function _ZNKSt3__15arrayIfLm16EE4dataB8se210000Ev
	.globl	__ZNKSt3__15arrayIfLm16EE4dataB8se210000Ev
	.weak_definition	__ZNKSt3__15arrayIfLm16EE4dataB8se210000Ev
	.p2align	2
__ZNKSt3__15arrayIfLm16EE4dataB8se210000Ev: ; @_ZNKSt3__15arrayIfLm16EE4dataB8se210000Ev
	.cfi_startproc
; %bb.0:                                ; %entry
	ret
	.cfi_endproc
                                        ; -- End function
	.private_extern	__ZNSt3__15arrayIfLm16EE4dataB8se210000Ev ; -- Begin function _ZNSt3__15arrayIfLm16EE4dataB8se210000Ev
	.globl	__ZNSt3__15arrayIfLm16EE4dataB8se210000Ev
	.weak_definition	__ZNSt3__15arrayIfLm16EE4dataB8se210000Ev
	.p2align	2
__ZNSt3__15arrayIfLm16EE4dataB8se210000Ev: ; @_ZNSt3__15arrayIfLm16EE4dataB8se210000Ev
	.cfi_startproc
; %bb.0:                                ; %entry
	ret
	.cfi_endproc
                                        ; -- End function
	.p2align	2                               ; -- Begin function _ZL12max_abs_diffPKfS0_
__ZL12max_abs_diffPKfS0_:               ; @_ZL12max_abs_diffPKfS0_
	.cfi_startproc
; %bb.0:                                ; %entry
	sub	sp, sp, #64
	stp	x22, x21, [sp, #16]             ; 16-byte Folded Spill
	stp	x20, x19, [sp, #32]             ; 16-byte Folded Spill
	stp	x29, x30, [sp, #48]             ; 16-byte Folded Spill
	add	x29, sp, #48
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	mov	x19, x1
	mov	x20, x0
	mov	x21, #0                         ; =0x0
	str	wzr, [sp, #12]
	cmp	x21, #64
	b.eq	LBB5_2
LBB5_1:                                 ; %for.body
                                        ; =>This Inner Loop Header: Depth=1
	ldr	s0, [x20, x21]
	ldr	s1, [x19, x21]
	fsub	s0, s0, s1
	bl	__ZNSt3__16__math4fabsB8se210000Ef
	str	s0, [sp, #8]
	add	x0, sp, #12
	add	x1, sp, #8
	bl	__ZNSt3__13maxB8se210000IfEERKT_S3_S3_
	ldr	s0, [x0]
	str	s0, [sp, #12]
	add	x21, x21, #4
	cmp	x21, #64
	b.ne	LBB5_1
LBB5_2:                                 ; %for.end
	ldr	s0, [sp, #12]
	ldp	x29, x30, [sp, #48]             ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #32]             ; 16-byte Folded Reload
	ldp	x22, x21, [sp, #16]             ; 16-byte Folded Reload
	add	sp, sp, #64
	ret
	.cfi_endproc
                                        ; -- End function
	.p2align	2                               ; -- Begin function _ZL16nearly_equal_matPKfS0_f
__ZL16nearly_equal_matPKfS0_f:          ; @_ZL16nearly_equal_matPKfS0_f
	.cfi_startproc
; %bb.0:                                ; %entry
	stp	d9, d8, [sp, #-32]!             ; 16-byte Folded Spill
	stp	x29, x30, [sp, #16]             ; 16-byte Folded Spill
	add	x29, sp, #16
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset b8, -24
	.cfi_offset b9, -32
	fmov	s8, s0
	bl	__ZL12max_abs_diffPKfS0_
	fcmp	s0, s8
	cset	w0, ls
	ldp	x29, x30, [sp, #16]             ; 16-byte Folded Reload
	ldp	d9, d8, [sp], #32               ; 16-byte Folded Reload
	ret
	.cfi_endproc
                                        ; -- End function
	.p2align	2                               ; -- Begin function _ZL9print_matPKcPKf
__ZL9print_matPKcPKf:                   ; @_ZL9print_matPKcPKf
	.cfi_startproc
; %bb.0:                                ; %entry
	sub	sp, sp, #80
	stp	x24, x23, [sp, #16]             ; 16-byte Folded Spill
	stp	x22, x21, [sp, #32]             ; 16-byte Folded Spill
	stp	x20, x19, [sp, #48]             ; 16-byte Folded Spill
	stp	x29, x30, [sp, #64]             ; 16-byte Folded Spill
	add	x29, sp, #64
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	.cfi_offset w23, -56
	.cfi_offset w24, -64
	mov	x19, x1
	str	x0, [sp]
Lloh14:
	adrp	x0, l_.str.7@PAGE
Lloh15:
	add	x0, x0, l_.str.7@PAGEOFF
	bl	_printf
	mov	x22, #0                         ; =0x0
Lloh16:
	adrp	x20, l_.str.8@PAGE
Lloh17:
	add	x20, x20, l_.str.8@PAGEOFF
Lloh18:
	adrp	x21, l_.str.9@PAGE
Lloh19:
	add	x21, x21, l_.str.9@PAGEOFF
	b	LBB7_2
LBB7_1:                                 ; %for.end
                                        ;   in Loop: Header=BB7_2 Depth=1
	mov	w0, #10                         ; =0xa
	bl	_putchar
	add	x22, x22, #1
	add	x19, x19, #16
LBB7_2:                                 ; %for.cond
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB7_4 Depth 2
	cmp	x22, #4
	b.eq	LBB7_5
; %bb.3:                                ; %for.body
                                        ;   in Loop: Header=BB7_2 Depth=1
	mov	x0, x20
	bl	_printf
	mov	x23, #0                         ; =0x0
	cmp	x23, #16
	b.eq	LBB7_1
LBB7_4:                                 ; %for.body4
                                        ;   Parent Loop BB7_2 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	s0, [x19, x23]
	fcvt	d0, s0
	str	d0, [sp]
	mov	x0, x21
	bl	_printf
	add	x23, x23, #4
	cmp	x23, #16
	b.ne	LBB7_4
	b	LBB7_1
LBB7_5:                                 ; %for.end9
	ldp	x29, x30, [sp, #64]             ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #48]             ; 16-byte Folded Reload
	ldp	x22, x21, [sp, #32]             ; 16-byte Folded Reload
	ldp	x24, x23, [sp, #16]             ; 16-byte Folded Reload
	add	sp, sp, #80
	ret
	.loh AdrpAdd	Lloh18, Lloh19
	.loh AdrpAdd	Lloh16, Lloh17
	.loh AdrpAdd	Lloh14, Lloh15
	.cfi_endproc
                                        ; -- End function
	.private_extern	__ZNSt3__13maxB8se210000IfEERKT_S3_S3_ ; -- Begin function _ZNSt3__13maxB8se210000IfEERKT_S3_S3_
	.globl	__ZNSt3__13maxB8se210000IfEERKT_S3_S3_
	.weak_definition	__ZNSt3__13maxB8se210000IfEERKT_S3_S3_
	.p2align	2
__ZNSt3__13maxB8se210000IfEERKT_S3_S3_: ; @_ZNSt3__13maxB8se210000IfEERKT_S3_S3_
	.cfi_startproc
; %bb.0:                                ; %entry
	stp	x29, x30, [sp, #-16]!           ; 16-byte Folded Spill
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	bl	__ZNSt3__13maxB8se210000IfNS_6__lessIvvEEEERKT_S5_S5_T0_
	ldp	x29, x30, [sp], #16             ; 16-byte Folded Reload
	ret
	.cfi_endproc
                                        ; -- End function
	.private_extern	__ZNSt3__16__math4fabsB8se210000Ef ; -- Begin function _ZNSt3__16__math4fabsB8se210000Ef
	.globl	__ZNSt3__16__math4fabsB8se210000Ef
	.weak_definition	__ZNSt3__16__math4fabsB8se210000Ef
	.p2align	2
__ZNSt3__16__math4fabsB8se210000Ef:     ; @_ZNSt3__16__math4fabsB8se210000Ef
	.cfi_startproc
; %bb.0:                                ; %entry
	fabs	s0, s0
	ret
	.cfi_endproc
                                        ; -- End function
	.private_extern	__ZNSt3__13maxB8se210000IfNS_6__lessIvvEEEERKT_S5_S5_T0_ ; -- Begin function _ZNSt3__13maxB8se210000IfNS_6__lessIvvEEEERKT_S5_S5_T0_
	.globl	__ZNSt3__13maxB8se210000IfNS_6__lessIvvEEEERKT_S5_S5_T0_
	.weak_definition	__ZNSt3__13maxB8se210000IfNS_6__lessIvvEEEERKT_S5_S5_T0_
	.p2align	2
__ZNSt3__13maxB8se210000IfNS_6__lessIvvEEEERKT_S5_S5_T0_: ; @_ZNSt3__13maxB8se210000IfNS_6__lessIvvEEEERKT_S5_S5_T0_
	.cfi_startproc
; %bb.0:                                ; %entry
	sub	sp, sp, #48
	stp	x20, x19, [sp, #16]             ; 16-byte Folded Spill
	stp	x29, x30, [sp, #32]             ; 16-byte Folded Spill
	add	x29, sp, #32
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	mov	x19, x1
	mov	x20, x0
	add	x0, sp, #15
	mov	x1, x20
	mov	x2, x19
	bl	__ZNKSt3__16__lessIvvEclB8se210000IffEEbRKT_RKT0_
	cmp	w0, #0
	csel	x0, x19, x20, ne
	ldp	x29, x30, [sp, #32]             ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #16]             ; 16-byte Folded Reload
	add	sp, sp, #48
	ret
	.cfi_endproc
                                        ; -- End function
	.private_extern	__ZNKSt3__16__lessIvvEclB8se210000IffEEbRKT_RKT0_ ; -- Begin function _ZNKSt3__16__lessIvvEclB8se210000IffEEbRKT_RKT0_
	.globl	__ZNKSt3__16__lessIvvEclB8se210000IffEEbRKT_RKT0_
	.weak_definition	__ZNKSt3__16__lessIvvEclB8se210000IffEEbRKT_RKT0_
	.p2align	2
__ZNKSt3__16__lessIvvEclB8se210000IffEEbRKT_RKT0_: ; @_ZNKSt3__16__lessIvvEclB8se210000IffEEbRKT_RKT0_
	.cfi_startproc
; %bb.0:                                ; %entry
	ldr	s0, [x1]
	ldr	s1, [x2]
	fcmp	s0, s1
	cset	w0, mi
	ret
	.cfi_endproc
                                        ; -- End function
	.section	__TEXT,__cstring,cstring_literals
l_.str:                                 ; @.str
	.asciz	"[%s] max|diff| = %.8f  => %s\n"

l_.str.1:                               ; @.str.1
	.asciz	"OK"

l_.str.2:                               ; @.str.2
	.asciz	"MISMATCH"

l_.str.3:                               ; @.str.3
	.asciz	"A"

l_.str.4:                               ; @.str.4
	.asciz	"B"

l_.str.5:                               ; @.str.5
	.asciz	"C_ref"

l_.str.6:                               ; @.str.6
	.asciz	"C_ker"

l_.str.7:                               ; @.str.7
	.asciz	"%s =\n"

l_.str.8:                               ; @.str.8
	.asciz	"  "

l_.str.9:                               ; @.str.9
	.asciz	"%8.4f "

l_.str.10:                              ; @.str.10
	.asciz	"\n"

.subsections_via_symbols
