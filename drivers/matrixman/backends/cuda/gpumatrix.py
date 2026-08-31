#!/usr/bin/env python3
"""Reusable legacy-CUDA execution helpers and a small standalone diagnostic.

The execution path uses only the CUDA Driver API through ``ctypes`` and an
embedded PTX module targeted at ``sm_21``.  It does not require ``nvcc`` or a
CUDA Toolkit at runtime, and it does not fall back to CPU arithmetic.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import sys

import numpy as np


CUresult = ctypes.c_int
CUdevice = ctypes.c_int
CUdeviceptr = ctypes.c_uint64
CUcontext = ctypes.c_void_p
CUmodule = ctypes.c_void_p
CUfunction = ctypes.c_void_p

CUDA_SUCCESS = 0
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76
CUDA_BLOCK_SIZE = 128


PTX = r"""
.version 3.0
.target sm_21
.address_size 64

.visible .entry matrix_add(
    .param .u64 p_a,
    .param .u64 p_b,
    .param .u64 p_out,
    .param .u32 p_count
)
{
    .reg .pred %p<1>;
    .reg .u32 %r<5>;
    .reg .u64 %rd<5>;
    .reg .f32 %f<4>;
    ld.param.u64 %rd1, [p_a];
    ld.param.u64 %rd2, [p_b];
    ld.param.u64 %rd3, [p_out];
    ld.param.u32 %r1, [p_count];
    // Legacy PTX requires special registers to be read with mov/cvt first.
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mul.lo.u32 %r2, %r2, %r3;
    mov.u32 %r4, %tid.x;
    add.u32 %r2, %r2, %r4;
    setp.ge.u32 %p0, %r2, %r1;
    @%p0 bra DONE;
    mul.wide.u32 %rd4, %r2, 4;
    add.u64 %rd1, %rd1, %rd4;
    add.u64 %rd2, %rd2, %rd4;
    add.u64 %rd3, %rd3, %rd4;
    ld.global.f32 %f1, [%rd1];
    ld.global.f32 %f2, [%rd2];
    add.f32 %f3, %f1, %f2;
    st.global.f32 [%rd3], %f3;
DONE:
    ret;
}

.visible .entry matrix_mul(
    .param .u64 p_a,
    .param .u64 p_b,
    .param .u64 p_out,
    .param .u32 p_m,
    .param .u32 p_k,
    .param .u32 p_n
)
{
    .reg .pred %p<2>;
    .reg .u32 %r<14>;
    .reg .u64 %rd<8>;
    .reg .f32 %f<5>;
    ld.param.u64 %rd1, [p_a];
    ld.param.u64 %rd2, [p_b];
    ld.param.u64 %rd3, [p_out];
    ld.param.u32 %r1, [p_m];
    ld.param.u32 %r2, [p_k];
    ld.param.u32 %r3, [p_n];
    mov.u32 %r4, %ctaid.x;
    mov.u32 %r11, %ntid.x;
    mul.lo.u32 %r4, %r4, %r11;
    mov.u32 %r12, %tid.x;
    add.u32 %r4, %r4, %r12;
    mul.lo.u32 %r5, %r1, %r3;
    setp.ge.u32 %p0, %r4, %r5;
    @%p0 bra DONE;
    div.u32 %r6, %r4, %r3;
    rem.u32 %r7, %r4, %r3;
    mov.u32 %r8, 0;
    mov.f32 %f1, 0.0;
LOOP:
    setp.ge.u32 %p1, %r8, %r2;
    @%p1 bra STORE;
    mul.lo.u32 %r9, %r6, %r2;
    add.u32 %r9, %r9, %r8;
    mul.wide.u32 %rd4, %r9, 4;
    add.u64 %rd5, %rd1, %rd4;
    ld.global.f32 %f2, [%rd5];
    mul.lo.u32 %r10, %r8, %r3;
    add.u32 %r10, %r10, %r7;
    mul.wide.u32 %rd6, %r10, 4;
    add.u64 %rd7, %rd2, %rd6;
    ld.global.f32 %f3, [%rd7];
    mul.f32 %f4, %f2, %f3;
    add.f32 %f1, %f1, %f4;
    add.u32 %r8, %r8, 1;
    bra LOOP;
STORE:
    mul.wide.u32 %rd4, %r4, 4;
    add.u64 %rd5, %rd3, %rd4;
    st.global.f32 [%rd5], %f1;
DONE:
    ret;
}

.visible .entry conv2d_nchw(
    .param .u64 p_input,
    .param .u64 p_weight,
    .param .u64 p_bias,
    .param .u64 p_output,
    .param .u32 p_n,
    .param .u32 p_c,
    .param .u32 p_h,
    .param .u32 p_w,
    .param .u32 p_k,
    .param .u32 p_r,
    .param .u32 p_s,
    .param .u32 p_out_h,
    .param .u32 p_out_w,
    .param .u32 p_stride_h,
    .param .u32 p_stride_w,
    .param .u32 p_pad_h,
    .param .u32 p_pad_w,
    .param .u32 p_dil_h,
    .param .u32 p_dil_w,
    .param .u32 p_groups
)
{
    .reg .pred %p<5>;
    .reg .u32 %r<36>;
    .reg .s32 %s<8>;
    .reg .u64 %rd<8>;
    .reg .f32 %f<4>;
    ld.param.u64 %rd1, [p_input];
    ld.param.u64 %rd2, [p_weight];
    ld.param.u64 %rd3, [p_bias];
    ld.param.u64 %rd4, [p_output];
    ld.param.u32 %r1, [p_n];
    ld.param.u32 %r2, [p_c];
    ld.param.u32 %r3, [p_h];
    ld.param.u32 %r4, [p_w];
    ld.param.u32 %r5, [p_k];
    ld.param.u32 %r6, [p_r];
    ld.param.u32 %r7, [p_s];
    ld.param.u32 %r8, [p_out_h];
    ld.param.u32 %r9, [p_out_w];
    ld.param.u32 %r10, [p_stride_h];
    ld.param.u32 %r11, [p_stride_w];
    ld.param.u32 %r12, [p_pad_h];
    ld.param.u32 %r13, [p_pad_w];
    ld.param.u32 %r14, [p_dil_h];
    ld.param.u32 %r15, [p_dil_w];
    ld.param.u32 %r16, [p_groups];
    cvt.s32.u32 %s6, %r3;
    cvt.s32.u32 %s7, %r4;

    // Legacy NVIDIA 390 JIT requires explicit special-register reads.
    mov.u32 %r17, %ctaid.x;
    mov.u32 %r18, %ntid.x;
    mul.lo.u32 %r17, %r17, %r18;
    mov.u32 %r19, %tid.x;
    add.u32 %r17, %r17, %r19;

    mul.lo.u32 %r18, %r5, %r8;
    mul.lo.u32 %r18, %r18, %r9;
    mul.lo.u32 %r18, %r18, %r1;
    setp.ge.u32 %p0, %r17, %r18;
    @%p0 bra CONV_DONE;
    mul.lo.u32 %r19, %r5, %r8;
    mul.lo.u32 %r19, %r19, %r9;
    div.u32 %r20, %r17, %r19;
    rem.u32 %r21, %r17, %r19;
    mul.lo.u32 %r22, %r8, %r9;
    div.u32 %r23, %r21, %r22;
    rem.u32 %r24, %r21, %r22;
    div.u32 %r25, %r24, %r9;
    rem.u32 %r26, %r24, %r9;

    div.u32 %r27, %r5, %r16;
    div.u32 %r28, %r2, %r16;
    div.u32 %r29, %r23, %r27;
    rem.u32 %r30, %r23, %r27;
    mul.lo.u32 %r31, %r29, %r28;

    setp.eq.u64 %p3, %rd3, 0;
    @%p3 bra CONV_NO_BIAS;
    mul.wide.u32 %rd5, %r23, 4;
    add.u64 %rd6, %rd3, %rd5;
    ld.global.f32 %f0, [%rd6];
    bra CONV_BIAS_DONE;
CONV_NO_BIAS:
    mov.f32 %f0, 0.0;
CONV_BIAS_DONE:
    mov.u32 %r32, 0;
    mov.u32 %r33, 0;
    mov.u32 %r34, 0;
CONV_IC:
    setp.ge.u32 %p0, %r32, %r28;
    @%p0 bra CONV_STORE;
CONV_KH:
    setp.ge.u32 %p0, %r33, %r6;
    @%p0 bra CONV_NEXT_IC;
CONV_KW:
    setp.ge.u32 %p0, %r34, %r7;
    @%p0 bra CONV_NEXT_KH;

    cvt.s32.u32 %s0, %r25;
    cvt.s32.u32 %s1, %r26;
    cvt.s32.u32 %s2, %r10;
    cvt.s32.u32 %s3, %r11;
    mul.lo.s32 %s0, %s0, %s2;
    mul.lo.s32 %s1, %s1, %s3;
    cvt.s32.u32 %s2, %r12;
    cvt.s32.u32 %s3, %r13;
    sub.s32 %s0, %s0, %s2;
    sub.s32 %s1, %s1, %s3;
    cvt.s32.u32 %s2, %r33;
    cvt.s32.u32 %s3, %r34;
    cvt.s32.u32 %s4, %r14;
    cvt.s32.u32 %s5, %r15;
    mul.lo.s32 %s2, %s2, %s4;
    mul.lo.s32 %s3, %s3, %s5;
    add.s32 %s0, %s0, %s2;
    add.s32 %s1, %s1, %s3;
    setp.ge.s32 %p1, %s0, 0;
    setp.lt.s32 %p2, %s0, %s6;
    and.pred %p1, %p1, %p2;
    setp.ge.s32 %p2, %s1, 0;
    setp.lt.s32 %p4, %s1, %s7;
    and.pred %p2, %p2, %p4;
    @!%p1 bra CONV_SKIP;
    @!%p2 bra CONV_SKIP;

    add.u32 %r35, %r31, %r32;
    mul.lo.u32 %r35, %r20, %r2;
    add.u32 %r35, %r35, %r31;
    add.u32 %r35, %r35, %r32;
    mul.lo.u32 %r35, %r35, %r3;
    cvt.u32.s32 %r18, %s0;
    add.u32 %r35, %r35, %r18;
    mul.lo.u32 %r35, %r35, %r4;
    cvt.u32.s32 %r18, %s1;
    add.u32 %r35, %r35, %r18;
    mul.wide.u32 %rd5, %r35, 4;
    add.u64 %rd6, %rd1, %rd5;
    ld.global.f32 %f1, [%rd6];

    mul.lo.u32 %r35, %r30, %r28;
    add.u32 %r35, %r35, %r32;
    mul.lo.u32 %r35, %r35, %r6;
    add.u32 %r35, %r35, %r33;
    mul.lo.u32 %r35, %r35, %r7;
    add.u32 %r35, %r35, %r34;
    mul.wide.u32 %rd5, %r35, 4;
    add.u64 %rd6, %rd2, %rd5;
    ld.global.f32 %f2, [%rd6];
    mul.f32 %f3, %f1, %f2;
    add.f32 %f0, %f0, %f3;
CONV_SKIP:
    add.u32 %r34, %r34, 1;
    bra CONV_KW;
CONV_NEXT_KH:
    mov.u32 %r34, 0;
    add.u32 %r33, %r33, 1;
    bra CONV_KH;
CONV_NEXT_IC:
    mov.u32 %r33, 0;
    add.u32 %r32, %r32, 1;
    bra CONV_IC;
CONV_STORE:
    mul.lo.u32 %r35, %r20, %r5;
    add.u32 %r35, %r35, %r23;
    mul.lo.u32 %r35, %r35, %r8;
    add.u32 %r35, %r35, %r25;
    mul.lo.u32 %r35, %r35, %r9;
    add.u32 %r35, %r35, %r26;
    mul.wide.u32 %rd5, %r35, 4;
    add.u64 %rd6, %rd4, %rd5;
    st.global.f32 [%rd6], %f0;
CONV_DONE:
    ret;
}

.visible .entry batch_norm_inference(
    .param .u64 p_input,
    .param .u64 p_running_mean,
    .param .u64 p_running_var,
    .param .u64 p_weight,
    .param .u64 p_bias,
    .param .u64 p_output,
    .param .u32 p_count,
    .param .u32 p_channels,
    .param .u32 p_spatial,
    .param .f32 p_eps
)
{
    .reg .pred %p<1>;
    .reg .u32 %r<12>;
    .reg .u64 %rd<8>;
    .reg .f32 %f<8>;
    ld.param.u64 %rd1, [p_input];
    ld.param.u64 %rd2, [p_running_mean];
    ld.param.u64 %rd3, [p_running_var];
    ld.param.u64 %rd4, [p_weight];
    ld.param.u64 %rd5, [p_bias];
    ld.param.u64 %rd6, [p_output];
    ld.param.u32 %r1, [p_count];
    ld.param.u32 %r2, [p_channels];
    ld.param.u32 %r3, [p_spatial];
    ld.param.f32 %f1, [p_eps];

    // Keep special-register reads explicit for the legacy NVIDIA 390 JIT.
    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %ntid.x;
    mul.lo.u32 %r4, %r4, %r5;
    mov.u32 %r6, %tid.x;
    add.u32 %r4, %r4, %r6;
    setp.ge.u32 %p0, %r4, %r1;
    @%p0 bra BN_DONE;

    div.u32 %r8, %r4, %r3;
    rem.u32 %r8, %r8, %r2;
    mul.wide.u32 %rd7, %r8, 4;
    add.u64 %rd7, %rd2, %rd7;
    ld.global.f32 %f2, [%rd7];
    mul.wide.u32 %rd7, %r8, 4;
    add.u64 %rd7, %rd3, %rd7;
    ld.global.f32 %f3, [%rd7];
    add.f32 %f3, %f3, %f1;
    sqrt.approx.f32 %f4, %f3;
    mul.wide.u32 %rd7, %r4, 4;
    add.u64 %rd7, %rd1, %rd7;
    ld.global.f32 %f5, [%rd7];
    sub.f32 %f5, %f5, %f2;
    div.approx.f32 %f5, %f5, %f4;
    mul.wide.u32 %rd7, %r8, 4;
    add.u64 %rd7, %rd4, %rd7;
    ld.global.f32 %f6, [%rd7];
    mul.f32 %f5, %f5, %f6;
    mul.wide.u32 %rd7, %r8, 4;
    add.u64 %rd7, %rd5, %rd7;
    ld.global.f32 %f6, [%rd7];
    add.f32 %f5, %f5, %f6;
    mul.wide.u32 %rd7, %r4, 4;
    add.u64 %rd7, %rd6, %rd7;
    st.global.f32 [%rd7], %f5;
BN_DONE:
    ret;
}

.visible .entry silu(
    .param .u64 p_input,
    .param .u64 p_output,
    .param .u32 p_count
)
{
    .reg .pred %p<1>;
    .reg .u32 %r<6>;
    .reg .u64 %rd<4>;
    .reg .f32 %f<6>;
    ld.param.u64 %rd1, [p_input];
    ld.param.u64 %rd2, [p_output];
    ld.param.u32 %r1, [p_count];
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mul.lo.u32 %r2, %r2, %r3;
    mov.u32 %r4, %tid.x;
    add.u32 %r2, %r2, %r4;
    setp.ge.u32 %p0, %r2, %r1;
    @%p0 bra SILU_DONE;
    mul.wide.u32 %rd3, %r2, 4;
    add.u64 %rd1, %rd1, %rd3;
    add.u64 %rd2, %rd2, %rd3;
    ld.global.f32 %f1, [%rd1];
    neg.f32 %f2, %f1;
    // exp(-x) = 2^(-x * log2(e)); both instructions are sm_21-compatible.
    mov.f32 %f3, 1.4426950409;
    mul.f32 %f2, %f2, %f3;
    ex2.approx.f32 %f2, %f2;
    mov.f32 %f3, 1.0;
    add.f32 %f2, %f2, %f3;
    div.approx.f32 %f2, %f1, %f2;
    st.global.f32 [%rd2], %f2;
SILU_DONE:
    ret;
}

.visible .entry split_copy(
    .param .u64 p_input,
    .param .u64 p_output,
    .param .u32 p_count,
    .param .u32 p_dim,
    .param .u32 p_offset,
    .param .u32 p_in0,
    .param .u32 p_in1,
    .param .u32 p_in2,
    .param .u32 p_in3,
    .param .u32 p_out0,
    .param .u32 p_out1,
    .param .u32 p_out2,
    .param .u32 p_out3
)
{
    .reg .pred %p<5>;
    .reg .u32 %r<24>;
    .reg .u64 %rd<4>;
    .reg .f32 %f<2>;
    ld.param.u64 %rd1, [p_input];
    ld.param.u64 %rd2, [p_output];
    ld.param.u32 %r1, [p_count];
    ld.param.u32 %r2, [p_dim];
    ld.param.u32 %r3, [p_offset];
    ld.param.u32 %r4, [p_in0];
    ld.param.u32 %r5, [p_in1];
    ld.param.u32 %r6, [p_in2];
    ld.param.u32 %r7, [p_in3];
    ld.param.u32 %r8, [p_out0];
    ld.param.u32 %r9, [p_out1];
    ld.param.u32 %r10, [p_out2];
    ld.param.u32 %r11, [p_out3];
    mov.u32 %r12, %ctaid.x;
    mov.u32 %r13, %ntid.x;
    mul.lo.u32 %r12, %r12, %r13;
    mov.u32 %r14, %tid.x;
    add.u32 %r12, %r12, %r14;
    setp.ge.u32 %p0, %r12, %r1;
    @%p0 bra SPLIT_DONE;

    // Decode a contiguous four-dimensional output index.
    mul.lo.u32 %r15, %r9, %r10;
    mul.lo.u32 %r15, %r15, %r11;
    div.u32 %r16, %r12, %r15;
    rem.u32 %r17, %r12, %r15;
    mul.lo.u32 %r15, %r10, %r11;
    div.u32 %r18, %r17, %r15;
    rem.u32 %r17, %r17, %r15;
    div.u32 %r19, %r17, %r11;
    rem.u32 %r20, %r17, %r11;
    setp.eq.u32 %p1, %r2, 0;
    @%p1 add.u32 %r16, %r16, %r3;
    setp.eq.u32 %p2, %r2, 1;
    @%p2 add.u32 %r18, %r18, %r3;
    setp.eq.u32 %p3, %r2, 2;
    @%p3 add.u32 %r19, %r19, %r3;
    setp.eq.u32 %p4, %r2, 3;
    @%p4 add.u32 %r20, %r20, %r3;

    mul.lo.u32 %r21, %r16, %r5;
    add.u32 %r21, %r21, %r18;
    mul.lo.u32 %r21, %r21, %r6;
    add.u32 %r21, %r21, %r19;
    mul.lo.u32 %r21, %r21, %r7;
    add.u32 %r21, %r21, %r20;
    mul.wide.u32 %rd3, %r21, 4;
    add.u64 %rd3, %rd1, %rd3;
    ld.global.f32 %f1, [%rd3];
    mul.wide.u32 %rd3, %r12, 4;
    add.u64 %rd3, %rd2, %rd3;
    st.global.f32 [%rd3], %f1;
SPLIT_DONE:
    ret;
}

.visible .entry cat_copy(
    .param .u64 p_input,
    .param .u64 p_output,
    .param .u32 p_count,
    .param .u32 p_dim,
    .param .u32 p_offset,
    .param .u32 p_in0,
    .param .u32 p_in1,
    .param .u32 p_in2,
    .param .u32 p_in3,
    .param .u32 p_out0,
    .param .u32 p_out1,
    .param .u32 p_out2,
    .param .u32 p_out3
)
{
    .reg .pred %p<5>;
    .reg .u32 %r<25>;
    .reg .u64 %rd<4>;
    .reg .f32 %f<2>;
    ld.param.u64 %rd1, [p_input];
    ld.param.u64 %rd2, [p_output];
    ld.param.u32 %r1, [p_count];
    ld.param.u32 %r2, [p_dim];
    ld.param.u32 %r3, [p_offset];
    ld.param.u32 %r4, [p_in0];
    ld.param.u32 %r5, [p_in1];
    ld.param.u32 %r6, [p_in2];
    ld.param.u32 %r7, [p_in3];
    ld.param.u32 %r8, [p_out0];
    ld.param.u32 %r9, [p_out1];
    ld.param.u32 %r10, [p_out2];
    ld.param.u32 %r11, [p_out3];
    mov.u32 %r12, %ctaid.x;
    mov.u32 %r13, %ntid.x;
    mul.lo.u32 %r12, %r12, %r13;
    mov.u32 %r14, %tid.x;
    add.u32 %r12, %r12, %r14;
    setp.ge.u32 %p0, %r12, %r1;
    @%p0 bra CAT_DONE;

    mul.lo.u32 %r15, %r9, %r10;
    mul.lo.u32 %r15, %r15, %r11;
    div.u32 %r16, %r12, %r15;
    rem.u32 %r17, %r12, %r15;
    mul.lo.u32 %r15, %r10, %r11;
    div.u32 %r18, %r17, %r15;
    rem.u32 %r17, %r17, %r15;
    div.u32 %r19, %r17, %r11;
    rem.u32 %r20, %r17, %r11;

    setp.eq.u32 %p1, %r2, 0;
    @%p1 bra CAT_DIM0;
    setp.eq.u32 %p1, %r2, 1;
    @%p1 bra CAT_DIM1;
    setp.eq.u32 %p1, %r2, 2;
    @%p1 bra CAT_DIM2;
    setp.eq.u32 %p1, %r2, 3;
    @%p1 bra CAT_DIM3;
    bra CAT_DONE;
CAT_DIM0:
    add.u32 %r22, %r3, %r4;
    setp.ge.u32 %p2, %r16, %r3;
    setp.lt.u32 %p3, %r16, %r22;
    and.pred %p2, %p2, %p3;
    @!%p2 bra CAT_DONE;
    sub.u32 %r16, %r16, %r3;
    bra CAT_COORD_DONE;
CAT_DIM1:
    add.u32 %r22, %r3, %r5;
    setp.ge.u32 %p2, %r18, %r3;
    setp.lt.u32 %p3, %r18, %r22;
    and.pred %p2, %p2, %p3;
    @!%p2 bra CAT_DONE;
    sub.u32 %r18, %r18, %r3;
    bra CAT_COORD_DONE;
CAT_DIM2:
    add.u32 %r22, %r3, %r6;
    setp.ge.u32 %p2, %r19, %r3;
    setp.lt.u32 %p3, %r19, %r22;
    and.pred %p2, %p2, %p3;
    @!%p2 bra CAT_DONE;
    sub.u32 %r19, %r19, %r3;
    bra CAT_COORD_DONE;
CAT_DIM3:
    add.u32 %r22, %r3, %r7;
    setp.ge.u32 %p2, %r20, %r3;
    setp.lt.u32 %p3, %r20, %r22;
    and.pred %p2, %p2, %p3;
    @!%p2 bra CAT_DONE;
    sub.u32 %r20, %r20, %r3;
CAT_COORD_DONE:

    mul.lo.u32 %r21, %r16, %r5;
    add.u32 %r21, %r21, %r18;
    mul.lo.u32 %r21, %r21, %r6;
    add.u32 %r21, %r21, %r19;
    mul.lo.u32 %r21, %r21, %r7;
    add.u32 %r21, %r21, %r20;
    mul.wide.u32 %rd3, %r21, 4;
    add.u64 %rd3, %rd1, %rd3;
    ld.global.f32 %f1, [%rd3];
    mul.wide.u32 %rd3, %r12, 4;
    add.u64 %rd3, %rd2, %rd3;
    st.global.f32 [%rd3], %f1;
CAT_DONE:
    ret;
}

.visible .entry upsample_nearest2d(
    .param .u64 p_input,
    .param .u64 p_output,
    .param .u32 p_count,
    .param .u32 p_channels,
    .param .u32 p_in_h,
    .param .u32 p_in_w,
    .param .u32 p_out_h,
    .param .u32 p_out_w
)
{
    .reg .pred %p<1>;
    .reg .u32 %r<18>;
    .reg .u64 %rd<4>;
    .reg .f32 %f<2>;
    ld.param.u64 %rd1, [p_input];
    ld.param.u64 %rd2, [p_output];
    ld.param.u32 %r1, [p_count];
    ld.param.u32 %r2, [p_channels];
    ld.param.u32 %r3, [p_in_h];
    ld.param.u32 %r4, [p_in_w];
    ld.param.u32 %r5, [p_out_h];
    ld.param.u32 %r6, [p_out_w];
    mov.u32 %r7, %ctaid.x;
    mov.u32 %r8, %ntid.x;
    mul.lo.u32 %r7, %r7, %r8;
    mov.u32 %r9, %tid.x;
    add.u32 %r7, %r7, %r9;
    setp.ge.u32 %p0, %r7, %r1;
    @%p0 bra UPSAMPLE_DONE;

    mul.lo.u32 %r10, %r5, %r6;
    div.u32 %r11, %r7, %r10;
    rem.u32 %r12, %r7, %r10;
    div.u32 %r13, %r12, %r6;
    rem.u32 %r14, %r12, %r6;
    mul.lo.u32 %r15, %r13, %r3;
    div.u32 %r15, %r15, %r5;
    mul.lo.u32 %r16, %r14, %r4;
    div.u32 %r16, %r16, %r6;
    mul.lo.u32 %r17, %r11, %r2;
    add.u32 %r17, %r17, %r15;
    mul.lo.u32 %r17, %r17, %r4;
    add.u32 %r17, %r17, %r16;
    mul.wide.u32 %rd3, %r17, 4;
    add.u64 %rd3, %rd1, %rd3;
    ld.global.f32 %f1, [%rd3];
    mul.wide.u32 %rd3, %r7, 4;
    add.u64 %rd3, %rd2, %rd3;
    st.global.f32 [%rd3], %f1;
UPSAMPLE_DONE:
    ret;
}
""".encode("ascii")


def load_driver() -> ctypes.CDLL:
    path = ctypes.util.find_library("cuda") or "libcuda.so.1"
    try:
        return ctypes.CDLL(path)
    except OSError as exc:
        raise RuntimeError("CUDA Driver API unavailable") from exc


def configure_driver(driver: ctypes.CDLL) -> None:
    driver.cuInit.argtypes = [ctypes.c_uint]
    driver.cuInit.restype = CUresult
    driver.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    driver.cuDeviceGetCount.restype = CUresult
    driver.cuDeviceGet.argtypes = [ctypes.POINTER(CUdevice), ctypes.c_int]
    driver.cuDeviceGet.restype = CUresult
    driver.cuDeviceGetName.argtypes = [ctypes.c_char_p, ctypes.c_int, CUdevice]
    driver.cuDeviceGetName.restype = CUresult
    driver.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, CUdevice]
    driver.cuDeviceGetAttribute.restype = CUresult
    driver.cuDeviceTotalMem_v2.argtypes = [ctypes.POINTER(ctypes.c_size_t), CUdevice]
    driver.cuDeviceTotalMem_v2.restype = CUresult
    driver.cuCtxCreate_v2.argtypes = [ctypes.POINTER(CUcontext), ctypes.c_uint, CUdevice]
    driver.cuCtxCreate_v2.restype = CUresult
    driver.cuCtxDestroy_v2.argtypes = [CUcontext]
    driver.cuCtxDestroy_v2.restype = CUresult
    driver.cuMemAlloc_v2.argtypes = [ctypes.POINTER(CUdeviceptr), ctypes.c_size_t]
    driver.cuMemAlloc_v2.restype = CUresult
    driver.cuMemFree_v2.argtypes = [CUdeviceptr]
    driver.cuMemFree_v2.restype = CUresult
    driver.cuMemcpyHtoD_v2.argtypes = [CUdeviceptr, ctypes.c_void_p, ctypes.c_size_t]
    driver.cuMemcpyHtoD_v2.restype = CUresult
    driver.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, CUdeviceptr, ctypes.c_size_t]
    driver.cuMemcpyDtoH_v2.restype = CUresult
    driver.cuModuleLoadData.argtypes = [ctypes.POINTER(CUmodule), ctypes.c_void_p]
    driver.cuModuleLoadData.restype = CUresult
    driver.cuModuleUnload.argtypes = [CUmodule]
    driver.cuModuleUnload.restype = CUresult
    driver.cuModuleGetFunction.argtypes = [ctypes.POINTER(CUfunction), CUmodule, ctypes.c_char_p]
    driver.cuModuleGetFunction.restype = CUresult
    driver.cuLaunchKernel.argtypes = [
        CUfunction, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ]
    driver.cuLaunchKernel.restype = CUresult
    driver.cuCtxSynchronize.argtypes = []
    driver.cuCtxSynchronize.restype = CUresult
    driver.cuGetErrorName.argtypes = [CUresult, ctypes.POINTER(ctypes.c_char_p)]
    driver.cuGetErrorName.restype = CUresult
    driver.cuGetErrorString.argtypes = [CUresult, ctypes.POINTER(ctypes.c_char_p)]
    driver.cuGetErrorString.restype = CUresult


def check(driver: ctypes.CDLL, result: int, action: str) -> None:
    if result == CUDA_SUCCESS:
        return
    name = ctypes.c_char_p()
    message = ctypes.c_char_p()
    driver.cuGetErrorName(result, ctypes.byref(name))
    driver.cuGetErrorString(result, ctypes.byref(message))
    raise RuntimeError(
        f"{action}: {name.value.decode() if name.value else result} "
        f"({message.value.decode() if message.value else 'unknown error'})"
    )


def detect_device(driver: ctypes.CDLL | None = None) -> tuple[ctypes.CDLL, dict[str, str]]:
    """Initialize CUDA and return the first device's capability information."""
    if driver is None:
        driver = load_driver()
        configure_driver(driver)
    check(driver, driver.cuInit(0), "cuInit")
    count = ctypes.c_int()
    check(driver, driver.cuDeviceGetCount(ctypes.byref(count)), "cuDeviceGetCount")
    if count.value < 1:
        raise RuntimeError("CUDA initialized, but no CUDA devices were found")
    device = CUdevice()
    check(driver, driver.cuDeviceGet(ctypes.byref(device), 0), "cuDeviceGet(0)")
    name = ctypes.create_string_buffer(256)
    major = ctypes.c_int()
    minor = ctypes.c_int()
    total_mem = ctypes.c_size_t()
    check(driver, driver.cuDeviceGetName(name, len(name), device), "cuDeviceGetName")
    check(driver, driver.cuDeviceGetAttribute(ctypes.byref(major), CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device), "query compute capability major")
    check(driver, driver.cuDeviceGetAttribute(ctypes.byref(minor), CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device), "query compute capability minor")
    check(driver, driver.cuDeviceTotalMem_v2(ctypes.byref(total_mem), device), "cuDeviceTotalMem")
    return driver, {
        "name": name.value.decode(errors="replace"),
        "compute_capability": f"{major.value}.{minor.value}",
        "memory_mib": f"{total_mem.value / (1024 * 1024):.1f}",
    }


class CudaExecutionBackend:
    """Minimal device-memory and kernel execution boundary for MatrixMan."""

    def __init__(self, device_index: int = 0):
        if device_index != 0:
            raise ValueError("only CUDA device 0 is currently supported")
        self.driver = load_driver()
        configure_driver(self.driver)
        self.driver, self.info = detect_device(self.driver)
        self.device = CUdevice()
        check(self.driver, self.driver.cuDeviceGet(ctypes.byref(self.device), device_index), "cuDeviceGet")
        self.context = CUcontext()
        self.module = CUmodule()
        self.closed = False
        try:
            check(self.driver, self.driver.cuCtxCreate_v2(ctypes.byref(self.context), 0, self.device), "cuCtxCreate")
            ptx = ctypes.create_string_buffer(PTX)
            check(
                self.driver,
                self.driver.cuModuleLoadData(
                    ctypes.byref(self.module), ctypes.cast(ptx, ctypes.c_void_p)
                ),
                "cuModuleLoadData (embedded sm_21 PTX module: matrix_add, matrix_mul, conv2d_nchw, batch_norm_inference, silu, split_copy, cat_copy, upsample_nearest2d)",
            )
            self.add_function = CUfunction()
            self.matmul_function = CUfunction()
            self.convolution_function = CUfunction()
            self.batch_norm_function = CUfunction()
            self.silu_function = CUfunction()
            self.split_function = CUfunction()
            self.cat_function = CUfunction()
            self.upsample_function = CUfunction()
            check(self.driver, self.driver.cuModuleGetFunction(ctypes.byref(self.add_function), self.module, b"matrix_add"), "get matrix_add")
            check(self.driver, self.driver.cuModuleGetFunction(ctypes.byref(self.matmul_function), self.module, b"matrix_mul"), "get matrix_mul")
            check(self.driver, self.driver.cuModuleGetFunction(ctypes.byref(self.convolution_function), self.module, b"conv2d_nchw"), "get conv2d_nchw")
            check(self.driver, self.driver.cuModuleGetFunction(ctypes.byref(self.batch_norm_function), self.module, b"batch_norm_inference"), "get batch_norm_inference")
            check(self.driver, self.driver.cuModuleGetFunction(ctypes.byref(self.silu_function), self.module, b"silu"), "get silu")
            check(self.driver, self.driver.cuModuleGetFunction(ctypes.byref(self.split_function), self.module, b"split_copy"), "get split_copy")
            check(self.driver, self.driver.cuModuleGetFunction(ctypes.byref(self.cat_function), self.module, b"cat_copy"), "get cat_copy")
            check(self.driver, self.driver.cuModuleGetFunction(ctypes.byref(self.upsample_function), self.module, b"upsample_nearest2d"), "get upsample_nearest2d")
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if self.closed:
            return
        if self.module:
            check(self.driver, self.driver.cuModuleUnload(self.module), "cuModuleUnload")
            self.module = CUmodule()
        if self.context:
            check(self.driver, self.driver.cuCtxDestroy_v2(self.context), "cuCtxDestroy")
            self.context = CUcontext()
        self.closed = True

    def __enter__(self) -> "CudaExecutionBackend":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def allocate(self, nbytes: int) -> CUdeviceptr:
        if nbytes <= 0:
            raise ValueError("device allocation size must be positive")
        pointer = CUdeviceptr()
        check(self.driver, self.driver.cuMemAlloc_v2(ctypes.byref(pointer), nbytes), "cuMemAlloc")
        return pointer

    def free(self, pointer: CUdeviceptr) -> None:
        if pointer and pointer.value:
            check(self.driver, self.driver.cuMemFree_v2(pointer), "cuMemFree")
            pointer.value = 0

    @staticmethod
    def _float32_array(array: np.ndarray, label: str) -> np.ndarray:
        if not isinstance(array, np.ndarray) or array.dtype != np.float32:
            raise TypeError(f"{label} must be a NumPy float32 array")
        if not array.flags.c_contiguous:
            raise ValueError(f"{label} must be C-contiguous")
        return array

    def to_device(self, array: np.ndarray) -> CUdeviceptr:
        array = self._float32_array(array, "host array")
        pointer = self.allocate(array.nbytes)
        try:
            check(self.driver, self.driver.cuMemcpyHtoD_v2(pointer, array.ctypes.data_as(ctypes.c_void_p), array.nbytes), "cuMemcpyHtoD")
        except Exception:
            self.free(pointer)
            raise
        return pointer

    def from_device(self, pointer: CUdeviceptr, shape: tuple[int, ...]) -> np.ndarray:
        output = np.empty(shape, dtype=np.float32)
        check(self.driver, self.driver.cuMemcpyDtoH_v2(output.ctypes.data_as(ctypes.c_void_p), pointer, output.nbytes), "cuMemcpyDtoH")
        return output

    def _launch(self, function: CUfunction, args: list[ctypes._SimpleCData], work_items: int) -> None:
        if work_items <= 0:
            raise ValueError("kernel work size must be positive")
        params = (ctypes.c_void_p * len(args))(
            *(ctypes.cast(ctypes.byref(arg), ctypes.c_void_p) for arg in args)
        )
        grid = (work_items + CUDA_BLOCK_SIZE - 1) // CUDA_BLOCK_SIZE
        check(self.driver, self.driver.cuLaunchKernel(function, grid, 1, 1, CUDA_BLOCK_SIZE, 1, 1, 0, None, params, None), "cuLaunchKernel")
        check(self.driver, self.driver.cuCtxSynchronize(), "cuCtxSynchronize")

    def add(self, a: CUdeviceptr, b: CUdeviceptr, output: CUdeviceptr, count: int) -> None:
        self._launch(self.add_function, [a, b, output, ctypes.c_uint(count)], count)

    def matmul(self, a: CUdeviceptr, b: CUdeviceptr, output: CUdeviceptr, m: int, k: int, n: int) -> None:
        if min(m, k, n) <= 0:
            raise ValueError("matrix dimensions must be positive")
        self._launch(
            self.matmul_function,
            [a, b, output, ctypes.c_uint(m), ctypes.c_uint(k), ctypes.c_uint(n)],
            m * n,
        )

    def convolution(
        self,
        input_pointer: CUdeviceptr,
        weight_pointer: CUdeviceptr,
        bias_pointer: CUdeviceptr,
        output_pointer: CUdeviceptr,
        n: int,
        c: int,
        h: int,
        w: int,
        k: int,
        r: int,
        s: int,
        out_h: int,
        out_w: int,
        stride_h: int,
        stride_w: int,
        pad_h: int,
        pad_w: int,
        dilation_h: int,
        dilation_w: int,
        groups: int,
    ) -> None:
        dimensions = (n, c, h, w, k, r, s, out_h, out_w)
        if min(dimensions) <= 0:
            raise ValueError("convolution dimensions must be positive")
        if min(stride_h, stride_w, dilation_h, dilation_w) <= 0:
            raise ValueError("convolution stride and dilation must be positive")
        if groups <= 0 or c % groups or k % groups:
            raise ValueError("convolution groups must divide input and output channels")
        total_outputs = n * k * out_h * out_w
        print(
            "[MatrixMan/CUDA] Conv2D launch: "
            f"N={n}, Cin={c}, Hin={h}, Win={w}, Cout={k}, "
            f"Kh={r}, Kw={s}, Hout={out_h}, Wout={out_w}, "
            f"stride_h/w={stride_h}/{stride_w}, pad_h/w={pad_h}/{pad_w}, "
            f"dilation_h/w={dilation_h}/{dilation_w}, groups={groups}, "
            f"total_outputs={total_outputs}"
        )
        self._launch(
            self.convolution_function,
            [
                input_pointer, weight_pointer, bias_pointer, output_pointer,
                ctypes.c_uint(n), ctypes.c_uint(c), ctypes.c_uint(h), ctypes.c_uint(w),
                ctypes.c_uint(k), ctypes.c_uint(r), ctypes.c_uint(s),
                ctypes.c_uint(out_h), ctypes.c_uint(out_w),
                ctypes.c_uint(stride_h), ctypes.c_uint(stride_w),
                ctypes.c_uint(pad_h), ctypes.c_uint(pad_w),
                ctypes.c_uint(dilation_h), ctypes.c_uint(dilation_w),
                ctypes.c_uint(groups),
            ],
            total_outputs,
        )

    def batch_norm(
        self,
        input_pointer: CUdeviceptr,
        running_mean_pointer: CUdeviceptr,
        running_var_pointer: CUdeviceptr,
        weight_pointer: CUdeviceptr,
        bias_pointer: CUdeviceptr,
        output_pointer: CUdeviceptr,
        count: int,
        channels: int,
        spatial: int,
        eps: float,
    ) -> None:
        if min(count, channels, spatial) <= 0:
            raise ValueError("batch norm dimensions must be positive")
        self._launch(
            self.batch_norm_function,
            [
                input_pointer, running_mean_pointer, running_var_pointer,
                weight_pointer, bias_pointer, output_pointer,
                ctypes.c_uint(count), ctypes.c_uint(channels),
                ctypes.c_uint(spatial), ctypes.c_float(eps),
            ],
            count,
        )

    def silu(self, input_pointer: CUdeviceptr, output_pointer: CUdeviceptr, count: int) -> None:
        if count <= 0:
            raise ValueError("SiLU tensor size must be positive")
        self._launch(
            self.silu_function,
            [input_pointer, output_pointer, ctypes.c_uint(count)],
            count,
        )

    def split_copy(
        self,
        input_pointer: CUdeviceptr,
        output_pointer: CUdeviceptr,
        count: int,
        dimension: int,
        offset: int,
        input_shape: tuple[int, int, int, int],
        output_shape: tuple[int, int, int, int],
    ) -> None:
        if count <= 0:
            raise ValueError("split output size must be positive")
        self._launch(
            self.split_function,
            [
                input_pointer, output_pointer, ctypes.c_uint(count),
                ctypes.c_uint(dimension), ctypes.c_uint(offset),
                *(ctypes.c_uint(value) for value in input_shape),
                *(ctypes.c_uint(value) for value in output_shape),
            ],
            count,
        )

    def cat_copy(
        self,
        input_pointer: CUdeviceptr,
        output_pointer: CUdeviceptr,
        count: int,
        dimension: int,
        offset: int,
        input_shape: tuple[int, int, int, int],
        output_shape: tuple[int, int, int, int],
    ) -> None:
        if count <= 0:
            raise ValueError("cat input size must be positive")
        self._launch(
            self.cat_function,
            [
                input_pointer, output_pointer, ctypes.c_uint(count),
                ctypes.c_uint(dimension), ctypes.c_uint(offset),
                *(ctypes.c_uint(value) for value in input_shape),
                *(ctypes.c_uint(value) for value in output_shape),
            ],
            count,
        )

    def upsample_nearest2d(
        self,
        input_pointer: CUdeviceptr,
        output_pointer: CUdeviceptr,
        count: int,
        channels: int,
        input_height: int,
        input_width: int,
        output_height: int,
        output_width: int,
    ) -> None:
        if min(count, channels, input_height, input_width, output_height, output_width) <= 0:
            raise ValueError("upsample dimensions must be positive")
        self._launch(
            self.upsample_function,
            [
                input_pointer, output_pointer, ctypes.c_uint(count),
                ctypes.c_uint(channels), ctypes.c_uint(input_height),
                ctypes.c_uint(input_width), ctypes.c_uint(output_height),
                ctypes.c_uint(output_width),
            ],
            count,
        )


def print_check(name: str, gpu: np.ndarray, cpu: np.ndarray) -> None:
    matches = np.allclose(gpu, cpu, rtol=1e-5, atol=1e-5)
    print(f"\n{name} CUDA result:")
    print(gpu)
    print(f"\n{name} NumPy reference:")
    print(cpu)
    print("max abs diff:", float(np.max(np.abs(gpu - cpu))))
    print("matches CPU reference:", bool(matches))
    if not matches:
        raise RuntimeError(f"{name} result does not match CPU reference")


def run_diagnostic() -> int:
    n = 4
    a = np.arange(1, n * n + 1, dtype=np.float32).reshape(n, n)
    b = np.arange(n * n, 0, -1, dtype=np.float32).reshape(n, n)
    np.set_printoptions(precision=3, suppress=True)
    with CudaExecutionBackend() as cuda:
        print("CUDA device:", cuda.info["name"])
        print("compute capability:", cuda.info["compute_capability"])
        print("device memory (MiB):", cuda.info["memory_mib"])
        print("\nA:")
        print(a)
        print("\nB:")
        print(b)
        a_dev = cuda.to_device(a)
        b_dev = cuda.to_device(b)
        add_dev = cuda.allocate(a.nbytes)
        mul_dev = cuda.allocate(a.nbytes)
        try:
            cuda.add(a_dev, b_dev, add_dev, a.size)
            cuda.matmul(a_dev, b_dev, mul_dev, n, n, n)
            print_check("C = A + B", cuda.from_device(add_dev, a.shape), a + b)
            print_check("C = A x B", cuda.from_device(mul_dev, a.shape), a @ b)
        finally:
            for pointer in (a_dev, b_dev, add_dev, mul_dev):
                cuda.free(pointer)

        m, k, p = 3, 5, 7
        non_square_a = np.arange(1, m * k + 1, dtype=np.float32).reshape(m, k)
        non_square_b = np.arange(1, k * p + 1, dtype=np.float32).reshape(k, p)
        a_dev = cuda.to_device(non_square_a)
        b_dev = cuda.to_device(non_square_b)
        output_dev = cuda.allocate(m * p * np.dtype(np.float32).itemsize)
        try:
            cuda.matmul(a_dev, b_dev, output_dev, m, k, p)
            print_check(
                "C = (3x5) A x (5x7) B",
                cuda.from_device(output_dev, (m, p)),
                non_square_a @ non_square_b,
            )
        finally:
            for pointer in (a_dev, b_dev, output_dev):
                cuda.free(pointer)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run_diagnostic())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
