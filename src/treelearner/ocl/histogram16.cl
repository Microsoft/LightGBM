/*!
 * Copyright (c) 2017-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2017-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 *
 * \brief This file can either be read and passed to an OpenCL compiler directly,
 *        or included in a C++11 source file as a string literal.
 */
#ifndef __OPENCL_VERSION__
// If we are including this file in C++,
// the entire source file following (except the last #endif) will become
// a raw string literal. The extra ")" is just for matching parentheses
// to make the editor happy. The extra ")" and extra endif will be skipped.
// DO NOT add anything between here and the next #ifdef, otherwise you need
// to modify the skip count at the end of this file.
R""()
#endif

#ifndef _HISTOGRAM_16_KERNEL_
#define _HISTOGRAM_16_KERNEL_

#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

// Configurable options:
// NUM_BANKS should be a power of 2
#ifndef NUM_BANKS
#define NUM_BANKS 8
#endif
// how many bits in thread ID represent the bank = log2(NUM_BANKS)
#ifndef BANK_BITS
#define BANK_BITS 3
#endif
// use double precision or not
#ifndef USE_DP_FLOAT
#define USE_DP_FLOAT 0
#endif
// ignore hessian, and use the local memory for hessian as an additional bank for gradient
#ifndef CONST_HESSIAN
#define CONST_HESSIAN 0
#endif


#define LOCAL_SIZE_0 256
#define NUM_BINS 16
// if USE_DP_FLOAT is set to 1, we will use double precision for the accumulator
#if USE_DP_FLOAT == 1
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
typedef double acc_type;
typedef ulong acc_int_type;
#define as_acc_type as_double
#define as_acc_int_type as_ulong
#else
typedef float acc_type;
typedef uint acc_int_type;
#define as_acc_type as_float
#define as_acc_int_type as_uint
#endif
// number of features to process in a 4-byte feature tuple
#define DWORD_FEATURES 8
// number of bits per feature
#define FEATURE_BITS (sizeof(uchar4) * 8 / DWORD_FEATURES)
// bit mask for number of features to process in a 4-byte feature tuple
#define DWORD_FEATURES_MASK (DWORD_FEATURES - 1)
// log2 of number of features to process in a 4-byte feature tuple
#define LOG2_DWORD_FEATURES 3
// mask for getting the bank ID
#define BANK_MASK (NUM_BANKS - 1)
// 8 features, each has a gradient and a hessian
#define HG_BIN_MULT (NUM_BANKS * DWORD_FEATURES * 2)
// 8 features, each has a counter
#define CNT_BIN_MULT (NUM_BANKS * DWORD_FEATURES)
// local memory size in bytes
#define LOCAL_MEM_SIZE (DWORD_FEATURES * (sizeof(uint) + 2 * sizeof(acc_type)) * NUM_BINS * NUM_BANKS)

// unroll the atomic operation for a few times. Takes more code space,
// but compiler can generate better code for faster atomics.
#define UNROLL_ATOMIC 1

// Options passed by compiler at run time:
// IGNORE_INDICES will be set when the kernel does not
// #define IGNORE_INDICES
// #define POWER_FEATURE_WORKGROUPS 10

// use all features and do not use feature mask
#ifndef ENABLE_ALL_FEATURES
#define ENABLE_ALL_FEATURES 1
#endif

// detect Nvidia platforms
#ifdef cl_nv_pragma_unroll
#define NVIDIA 1
#endif

// use binary patching for AMD GCN 1.2 or newer
#ifndef AMD_USE_DS_ADD_F32
#define AMD_USE_DS_ADD_F32 0
#endif

typedef uint data_size_t;
typedef float score_t;

#define ATOMIC_FADD_SUB1 { \
    expected.f_val = current.f_val; \
    next.f_val = expected.f_val + val; \
    current.u_val = atom_cmpxchg((volatile __local acc_int_type *)addr, expected.u_val, next.u_val); \
    if (current.u_val == expected.u_val) \
        goto end; \
    }
#define ATOMIC_FADD_SUB2  ATOMIC_FADD_SUB1 \
                          ATOMIC_FADD_SUB1
#define ATOMIC_FADD_SUB4  ATOMIC_FADD_SUB2 \
                          ATOMIC_FADD_SUB2
#define ATOMIC_FADD_SUB8  ATOMIC_FADD_SUB4 \
                          ATOMIC_FADD_SUB4
#define ATOMIC_FADD_SUB16 ATOMIC_FADD_SUB8 \
                          ATOMIC_FADD_SUB8
#define ATOMIC_FADD_SUB32 ATOMIC_FADD_SUB16\
                          ATOMIC_FADD_SUB16
#define ATOMIC_FADD_SUB64 ATOMIC_FADD_SUB32\
                          ATOMIC_FADD_SUB32


// atomic add for float number in local memory
inline void atomic_local_add_f(__local acc_type *addr, const float val)
{
    union{
        acc_int_type u_val;
        acc_type f_val;
    } next, expected, current;
#if (NVIDIA == 1 && USE_DP_FLOAT == 0)
    float res = 0;
    asm volatile ("atom.shared.add.f32 %0, [%1], %2;" : "=f"(res) : "l"(addr), "f"(val));
#elif (AMD_USE_DS_ADD_F32 == 1 && USE_DP_FLAT == 0)
    // this instruction (DS_AND_U32) will be patched into a DS_ADD_F32
    // we need to hack here because DS_ADD_F32 is not exposed via OpenCL
    atom_and((__local acc_int_type *)addr, as_acc_int_type(val));
#else
    current.f_val = *addr;
    #if UNROLL_ATOMIC == 1
    // provide a fast path
    // then do the complete loop
    // this should work on all devices
    ATOMIC_FADD_SUB8
    ATOMIC_FADD_SUB4
    #endif
    do {
        expected.f_val = current.f_val;
        next.f_val = expected.f_val + val;
        current.u_val = atom_cmpxchg((volatile __local acc_int_type *)addr, expected.u_val, next.u_val);
    } while (current.u_val != expected.u_val);
    end:
        ;
#endif
}

/* Makes MSVC happy with long string literal
)""
R""()
*/
// this function will be called by histogram16
// we have one sub-histogram of one feature in registers, and need to read others
void within_kernel_reduction16x8(uchar8 feature_mask,
                           __global const acc_type* restrict feature4_sub_hist,
                           const uint skip_id,
                           acc_type stat_val,
                           const ushort num_sub_hist,
                           __global acc_type* restrict output_buf,
                           __local acc_type * restrict local_hist) {
    const ushort ltid = get_local_id(0); // range 0 - 255
    const ushort lsize = LOCAL_SIZE_0;
    ushort feature_id = ltid & DWORD_FEATURES_MASK; // range 0 - 7
    uchar is_hessian_first = (ltid >> LOG2_DWORD_FEATURES) & 1; // hessian or gradient
    ushort bin_id = ltid >> (LOG2_DWORD_FEATURES + 1); // range 0 - 16
    ushort i;
    #if POWER_FEATURE_WORKGROUPS != 0
    // if there is only 1 work group, no need to do the reduction
    // add all sub-histograms for 4 features
    __global const acc_type* restrict p = feature4_sub_hist + ltid;
    for (i = 0; i < skip_id; ++i) {
            // 256 threads working on 8 features' 16 bins, gradient and Hessian
            stat_val += *p;
            p += NUM_BINS * DWORD_FEATURES * 2;
    }
    // skip the counters we already have
    p += 2 * DWORD_FEATURES * NUM_BINS;
    for (i = i + 1; i < num_sub_hist; ++i) {
            stat_val += *p;
            p += NUM_BINS * DWORD_FEATURES * 2;
    }
    #endif
    // printf("thread %d:feature=%d, bin_id=%d, hessian=%d, stat_val=%f, cnt=%d", ltid, feature_id, bin_id, is_hessian_first, stat_val, cnt_val);
    // now overwrite the local_hist for final reduction and output
    // reverse the f7...f0 order to match the real order
    feature_id = DWORD_FEATURES_MASK - feature_id;
    local_hist[feature_id * 2 * NUM_BINS + bin_id * 2 + is_hessian_first] = stat_val;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (i = ltid; i < DWORD_FEATURES * 2 * NUM_BINS; i += lsize) {
        output_buf[i] = local_hist[i];
    }
}


/* Makes MSVC happy with long string literal
)""
R""()
*/

__attribute__((reqd_work_group_size(LOCAL_SIZE_0, 1, 1)))
#if USE_CONSTANT_BUF == 1
__kernel void histogram16(__global const uchar4* restrict feature_data_base,
                      __constant const uchar8* restrict feature_masks __attribute__((max_constant_size(65536))),
                      const data_size_t feature_size,
                      __constant const data_size_t* restrict data_indices __attribute__((max_constant_size(65536))),
                      const data_size_t num_data,
                      __constant const score_t* restrict ordered_gradients __attribute__((max_constant_size(65536))),
#if CONST_HESSIAN == 0
                      __constant const score_t* restrict ordered_hessians __attribute__((max_constant_size(65536))),
#else
                      const score_t const_hessian,
#endif
                      __global char* restrict output_buf,
                      __global volatile int * sync_counters,
                      __global acc_type* restrict hist_buf_base) {
#else
__kernel void histogram16(__global const uchar4* feature_data_base,
                      __constant const uchar8* restrict feature_masks __attribute__((max_constant_size(65536))),
                      const data_size_t feature_size,
                      __global const data_size_t* data_indices,
                      const data_size_t num_data,
                      __global const score_t*  ordered_gradients,
#if CONST_HESSIAN == 0
                      __global const score_t*  ordered_hessians,
#else
                      const score_t const_hessian,
#endif
                      __global char* restrict output_buf,
                      __global volatile int * sync_counters,
                      __global acc_type* restrict hist_buf_base) {
#endif
    // allocate the local memory array aligned with float2, to guarantee correct alignment on NVIDIA platforms
    // otherwise a "Misaligned Address" exception may occur
    __local float2 shared_array[LOCAL_MEM_SIZE/sizeof(float2)];
    const uint gtid = get_global_id(0);
    const uint gsize = get_global_size(0);
    const ushort ltid = get_local_id(0);
    const ushort lsize = LOCAL_SIZE_0; // get_local_size(0);
    const ushort group_id = get_group_id(0);

    // local memory per workgroup is 12 KB
    // clear local memory
    __local uint * ptr = (__local uint *) shared_array;
    for (int i = ltid; i < LOCAL_MEM_SIZE/sizeof(uint); i += lsize) {
        ptr[i] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // gradient/hessian histograms
    // assume this starts at 32 * 4 = 128-byte boundary
    // each bank: 2 * 8 * 16 * size_of(float) = 1 KB
    // there are 8 banks (sub-histograms) used by 256 threads total 8 KB
    /* memory layout of gh_hist:
       -----------------------------------------------------------------------------------------------
       bk0_g_f0_bin0   bk0_g_f1_bin0   bk0_g_f2_bin0   bk0_g_f3_bin0   bk0_g_f4_bin0   bk0_g_f5_bin0   bk0_g_f6_bin0   bk0_g_f7_bin0
       bk0_h_f0_bin0   bk0_h_f1_bin0   bk0_h_f2_bin0   bk0_h_f3_bin0   bk0_h_f4_bin0   bk0_h_f5_bin0   bk0_h_f6_bin0   bk0_h_f7_bin0
       bk1_g_f0_bin0   bk1_g_f1_bin0   bk1_g_f2_bin0   bk1_g_f3_bin0   bk1_g_f4_bin0   bk1_g_f5_bin0   bk1_g_f6_bin0   bk1_g_f7_bin0
       bk1_h_f0_bin0   bk1_h_f1_bin0   bk1_h_f2_bin0   bk1_h_f3_bin0   bk1_h_f4_bin0   bk1_h_f5_bin0   bk1_h_f6_bin0   bk1_h_f7_bin0
       bk2_g_f0_bin0   bk2_g_f1_bin0   bk2_g_f2_bin0   bk2_g_f3_bin0   bk2_g_f4_bin0   bk2_g_f5_bin0   bk2_g_f6_bin0   bk2_g_f7_bin0
       bk2_h_f0_bin0   bk2_h_f1_bin0   bk2_h_f2_bin0   bk2_h_f3_bin0   bk2_h_f4_bin0   bk2_h_f5_bin0   bk2_h_f6_bin0   bk2_h_f7_bin0
       bk3_g_f0_bin0   bk3_g_f1_bin0   bk3_g_f2_bin0   bk3_g_f3_bin0   bk3_g_f4_bin0   bk3_g_f5_bin0   bk3_g_f6_bin0   bk3_g_f7_bin0
       bk3_h_f0_bin0   bk3_h_f1_bin0   bk3_h_f2_bin0   bk3_h_f3_bin0   bk3_h_f4_bin0   bk3_h_f5_bin0   bk3_h_f6_bin0   bk3_h_f7_bin0
       bk4_g_f0_bin0   bk4_g_f1_bin0   bk4_g_f2_bin0   bk4_g_f3_bin0   bk4_g_f4_bin0   bk4_g_f5_bin0   bk4_g_f6_bin0   bk4_g_f7_bin0
       bk4_h_f0_bin0   bk4_h_f1_bin0   bk4_h_f2_bin0   bk4_h_f3_bin0   bk4_h_f4_bin0   bk4_h_f5_bin0   bk4_h_f6_bin0   bk4_h_f7_bin0
       bk5_g_f0_bin0   bk5_g_f1_bin0   bk5_g_f2_bin0   bk5_g_f3_bin0   bk5_g_f4_bin0   bk5_g_f5_bin0   bk5_g_f6_bin0   bk5_g_f7_bin0
       bk5_h_f0_bin0   bk5_h_f1_bin0   bk5_h_f2_bin0   bk5_h_f3_bin0   bk5_h_f4_bin0   bk5_h_f5_bin0   bk5_h_f6_bin0   bk5_h_f7_bin0
       bk6_g_f0_bin0   bk6_g_f1_bin0   bk6_g_f2_bin0   bk6_g_f3_bin0   bk6_g_f4_bin0   bk6_g_f5_bin0   bk6_g_f6_bin0   bk6_g_f7_bin0
       bk6_h_f0_bin0   bk6_h_f1_bin0   bk6_h_f2_bin0   bk6_h_f3_bin0   bk6_h_f4_bin0   bk6_h_f5_bin0   bk6_h_f6_bin0   bk6_h_f7_bin0
       bk7_g_f0_bin0   bk7_g_f1_bin0   bk7_g_f2_bin0   bk7_g_f3_bin0   bk7_g_f4_bin0   bk7_g_f5_bin0   bk7_g_f6_bin0   bk7_g_f7_bin0
       bk7_h_f0_bin0   bk7_h_f1_bin0   bk7_h_f2_bin0   bk7_h_f3_bin0   bk7_h_f4_bin0   bk7_h_f5_bin0   bk7_h_f6_bin0   bk7_h_f7_bin0
       ...
       bk0_g_f0_bin16  bk0_g_f1_bin16  bk0_g_f2_bin16  bk0_g_f3_bin16  bk0_g_f4_bin16  bk0_g_f5_bin16  bk0_g_f6_bin16  bk0_g_f7_bin16
       bk0_h_f0_bin16  bk0_h_f1_bin16  bk0_h_f2_bin16  bk0_h_f3_bin16  bk0_h_f4_bin16  bk0_h_f5_bin16  bk0_h_f6_bin16  bk0_h_f7_bin16
       bk1_g_f0_bin16  bk1_g_f1_bin16  bk1_g_f2_bin16  bk1_g_f3_bin16  bk1_g_f4_bin16  bk1_g_f5_bin16  bk1_g_f6_bin16  bk1_g_f7_bin16
       bk1_h_f0_bin16  bk1_h_f1_bin16  bk1_h_f2_bin16  bk1_h_f3_bin16  bk1_h_f4_bin16  bk1_h_f5_bin16  bk1_h_f6_bin16  bk1_h_f7_bin16
       bk2_g_f0_bin16  bk2_g_f1_bin16  bk2_g_f2_bin16  bk2_g_f3_bin16  bk2_g_f4_bin16  bk2_g_f5_bin16  bk2_g_f6_bin16  bk2_g_f7_bin16
       bk2_h_f0_bin16  bk2_h_f1_bin16  bk2_h_f2_bin16  bk2_h_f3_bin16  bk2_h_f4_bin16  bk2_h_f5_bin16  bk2_h_f6_bin16  bk2_h_f7_bin16
       bk3_g_f0_bin16  bk3_g_f1_bin16  bk3_g_f2_bin16  bk3_g_f3_bin16  bk3_g_f4_bin16  bk3_g_f5_bin16  bk3_g_f6_bin16  bk3_g_f7_bin16
       bk3_h_f0_bin16  bk3_h_f1_bin16  bk3_h_f2_bin16  bk3_h_f3_bin16  bk3_h_f4_bin16  bk3_h_f5_bin16  bk3_h_f6_bin16  bk3_h_f7_bin16
       bk4_g_f0_bin16  bk4_g_f1_bin16  bk4_g_f2_bin16  bk4_g_f3_bin16  bk4_g_f4_bin16  bk4_g_f5_bin16  bk4_g_f6_bin16  bk4_g_f7_bin16
       bk4_h_f0_bin16  bk4_h_f1_bin16  bk4_h_f2_bin16  bk4_h_f3_bin16  bk4_h_f4_bin16  bk4_h_f5_bin16  bk4_h_f6_bin16  bk4_h_f7_bin16
       bk5_g_f0_bin16  bk5_g_f1_bin16  bk5_g_f2_bin16  bk5_g_f3_bin16  bk5_g_f4_bin16  bk5_g_f5_bin16  bk5_g_f6_bin16  bk5_g_f7_bin16
       bk5_h_f0_bin16  bk5_h_f1_bin16  bk5_h_f2_bin16  bk5_h_f3_bin16  bk5_h_f4_bin16  bk5_h_f5_bin16  bk5_h_f6_bin16  bk5_h_f7_bin16
       bk6_g_f0_bin16  bk6_g_f1_bin16  bk6_g_f2_bin16  bk6_g_f3_bin16  bk6_g_f4_bin16  bk6_g_f5_bin16  bk6_g_f6_bin16  bk6_g_f7_bin16
       bk6_h_f0_bin16  bk6_h_f1_bin16  bk6_h_f2_bin16  bk6_h_f3_bin16  bk6_h_f4_bin16  bk6_h_f5_bin16  bk6_h_f6_bin16  bk6_h_f7_bin16
       bk7_g_f0_bin16  bk7_g_f1_bin16  bk7_g_f2_bin16  bk7_g_f3_bin16  bk7_g_f4_bin16  bk7_g_f5_bin16  bk7_g_f6_bin16  bk7_g_f7_bin16
       bk7_h_f0_bin16  bk7_h_f1_bin16  bk7_h_f2_bin16  bk7_h_f3_bin16  bk7_h_f4_bin16  bk7_h_f5_bin16  bk7_h_f6_bin16  bk7_h_f7_bin16
       -----------------------------------------------------------------------------------------------
    */
    // with this organization, the LDS/shared memory bank is independent of the bin value
    // all threads within a quarter-wavefront (half-warp) will not have any bank conflict

    __local acc_type * gh_hist = (__local acc_type *)shared_array;
    // counter histogram
    // each bank: 8 * 16 * size_of(uint) = 0.5 KB
    // there are 8 banks used by 256 threads total 4 KB
    /* memory layout in cnt_hist:
       -----------------------------------------------
       bk0_c_f0_bin0   bk0_c_f1_bin0   bk0_c_f2_bin0   bk0_c_f3_bin0   bk0_c_f4_bin0   bk0_c_f5_bin0   bk0_c_f6_bin0   bk0_c_f7_bin0
       bk1_c_f0_bin0   bk1_c_f1_bin0   bk1_c_f2_bin0   bk1_c_f3_bin0   bk1_c_f4_bin0   bk1_c_f5_bin0   bk1_c_f6_bin0   bk1_c_f7_bin0
       bk2_c_f0_bin0   bk2_c_f1_bin0   bk2_c_f2_bin0   bk2_c_f3_bin0   bk2_c_f4_bin0   bk2_c_f5_bin0   bk2_c_f6_bin0   bk2_c_f7_bin0
       bk3_c_f0_bin0   bk3_c_f1_bin0   bk3_c_f2_bin0   bk3_c_f3_bin0   bk3_c_f4_bin0   bk3_c_f5_bin0   bk3_c_f6_bin0   bk3_c_f7_bin0
       bk4_c_f0_bin0   bk4_c_f1_bin0   bk4_c_f2_bin0   bk4_c_f3_bin0   bk4_c_f4_bin0   bk4_c_f5_bin0   bk4_c_f6_bin0   bk4_c_f7_bin0
       bk5_c_f0_bin0   bk5_c_f1_bin0   bk5_c_f2_bin0   bk5_c_f3_bin0   bk5_c_f4_bin0   bk5_c_f5_bin0   bk5_c_f6_bin0   bk5_c_f7_bin0
       bk6_c_f0_bin0   bk6_c_f1_bin0   bk6_c_f2_bin0   bk6_c_f3_bin0   bk6_c_f4_bin0   bk6_c_f5_bin0   bk6_c_f6_bin0   bk6_c_f7_bin0
       bk7_c_f0_bin0   bk7_c_f1_bin0   bk7_c_f2_bin0   bk7_c_f3_bin0   bk7_c_f4_bin0   bk7_c_f5_bin0   bk7_c_f6_bin0   bk7_c_f7_bin0
       ...
       bk0_c_f0_bin16  bk0_c_f1_bin16  bk0_c_f2_bin16  bk0_c_f3_bin16  bk0_c_f4_bin16  bk0_c_f5_bin16  bk0_c_f6_bin16  bk0_c_f7_bin0
       bk1_c_f0_bin16  bk1_c_f1_bin16  bk1_c_f2_bin16  bk1_c_f3_bin16  bk1_c_f4_bin16  bk1_c_f5_bin16  bk1_c_f6_bin16  bk1_c_f7_bin0
       bk2_c_f0_bin16  bk2_c_f1_bin16  bk2_c_f2_bin16  bk2_c_f3_bin16  bk2_c_f4_bin16  bk2_c_f5_bin16  bk2_c_f6_bin16  bk2_c_f7_bin0
       bk3_c_f0_bin16  bk3_c_f1_bin16  bk3_c_f2_bin16  bk3_c_f3_bin16  bk3_c_f4_bin16  bk3_c_f5_bin16  bk3_c_f6_bin16  bk3_c_f7_bin0
       bk4_c_f0_bin16  bk4_c_f1_bin16  bk4_c_f2_bin16  bk4_c_f3_bin16  bk4_c_f4_bin16  bk4_c_f5_bin16  bk4_c_f6_bin16  bk4_c_f7_bin0
       bk5_c_f0_bin16  bk5_c_f1_bin16  bk5_c_f2_bin16  bk5_c_f3_bin16  bk5_c_f4_bin16  bk5_c_f5_bin16  bk5_c_f6_bin16  bk5_c_f7_bin0
       bk6_c_f0_bin16  bk6_c_f1_bin16  bk6_c_f2_bin16  bk6_c_f3_bin16  bk6_c_f4_bin16  bk6_c_f5_bin16  bk6_c_f6_bin16  bk6_c_f7_bin0
       bk7_c_f0_bin16  bk7_c_f1_bin16  bk7_c_f2_bin16  bk7_c_f3_bin16  bk7_c_f4_bin16  bk7_c_f5_bin16  bk7_c_f6_bin16  bk7_c_f7_bin0
       -----------------------------------------------
    */
    #if CONST_HESSIAN == 1
    __local uint * cnt_hist = (__local uint *)(gh_hist + 2 * DWORD_FEATURES * NUM_BINS * NUM_BANKS);
    #endif

    // thread 0, 1, 2, 3, 4, 5, 6, 7 compute histograms for gradients first
    // thread 8, 9, 10, 11, 12, 13, 14, 15 compute histograms for Hessians first
    // etc.
    uchar is_hessian_first = (ltid >> LOG2_DWORD_FEATURES) & 1;
    // thread 0-15 write result to bank0, 16-31 to bank1, 32-47 to bank2, 48-63 to bank3, etc
    ushort bank = (ltid >> (LOG2_DWORD_FEATURES + 1)) & BANK_MASK;

    ushort group_feature = group_id >> POWER_FEATURE_WORKGROUPS;
    // each 2^POWER_FEATURE_WORKGROUPS workgroups process on one feature (compile-time constant)
    // feature_size is the number of examples per feature
    __global const uchar4* feature_data = feature_data_base + group_feature * feature_size;
    // size of threads that process this feature4
    const uint subglobal_size = lsize * (1 << POWER_FEATURE_WORKGROUPS);
    // equivalent thread ID in this subgroup for this feature4
    const uint subglobal_tid  = gtid - group_feature * subglobal_size;
    // extract feature mask, when a byte is set to 0, that feature is disabled
    #if ENABLE_ALL_FEATURES == 1
    // hopefully the compiler will propagate the constants and eliminate all branches
    uchar8 feature_mask = (uchar8)(0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff);
    #else
    uchar8 feature_mask = feature_masks[group_feature];
    #endif
    // exit if all features are masked
    if (!as_ulong(feature_mask)) {
        return;
    }

    // STAGE 1: read feature data, and gradient and hessian
    // first half of the threads read feature data from global memory
    // 4 features stored in a tuple MSB...(0, 1, 2, 3)...LSB
    // We will prefetch data into the "next" variable at the beginning of each iteration
    uchar4 feature4;
    uchar4 feature4_next;
    // offset used to rotate feature4 vector, & 0x7
    ushort offset = (ltid & DWORD_FEATURES_MASK);
    #if ENABLE_ALL_FEATURES == 0
    // rotate feature_mask to match the feature order of each thread
    feature_mask = as_uchar8(rotate(as_ulong(feature_mask), (ulong)offset*8));
    #endif
    // store gradient and hessian
    float stat1, stat2;
    float stat1_next, stat2_next;
    ushort bin, addr, addr2;
    data_size_t ind;
    data_size_t ind_next;
    stat1 = ordered_gradients[subglobal_tid];
    #if CONST_HESSIAN == 0
    stat2 = ordered_hessians[subglobal_tid];
    #endif
    #ifdef IGNORE_INDICES
    ind = subglobal_tid;
    #else
    ind = data_indices[subglobal_tid];
    #endif
    feature4 = feature_data[ind];

/* Makes MSVC happy with long string literal
)""
R""()
*/
    // there are 2^POWER_FEATURE_WORKGROUPS workgroups processing each feature4
    for (uint i = subglobal_tid; i < num_data; i += subglobal_size) {
        // prefetch the next iteration variables
        // we don't need boundary check because we have made the buffer larger
        stat1_next = ordered_gradients[i + subglobal_size];
        #if CONST_HESSIAN == 0
        stat2_next = ordered_hessians[i + subglobal_size];
        #endif
        #ifdef IGNORE_INDICES
        // we need to check to bounds here
        ind_next = i + subglobal_size < num_data ? i + subglobal_size : i;
        // start load next feature as early as possible
        feature4_next = feature_data[ind_next];
        #else
        ind_next = data_indices[i + subglobal_size];
        #endif
        #if CONST_HESSIAN == 0
        // swap gradient and hessian for threads 8, 9, 10, 11, 12, 13, 14, 15
        float tmp = stat1;
        stat1 = is_hessian_first ? stat2 : stat1;
        stat2 = is_hessian_first ? tmp   : stat2;
        // stat1 = select(stat1, stat2, is_hessian_first);
        // stat2 = select(stat2, tmp, is_hessian_first);
        #endif

        // STAGE 2: accumulate gradient and hessian
        offset = (ltid & DWORD_FEATURES_MASK);
        // printf("thread %x, %08x -> %08x", ltid, as_uint(feature4), rotate(as_uint(feature4), (uint)(offset * FEATURE_BITS)));
        feature4 = as_uchar4(rotate(as_uint(feature4), (uint)(offset * FEATURE_BITS)));
        if (feature_mask.s7) {
            bin = feature4.s3 >> 4;
            addr = bin * HG_BIN_MULT + bank * 2 * DWORD_FEATURES + is_hessian_first * DWORD_FEATURES + offset;
            addr2 = addr + DWORD_FEATURES - 2 * DWORD_FEATURES * is_hessian_first;
            // thread 0, 1, 2, 3, 4, 5, 6, 7 now process feature 0, 1, 2, 3, 4, 5, 6 ,7's gradients for example 0, 1, 2, 3, 4, 5, 6, 7
            // thread 8, 9, 10, 11, 12, 13, 14, 15 now process feature 0, 1, 2, 3, 4, 5, 6, 7's Hessians  for example 8, 9, 10, 11, 12, 13, 14, 15
            atomic_local_add_f(gh_hist + addr, stat1);
            // thread 0, 1, 2, 3, 4, 5, 6, 7 now process feature 0, 1, 2, 3, 4, 5, 6, 7's Hessians  for example 0, 1, 2, 3, 4, 5, 6, 7
            // thread 8, 9, 10, 11, 12, 13, 14, 15 now process feature 0, 1, 2, 3, 4, 5, 6, 7's gradients for example 8, 9, 10, 11, 12, 13, 14, 15
            #if CONST_HESSIAN == 0
            atomic_local_add_f(gh_hist + addr2, stat2);
            #endif
        }
        offset = (offset + 1) & DWORD_FEATURES_MASK;
        if (feature_mask.s6) {
            bin = feature4.s3 & 0xf;
            addr = bin * HG_BIN_MULT + bank * 2 * DWORD_FEATURES + is_hessian_first * DWORD_FEATURES + offset;
            addr2 = addr + DWORD_FEATURES - 2 * DWORD_FEATURES * is_hessian_first;
            // thread 0, 1, 2, 3, 4, 5, 6, 7 now process feature 1, 2, 3, 4, 5, 6 ,7, 0's gradients for example 0, 1, 2, 3, 4, 5, 6, 7
            // thread 8, 9, 10, 11, 12, 13, 14, 15 now process feature 1, 2, 3, 4, 5, 6, 7, 0's Hessians  for example 8, 9, 10, 11, 12, 13, 14, 15
            atomic_local_add_f(gh_hist + addr, stat1);
            // thread 0, 1, 2, 3, 4, 5, 6, 7 now process feature 1, 2, 3, 4, 5, 6, 7, 0's Hessians  for example 0, 1, 2, 3, 4, 5, 6, 7
            // thread 8, 9, 10, 11, 12, 13, 14, 15 now process feature 1, 2, 3, 4, 5, 6, 7, 0's gradients for example 8, 9, 10, 11, 12, 13, 14, 15
            #if CONST_HESSIAN == 0
            atomic_local_add_f(gh_hist + addr2, stat2);
            #endif
        }

        offset = (offset + 1) & DWORD_FEATURES_MASK;
        if (feature_mask.s5) {
            bin = feature4.s2 >> 4;
            addr = bin * HG_BIN_MULT + bank * 2 * DWORD_FEATURES + is_hessian_first * DWORD_FEATURES + offset;
            addr2 = addr + DWORD_FEATURES - 2 * DWORD_FEATURES * is_hessian_first;
            // thread 0, 1, 2, 3, 4, 5, 6, 7 now process feature 2, 3, 4, 5, 6, 7, 0, 1's gradients for example 0, 1, 2, 3, 4, 5, 6, 7
            // thread 8, 9, 10, 11, 12, 13, 14, 15 now process feature 2, 3, 4, 5, 6, 7, 0, 1's Hessians  for example 8, 9, 10, 11, 12, 13, 14, 15
            atomic_local_add_f(gh_hist + addr, stat1);
            // thread 0, 1, 2, 3, 4, 5, 6, 7 now process feature 2, 3, 4, 5, 6, 7, 0, 1's Hessians  for example 0, 1, 2, 3, 4, 5, 6, 7
            // thread 8, 9, 10, 11, 12, 13, 14, 15 now process feature 2, 3, 4, 5, 6, 7, 0, 1's gradients for example 8, 9, 10, 11, 12, 13, 14, 15
            #if CONST_HESSIAN == 0
            atomic_local_add_f(gh_hist + addr2, stat2);
            #endif
        }
        offset = (offset + 1) & DWORD_FEATURES_MASK;
        if (feature_mask.s4) {
            bin = feature4.s2 & 0xf;
            addr = bin * HG_BIN_MULT + bank * 2 * DWORD_FEATURES + is_hessian_first * DWORD_FEATURES + offset;
            addr2 = addr + DWORD_FEATURES - 2 * DWORD_FEATURES * is_hessian_first;
            // thread 0, 1, 2, 3, 4, 5, 6, 7 now process feature 3, 4, 5, 6, 7, 0, 1, 2's gradients for example 0, 1, 2, 3, 4, 5, 6, 7
            // thread 8, 9, 10, 11, 12, 13, 14, 15 now process feature 3, 4, 5, 6, 7, 0, 1, 2's Hessians  for example 8, 9, 10, 11, 12, 13, 14, 15
            atomic_local_add_f(gh_hist + addr, stat1);
            // thread 0, 1, 2, 3, 4, 5, 6, 7 now process feature 3, 4, 5, 6, 7, 0, 1, 2's Hessians  for example 0, 1, 2, 3, 4, 5, 6, 7
            // thread 8, 9, 10, 11, 12, 13, 14, 15 now process feature 3, 4, 5, 6, 7, 0, 1, 2's gradients for example 8, 9, 10, 11, 12, 13, 14, 15
            #if CONST_HESSIAN == 0
            atomic_local_add_f(gh_hist + addr2, stat2);
            #endif
        }


        // prefetch the next iteration variables
        // we don't need boundary check because if it is out of boundary, ind_next = 0
        #ifndef IGNORE_INDICES
        feature4_next = feature_data[ind_next];
        #endif

        offset = (offset + 1) & DWORD_FEATURES_MASK;
        if (feature_mask.s3) {
            bin = feature4.s1 >> 4;
            addr = bin * HG_BIN_MULT + bank * 2 * DWORD_FEATURES + is_hessian_first * DWORD_FEATURES + offset;
            addr2 = addr + DWORD_FEATURES - 2 * DWORD_FEATURES * is_hessian_first;
            // thread 0, 1, 2, 3, 4, 5, 6, 7 now process feature 4, 5, 6, 7, 0, 1, 2, 3's gradients for example 0, 1, 2, 3, 4, 5, 6, 7
            // thread 8, 9, 10, 11, 12, 13, 14, 15 now process feature 4, 5, 6, 7, 0, 1, 2, 3's hessians  for example 8, 9, 10, 11, 12, 13, 14, 15
            atomic_local_add_f(gh_hist + addr, stat1);
            // thread 0, 1, 2, 3, 4, 5, 6, 7 now process feature 4, 5, 6, 7, 0, 1, 2, 3's hessians  for example 0, 1, 2, 3, 4, 5, 6, 7
            // thread 8, 9, 10, 11, 12, 13, 14, 15 now process feature 4, 5, 6, 7, 0, 1, 2, 3's gradients for example 8, 9, 10, 11, 12, 13, 14, 15
            #if CONST_HESSIAN == 0
            atomic_local_add_f(gh_hist + addr2, stat2);
            #endif
        }
        offset = (offset + 1) & DWORD_FEATURES_MASK;
        if (feature_mask.s2) {
            bin = feature4.s1 & 0xf;
            addr = bin * HG_BIN_MULT + bank * 2 * DWORD_FEATURES + is_hessian_first * DWORD_FEATURES + offset;
            addr2 = addr + DWORD_FEATURES - 2 * DWORD_FEATURES * is_hessian_first;
            // thread 0, 1, 2, 3, 4, 5, 6, 7 now process feature 5, 6, 7, 0, 1, 2, 3, 4's gradients for example 0, 1, 2, 3, 4, 5, 6, 7
            // thread 8, 9, 10, 11, 12, 13, 14, 15 now process feature 5, 6, 7, 0, 1, 2, 3, 4's Hessians  for example 8, 9, 10, 11, 12, 13, 14, 15
            atomic_local_add_f(gh_hist + addr, stat1);
            // thread 0, 1, 2, 3, 4, 5, 6, 7 now process feature 5, 6, 7, 0, 1, 2, 3, 4's Hessians  for example 0, 1, 2, 3, 4, 5, 6, 7
            // thread 8, 9, 10, 11, 12, 13, 14, 15 now process feature 5, 6, 7, 0, 1, 2, 3, 4's gradients for example 8, 9, 10, 11, 12, 13, 14, 15
            #if CONST_HESSIAN == 0
            atomic_local_add_f(gh_hist + addr2, stat2);
            #endif
        }

        offset = (offset + 1) & DWORD_FEATURES_MASK;
        if (feature_mask.s1) {
            bin = feature4.s0 >> 4;
            addr = bin * HG_BIN_MULT + bank * 2 * DWORD_FEATURES + is_hessian_first * DWORD_FEATURES + offset;
            addr2 = addr + DWORD_FEATURES - 2 * DWORD_FEATURES * is_hessian_first;
            // thread 0, 1, 2, 3, 4, 5, 6, 7 now process feature 6, 7, 0, 1, 2, 3, 4, 5's gradients for example 0, 1, 2, 3, 4, 5, 6, 7
            // thread 8, 9, 10, 11, 12, 13, 14, 15 now process feature 6, 7, 0, 1, 2, 3, 4, 5's Hessians  for example 8, 9, 10, 11, 12, 13, 14, 15
            atomic_local_add_f(gh_hist + addr, stat1);
            // thread 0, 1, 2, 3, 4, 5, 6, 7 now process feature 6, 7, 0, 1, 2, 3, 4, 5's Hessians  for example 0, 1, 2, 3, 4, 5, 6, 7
            // thread 8, 9, 10, 11, 12, 13, 14, 15 now process feature 6, 7, 0, 1, 2, 3, 4, 5's gradients for example 8, 9, 10, 11, 12, 13, 14, 15
            #if CONST_HESSIAN == 0
            atomic_local_add_f(gh_hist + addr2, stat2);
            #endif
        }
        offset = (offset + 1) & DWORD_FEATURES_MASK;
        if (feature_mask.s0) {
            bin = feature4.s0 & 0xf;
            addr = bin * HG_BIN_MULT + bank * 2 * DWORD_FEATURES + is_hessian_first * DWORD_FEATURES + offset;
            addr2 = addr + DWORD_FEATURES - 2 * DWORD_FEATURES * is_hessian_first;
            // thread 0, 1, 2, 3, 4, 5, 6, 7 now process feature 7, 0, 1, 2, 3, 4, 5, 6's gradients for example 0, 1, 2, 3, 4, 5, 6, 7
            // thread 8, 9, 10, 11, 12, 13, 14, 15 now process feature 7, 0, 1, 2, 3, 4, 5, 6's Hessians  for example 8, 9, 10, 11, 12, 13, 14, 15
            atomic_local_add_f(gh_hist + addr, stat1);
            // thread 0, 1, 2, 3, 4, 5, 6, 7 now process feature 7, 0, 1, 2, 3, 4, 5, 6's Hessians  for example 0, 1, 2, 3, 4, 5, 6, 7
            // thread 8, 9, 10, 11, 12, 13, 14, 15 now process feature 7, 0, 1, 2, 3, 4, 5, 6's gradients for example 8, 9, 10, 11, 12, 13, 14, 15
            #if CONST_HESSIAN == 0
            atomic_local_add_f(gh_hist + addr2, stat2);
            #endif
        }
        #if CONST_HESSIAN == 1
        // STAGE 3: accumulate counter
        // there are 8 counters for 8 features
        // thread 0, 1, 2, 3, 4, 5, 6, 7 now process feature 0, 1, 2, 3, 4, 5, 6, 7's counts for example 0, 1, 2, 3, 4, 5, 6, 7
        offset = (ltid & DWORD_FEATURES_MASK);
        if (feature_mask.s7) {
            bin = feature4.s3 >> 4;
            addr = bin * CNT_BIN_MULT + bank * DWORD_FEATURES + offset;
            // printf("thread %x add counter %d feature %d (0)\n", ltid, bin, offset);
            atom_inc(cnt_hist + addr);
        }
        // thread 0, 1, 2, 3, 4, 5, 6, 7 now process feature 1, 2, 3, 4, 5, 6, 7, 0's counts for example 0, 1, 2, 3, 4, 5, 6, 7
        offset = (offset + 1) & DWORD_FEATURES_MASK;
        if (feature_mask.s6) {
            bin = feature4.s3 & 0xf;
            addr = bin * CNT_BIN_MULT + bank * DWORD_FEATURES + offset;
            // printf("thread %x add counter %d feature %d (1)\n", ltid, bin, offset);
            atom_inc(cnt_hist + addr);
        }
        // thread 0, 1, 2, 3, 4, 5, 6, 7 now process feature 2, 3, 4, 5, 6, 7, 0, 1's counts for example 0, 1, 2, 3, 4, 5, 6, 7
        offset = (offset + 1) & DWORD_FEATURES_MASK;
        if (feature_mask.s5) {
            bin = feature4.s2 >> 4;
            addr = bin * CNT_BIN_MULT + bank * DWORD_FEATURES + offset;
            // printf("thread %x add counter %d feature %d (2)\n", ltid, bin, offset);
            atom_inc(cnt_hist + addr);
        }
        // thread 0, 1, 2, 3, 4, 5, 6, 7 now process feature 3, 4, 5, 6, 7, 0, 1, 2's counts for example 0, 1, 2, 3, 4, 5, 6, 7
        offset = (offset + 1) & DWORD_FEATURES_MASK;
        if (feature_mask.s4) {
            bin = feature4.s2 & 0xf;
            addr = bin * CNT_BIN_MULT + bank * DWORD_FEATURES + offset;
            // printf("thread %x add counter %d feature %d (3)\n", ltid, bin, offset);
            atom_inc(cnt_hist + addr);
        }
        // thread 0, 1, 2, 3, 4, 5, 6, 7 now process feature 4, 5, 6, 7, 0, 1, 2, 3's counts for example 0, 1, 2, 3, 4, 5, 6, 7
        offset = (offset + 1) & DWORD_FEATURES_MASK;
        if (feature_mask.s3) {
            bin = feature4.s1 >> 4;
            addr = bin * CNT_BIN_MULT + bank * DWORD_FEATURES + offset;
            // printf("thread %x add counter %d feature %d (4)\n", ltid, bin, offset);
            atom_inc(cnt_hist + addr);
        }
        // thread 0, 1, 2, 3, 4, 5, 6, 7 now process feature 5, 6, 7, 0, 1, 2, 3, 4's counts for example 0, 1, 2, 3, 4, 5, 6, 7
        offset = (offset + 1) & DWORD_FEATURES_MASK;
        if (feature_mask.s2) {
            bin = feature4.s1 & 0xf;
            addr = bin * CNT_BIN_MULT + bank * DWORD_FEATURES + offset;
            // printf("thread %x add counter %d feature %d (5)\n", ltid, bin, offset);
            atom_inc(cnt_hist + addr);
        }
        // thread 0, 1, 2, 3, 4, 5, 6, 7 now process feature 6, 7, 0, 1, 2, 3, 4, 5's counts for example 0, 1, 2, 3, 4, 5, 6, 7
        offset = (offset + 1) & DWORD_FEATURES_MASK;
        if (feature_mask.s1) {
            bin = feature4.s0 >> 4;
            addr = bin * CNT_BIN_MULT + bank * DWORD_FEATURES + offset;
            // printf("thread %x add counter %d feature %d (6)\n", ltid, bin, offset);
            atom_inc(cnt_hist + addr);
        }
        // thread 0, 1, 2, 3, 4, 5, 6, 7 now process feature 7, 0, 1, 2, 3, 4, 5, 6's counts for example 0, 1, 2, 3, 4, 5, 6, 7
        offset = (offset + 1) & DWORD_FEATURES_MASK;
        if (feature_mask.s0) {
            bin = feature4.s0 & 0xf;
            addr = bin * CNT_BIN_MULT + bank * DWORD_FEATURES + offset;
            // printf("thread %x add counter %d feature %d (7)\n", ltid, bin, offset);
            atom_inc(cnt_hist + addr);
        }
        #endif
        stat1 = stat1_next;
        stat2 = stat2_next;
        feature4 = feature4_next;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

/* Makes MSVC happy with long string literal
)""
R""()
*/

    #if ENABLE_ALL_FEATURES == 0
    // restore feature_mask
    feature_mask = feature_masks[group_feature];
    #endif

    // now reduce the 4 banks of subhistograms into 1
    acc_type stat_val = 0.0f;
    uint cnt_val = 0;
    // 256 threads, working on 8 features and 16 bins, 2 stats
    // so each thread has an independent feature/bin/stat to work on.
    const ushort feature_id = ltid & DWORD_FEATURES_MASK; // bits 0 - 2 of ltid, range 0 - 7
    ushort bin_id = ltid >> (LOG2_DWORD_FEATURES + 1); // bits 3 is is_hessian_first; bits 4 - 7 range 0 - 16 is bin ID
    offset = (ltid >> (LOG2_DWORD_FEATURES + 1)) & BANK_MASK; // helps avoid LDS bank conflicts
    for (int i = 0; i < NUM_BANKS; ++i) {
        ushort bank_id = (i + offset) & BANK_MASK;
        stat_val += gh_hist[bin_id * HG_BIN_MULT + bank_id * 2 * DWORD_FEATURES + is_hessian_first * DWORD_FEATURES + feature_id];
    }
    #if CONST_HESSIAN == 1
    if (ltid < LOCAL_SIZE_0 / 2) {
        // first 128 threads accumulate the 8 * 16 = 128 counter values
        bin_id = ltid >> LOG2_DWORD_FEATURES; // bits 3 - 6 range 0 - 16 is bin ID
        offset = (ltid >> LOG2_DWORD_FEATURES) & BANK_MASK; // helps avoid LDS bank conflicts
        for (int i = 0; i < NUM_BANKS; ++i) {
            ushort bank_id = (i + offset) & BANK_MASK;
            cnt_val += cnt_hist[bin_id * CNT_BIN_MULT + bank_id * DWORD_FEATURES + feature_id];
        }
    }
    #endif

    // now thread 0 - 7  holds feature 0 - 7's gradient for bin 0 and counter bin 0
    // now thread 8 - 15 holds feature 0 - 7's hessian  for bin 0 and counter bin 1
    // now thread 16- 23 holds feature 0 - 7's gradient for bin 1 and counter bin 2
    // now thread 24- 31 holds feature 0 - 7's hessian  for bin 1 and counter bin 3
    // etc,

#if CONST_HESSIAN == 1
    // Combine the two banks into one, and fill the Hessians with counter value * hessian constant
    barrier(CLK_LOCAL_MEM_FENCE);
    gh_hist[ltid] = stat_val;
    if (ltid < LOCAL_SIZE_0 / 2) {
        cnt_hist[ltid] = cnt_val;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (is_hessian_first) {
        // these are the Hessians
        // thread 8 - 15 read counters stored by thread 0 - 7
        // thread 24- 31 read counters stored by thread 8 - 15
        // thread 40- 47 read counters stored by thread 16- 23, etc
        stat_val = const_hessian *
                   cnt_hist[((ltid - DWORD_FEATURES) >> (LOG2_DWORD_FEATURES + 1)) * DWORD_FEATURES + (ltid & DWORD_FEATURES_MASK)];
    }
    else {
        // these are the gradients
        // thread 0 - 7  read gradients stored by thread 8 - 15
        // thread 16- 23 read gradients stored by thread 24- 31
        // thread 32- 39 read gradients stored by thread 40- 47, etc
        stat_val += gh_hist[ltid + DWORD_FEATURES];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#endif

    // write to output
    // write gradients and Hessians histogram for all 4 features
    // output data in linear order for further reduction
    // output size = 4 (features) * 2 (counters) * 64 (bins) * sizeof(float)
    /* memory layout of output:
       g_f0_bin0   g_f1_bin0   g_f2_bin0   g_f3_bin0   g_f4_bin0   g_f5_bin0   g_f6_bin0   g_f7_bin0
       h_f0_bin0   h_f1_bin0   h_f2_bin0   h_f3_bin0   h_f4_bin0   h_f5_bin0   h_f6_bin0   h_f7_bin0
       g_f0_bin1   g_f1_bin1   g_f2_bin1   g_f3_bin1   g_f4_bin1   g_f5_bin1   g_f6_bin1   g_f7_bin1
       h_f0_bin1   h_f1_bin1   h_f2_bin1   h_f3_bin1   h_f4_bin1   h_f5_bin1   h_f6_bin1   h_f7_bin1
       ...
       ...
       g_f0_bin16  g_f1_bin16  g_f2_bin16  g_f3_bin16  g_f4_bin16  g_f5_bin16  g_f6_bin16  g_f7_bin16
       h_f0_bin16  h_f1_bin16  h_f2_bin16  h_f3_bin16  h_f4_bin16  h_f5_bin16  h_f6_bin16  h_f7_bin16
       c_f0_bin0   c_f1_bin0   c_f2_bin0   c_f3_bin0   c_f4_bin0   c_f5_bin0   c_f6_bin0   c_f7_bin0
       c_f0_bin1   c_f1_bin1   c_f2_bin1   c_f3_bin1   c_f4_bin1   c_f5_bin1   c_f6_bin1   c_f7_bin1
       ...
       c_f0_bin16  c_f1_bin16  c_f2_bin16  c_f3_bin16  c_f4_bin16  c_f5_bin16  c_f6_bin16  c_f7_bin16
    */
    // if there is only one workgroup processing this feature4, don't even need to write
    uint feature4_id = (group_id >> POWER_FEATURE_WORKGROUPS);
    #if POWER_FEATURE_WORKGROUPS != 0
    __global acc_type * restrict output = (__global acc_type * restrict)output_buf + group_id * DWORD_FEATURES * 2 * NUM_BINS;
    // if g_val and h_val are double, they are converted to float here
    // write gradients and Hessians for 8 features
    output[0 * DWORD_FEATURES * NUM_BINS + ltid] = stat_val;
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    mem_fence(CLK_GLOBAL_MEM_FENCE);
    // To avoid the cost of an extra reducing kernel, we have to deal with some
    // gray area in OpenCL. We want the last work group that process this feature to
    // make the final reduction, and other threads will just quit.
    // This requires that the results written by other workgroups available to the
    // last workgroup (memory consistency)
    #if NVIDIA == 1
    // this is equivalent to CUDA __threadfence();
    // ensure the writes above goes to main memory and other workgroups can see it
    asm volatile("{\n\tmembar.gl;\n\t}\n\t" :::"memory");
    #else
    // FIXME: how to do the above on AMD GPUs??
    // GCN ISA says that the all writes will bypass L1 cache (write through),
    // however when the last thread is reading sub-histogram data we have to
    // make sure that no part of data is modified in local L1 cache of other workgroups.
    // Otherwise reading can be a problem (atomic operations to get consistency).
    // But in our case, the sub-histogram of this workgroup cannot be in the cache
    // of another workgroup, so the following trick will work just fine.
    #endif
    // Now, we want one workgroup to do the final reduction.
    // Other workgroups processing the same feature quit.
    // The is done by using an global atomic counter.
    // On AMD GPUs ideally this should be done in GDS,
    // but currently there is no easy way to access it via OpenCL.
    __local uint * counter_val = (__local uint *)(gh_hist + 2 * DWORD_FEATURES * NUM_BINS * NUM_BANKS);
    if (ltid == 0) {
        // all workgroups processing the same feature add this counter
        *counter_val = atom_inc(sync_counters + feature4_id);
    }
    // make sure everyone in this workgroup is here
    barrier(CLK_LOCAL_MEM_FENCE);
    // everyone in this workgroup: if we are the last workgroup, then do reduction!
    if (*counter_val == (1 << POWER_FEATURE_WORKGROUPS) - 1) {
        if (ltid == 0) {
            // printf("workgroup %d start reduction!\n", group_id);
            // printf("feature_data[0] = %d %d %d %d", feature_data[0].s0, feature_data[0].s1, feature_data[0].s2, feature_data[0].s3);
            // clear the sync counter for using it next time
            sync_counters[feature4_id] = 0;
        }
    #else
    // only 1 work group, no need to increase counter
    // the reduction will become a simple copy
    if (1) {
        barrier(CLK_LOCAL_MEM_FENCE);
    #endif
        // locate our feature4's block in output memory
        uint output_offset = (feature4_id << POWER_FEATURE_WORKGROUPS);
        __global acc_type const * restrict feature4_subhists =
                 (__global acc_type *)output_buf + output_offset * DWORD_FEATURES * 2 * NUM_BINS;
        // skip reading the data already in local memory
        uint skip_id = group_id ^ output_offset;
        // locate output histogram location for this feature4
        __global acc_type* restrict hist_buf = hist_buf_base + feature4_id * DWORD_FEATURES * 2 * NUM_BINS;
        within_kernel_reduction16x8(feature_mask, feature4_subhists, skip_id, stat_val,
                                    1 << POWER_FEATURE_WORKGROUPS, hist_buf, (__local acc_type *)shared_array);
    }
}

// The following line ends the string literal, adds an extra #endif at the end
// )"" "\n#endif"
#endif
