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

#ifndef _HISTOGRAM_256_KERNEL_
#define _HISTOGRAM_256_KERNEL_

#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

// use double precision or not
#ifndef USE_DP_FLOAT
#define USE_DP_FLOAT 0
#endif
// ignore hessian, and use the local memory for hessian as an additional bank for gradient
#ifndef CONST_HESSIAN
#define CONST_HESSIAN 0
#endif

#define LOCAL_SIZE_0 256
#define NUM_BINS 256
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
#define LOCAL_MEM_SIZE (4 * (sizeof(uint) + 2 * sizeof(acc_type)) * NUM_BINS)

// unroll the atomic operation for a few times. Takes more code space,
// but compiler can generate better code for faster atomics.
#define UNROLL_ATOMIC 1

// Options passed by compiler at run time:
// IGNORE_INDICES will be set when the kernel does not
// #define IGNORE_INDICES
// #define POWER_FEATURE_WORKGROUPS 10

// detect Nvidia platforms
#ifdef cl_nv_pragma_unroll
#define NVIDIA 1
#endif

// use all features and do not use feature mask
#ifndef ENABLE_ALL_FEATURES
#define ENABLE_ALL_FEATURES 1
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
    ATOMIC_FADD_SUB2
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
// this function will be called by histogram256
// we have one sub-histogram of one feature in local memory, and need to read others
void within_kernel_reduction256x4(uchar4 feature_mask,
                           __global const acc_type* restrict feature4_sub_hist,
                           const uint skip_id,
                           const uint old_val_f0_cont_bin0,
                           const ushort num_sub_hist,
                           __global acc_type* restrict output_buf,
                           __local acc_type* restrict local_hist) {
    const ushort ltid = get_local_id(0);
    const ushort lsize = LOCAL_SIZE_0;
    // initialize register counters from our local memory
    // TODO: try to avoid bank conflict here
    acc_type f0_grad_bin = local_hist[ltid * 8];
    acc_type f1_grad_bin = local_hist[ltid * 8 + 1];
    acc_type f2_grad_bin = local_hist[ltid * 8 + 2];
    acc_type f3_grad_bin = local_hist[ltid * 8 + 3];
    acc_type f0_hess_bin = local_hist[ltid * 8 + 4];
    acc_type f1_hess_bin = local_hist[ltid * 8 + 5];
    acc_type f2_hess_bin = local_hist[ltid * 8 + 6];
    acc_type f3_hess_bin = local_hist[ltid * 8 + 7];
    ushort i;
    // printf("%d-pre(skip %d): %f %f %f %f %f %f %f %f %d %d %d %d", ltid, skip_id, f0_grad_bin, f1_grad_bin, f2_grad_bin, f3_grad_bin, f0_hess_bin, f1_hess_bin, f2_hess_bin, f3_hess_bin, f0_cont_bin, f1_cont_bin, f2_cont_bin, f3_cont_bin);
#if POWER_FEATURE_WORKGROUPS != 0
    // add all sub-histograms for 4 features
    __global const acc_type* restrict p = feature4_sub_hist + ltid;
    for (i = 0; i < skip_id; ++i) {
        if (feature_mask.s3) {
            f0_grad_bin += *p;          p += NUM_BINS;
            f0_hess_bin += *p;          p += NUM_BINS;
        }
        else {
            p += 2 * NUM_BINS;
        }
        if (feature_mask.s2) {
            f1_grad_bin += *p;          p += NUM_BINS;
            f1_hess_bin += *p;          p += NUM_BINS;
        }
        else {
            p += 2 * NUM_BINS;
        }
        if (feature_mask.s1) {
            f2_grad_bin += *p;          p += NUM_BINS;
            f2_hess_bin += *p;          p += NUM_BINS;
        }
        else {
            p += 2 * NUM_BINS;
        }
        if (feature_mask.s0) {
            f3_grad_bin += *p;          p += NUM_BINS;
            f3_hess_bin += *p;          p += NUM_BINS;
        }
        else {
            p += 2 * NUM_BINS;
        }
    }
    // skip the counters we already have
    p += 2 * 4 * NUM_BINS;
    for (i = i + 1; i < num_sub_hist; ++i) {
        if (feature_mask.s3) {
            f0_grad_bin += *p;          p += NUM_BINS;
            f0_hess_bin += *p;          p += NUM_BINS;
        }
        else {
            p += 2 * NUM_BINS;
        }
        if (feature_mask.s2) {
            f1_grad_bin += *p;          p += NUM_BINS;
            f1_hess_bin += *p;          p += NUM_BINS;
        }
        else {
            p += 2 * NUM_BINS;
        }
        if (feature_mask.s1) {
            f2_grad_bin += *p;          p += NUM_BINS;
            f2_hess_bin += *p;          p += NUM_BINS;
        }
        else {
            p += 2 * NUM_BINS;
        }
        if (feature_mask.s0) {
            f3_grad_bin += *p;          p += NUM_BINS;
            f3_hess_bin += *p;          p += NUM_BINS;
        }
        else {
            p += 2 * NUM_BINS;
        }
    }
    // printf("%d-aft: %f %f %f %f %f %f %f %f %d %d %d %d", ltid, f0_grad_bin, f1_grad_bin, f2_grad_bin, f3_grad_bin, f0_hess_bin, f1_hess_bin, f2_hess_bin, f3_hess_bin, f0_cont_bin, f1_cont_bin, f2_cont_bin, f3_cont_bin);
    #endif
    // now overwrite the local_hist for final reduction and output
    barrier(CLK_LOCAL_MEM_FENCE);
    #if USE_DP_FLOAT == 0
    // reverse the f3...f0 order to match the real order
    local_hist[0 * 2 * NUM_BINS + ltid * 2 + 0] = f3_grad_bin;
    local_hist[0 * 2 * NUM_BINS + ltid * 2 + 1] = f3_hess_bin;
    local_hist[1 * 2 * NUM_BINS + ltid * 2 + 0] = f2_grad_bin;
    local_hist[1 * 2 * NUM_BINS + ltid * 2 + 1] = f2_hess_bin;
    local_hist[2 * 2 * NUM_BINS + ltid * 2 + 0] = f1_grad_bin;
    local_hist[2 * 2 * NUM_BINS + ltid * 2 + 1] = f1_hess_bin;
    local_hist[3 * 2 * NUM_BINS + ltid * 2 + 0] = f0_grad_bin;
    local_hist[3 * 2 * NUM_BINS + ltid * 2 + 1] = f0_hess_bin;
    barrier(CLK_LOCAL_MEM_FENCE);
    /*
    for (ushort i = ltid; i < 4 * 3 * NUM_BINS; i += lsize) {
        output_buf[i] = local_hist[i];
    }
    */
    i = ltid;
    if (feature_mask.s0) {
        output_buf[i] = local_hist[i];
        output_buf[i + NUM_BINS] = local_hist[i + NUM_BINS];
    }
    i += 1 * 2 * NUM_BINS;
    if (feature_mask.s1) {
        output_buf[i] = local_hist[i];
        output_buf[i + NUM_BINS] = local_hist[i + NUM_BINS];
    }
    i += 1 * 2 * NUM_BINS;
    if (feature_mask.s2) {
        output_buf[i] = local_hist[i];
        output_buf[i + NUM_BINS] = local_hist[i + NUM_BINS];
    }
    i += 1 * 2 * NUM_BINS;
    if (feature_mask.s3 && i < 4 * 2 * NUM_BINS) {
        output_buf[i] = local_hist[i];
        output_buf[i + NUM_BINS] = local_hist[i + NUM_BINS];
    }
    #else
    // when double precision is used, we need to write twice, because local memory size is not enough
    local_hist[0 * 2 * NUM_BINS + ltid * 2 + 0] = f3_grad_bin;
    local_hist[0 * 2 * NUM_BINS + ltid * 2 + 1] = f3_hess_bin;
    local_hist[1 * 2 * NUM_BINS + ltid * 2 + 0] = f2_grad_bin;
    local_hist[1 * 2 * NUM_BINS + ltid * 2 + 1] = f2_hess_bin;
    barrier(CLK_LOCAL_MEM_FENCE);
    /*
    for (ushort i = ltid; i < 2 * 3 * NUM_BINS; i += lsize) {
        output_buf[i] = local_hist[i];
    }
    */
    i = ltid;
    if (feature_mask.s0) {
        output_buf[i] = local_hist[i];
        output_buf[i + NUM_BINS] = local_hist[i + NUM_BINS];
    }
    i += 1 * 2 * NUM_BINS;
    if (feature_mask.s1) {
        output_buf[i] = local_hist[i];
        output_buf[i + NUM_BINS] = local_hist[i + NUM_BINS];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    local_hist[0 * 2 * NUM_BINS + ltid * 2 + 0] = f1_grad_bin;
    local_hist[0 * 2 * NUM_BINS + ltid * 2 + 1] = f1_hess_bin;
    local_hist[1 * 2 * NUM_BINS + ltid * 2 + 0] = f0_grad_bin;
    local_hist[1 * 2 * NUM_BINS + ltid * 2 + 1] = f0_hess_bin;
    barrier(CLK_LOCAL_MEM_FENCE);
    /*
    for (ushort i = ltid; i < 2 * 3 * NUM_BINS; i += lsize) {
        output_buf[i + 2 * 3 * NUM_BINS] = local_hist[i];
    }
    */
    i = ltid;
    if (feature_mask.s2) {
        output_buf[i + 2 * 2 * NUM_BINS] = local_hist[i];
        output_buf[i + 2 * 2 * NUM_BINS + NUM_BINS] = local_hist[i + NUM_BINS];
    }
    i += 1 * 2 * NUM_BINS;
    if (feature_mask.s3) {
        output_buf[i + 2 * 2 * NUM_BINS] = local_hist[i];
        output_buf[i + 2 * 2 * NUM_BINS + NUM_BINS] = local_hist[i + NUM_BINS];
    }
    #endif
}

/* Makes MSVC happy with long string literal
)""
R""()
*/
__attribute__((reqd_work_group_size(LOCAL_SIZE_0, 1, 1)))
#if USE_CONSTANT_BUF == 1
__kernel void histogram256(__global const uchar4* restrict feature_data_base,
                      __constant const uchar4* restrict feature_masks __attribute__((max_constant_size(65536))),
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
__kernel void histogram256(__global const uchar4* feature_data_base,
                      __constant const uchar4* restrict feature_masks __attribute__((max_constant_size(65536))),
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
    // total size: 2 * 4 * 256 * size_of(float) = 8 KB
    // organization: each feature/grad/hessian is at a different bank,
    //               as independent of the feature value as possible
    __local acc_type * gh_hist = (__local acc_type *)shared_array;
    // counter histogram
    // total size: 4 * 256 * size_of(uint) = 4 KB
    #if CONST_HESSIAN == 1
    __local uint * cnt_hist = (__local uint *)(gh_hist + 2 * 4 * NUM_BINS);
    #endif

    // thread 0, 1, 2, 3 compute histograms for gradients first
    // thread 4, 5, 6, 7 compute histograms for Hessians  first
    // etc.
    uchar is_hessian_first = (ltid >> 2) & 1;

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
    uchar4 feature_mask = (uchar4)(0xff, 0xff, 0xff, 0xff);
    #else
    uchar4 feature_mask = feature_masks[group_feature];
    #endif
    // exit if all features are masked
    if (!as_uint(feature_mask)) {
        return;
    }

    // STAGE 1: read feature data, and gradient and hessian
    // first half of the threads read feature data from global memory
    // 4 features stored in a tuple MSB...(0, 1, 2, 3)...LSB
    // We will prefetch data into the "next" variable at the beginning of each iteration
    uchar4 feature4;
    uchar4 feature4_next;
    uchar4 feature4_prev;
    // offset used to rotate feature4 vector
    ushort offset = (ltid & 0x3);
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
    feature4_prev = feature4;
    feature4_prev = as_uchar4(rotate(as_uint(feature4_prev), (uint)offset*8));
    #if ENABLE_ALL_FEATURES == 0
    // rotate feature_mask to match the feature order of each thread
    feature_mask = as_uchar4(rotate(as_uint(feature_mask), (uint)offset*8));
    #endif
    acc_type s3_stat1 = 0.0f, s3_stat2 = 0.0f;
    acc_type s2_stat1 = 0.0f, s2_stat2 = 0.0f;
    acc_type s1_stat1 = 0.0f, s1_stat2 = 0.0f;
    acc_type s0_stat1 = 0.0f, s0_stat2 = 0.0f;


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
        // swap gradient and hessian for threads 4, 5, 6, 7
        float tmp = stat1;
        stat1 = is_hessian_first ? stat2 : stat1;
        stat2 = is_hessian_first ? tmp   : stat2;
        // stat1 = select(stat1, stat2, is_hessian_first);
        // stat2 = select(stat2, tmp, is_hessian_first);
        #endif

        // STAGE 2: accumulate gradient and hessian
        offset = (ltid & 0x3);
        feature4 = as_uchar4(rotate(as_uint(feature4), (uint)offset*8));
        bin = feature4.s3;
        if ((bin != feature4_prev.s3) && feature_mask.s3) {
            // printf("%3d (%4d): writing s3 %d %d offset %d", ltid, i, bin, feature4_prev.s3, offset);
            bin = feature4_prev.s3;
            feature4_prev.s3 = feature4.s3;
            addr = bin * 8 + is_hessian_first * 4 + offset;
            addr2 = addr + 4 - 8 * is_hessian_first;
            atomic_local_add_f(gh_hist + addr, s3_stat1);
            // thread 0, 1, 2, 3 now process feature 0, 1, 2, 3's gradients for example 0, 1, 2, 3
            // thread 4, 5, 6, 7 now process feature 0, 1, 2, 3's Hessians  for example 4, 5, 6, 7
            #if CONST_HESSIAN == 0
            atomic_local_add_f(gh_hist + addr2, s3_stat2);
            #endif
            // thread 0, 1, 2, 3 now process feature 0, 1, 2, 3's Hessians  for example 0, 1, 2, 3
            // thread 4, 5, 6, 7 now process feature 0, 1, 2, 3's gradients for example 4, 5, 6, 7
            s3_stat1 = stat1;
            s3_stat2 = stat2;
        }
        else {
            // printf("%3d (%4d): acc s3 %d", ltid, i, bin);
            s3_stat1 += stat1;
            s3_stat2 += stat2;
        }

        bin = feature4.s2;
        offset = (offset + 1) & 0x3;
        if ((bin != feature4_prev.s2) && feature_mask.s2) {
            // printf("%3d (%4d): writing s2 %d %d feature %d", ltid, i, bin, feature4_prev.s2, offset);
            bin = feature4_prev.s2;
            feature4_prev.s2 = feature4.s2;
            addr = bin * 8 + is_hessian_first * 4 + offset;
            addr2 = addr + 4 - 8 * is_hessian_first;
            atomic_local_add_f(gh_hist + addr, s2_stat1);
            // thread 0, 1, 2, 3 now process feature 1, 2, 3, 0's gradients for example 0, 1, 2, 3
            // thread 4, 5, 6, 7 now process feature 1, 2, 3, 0's Hessians  for example 4, 5, 6, 7
            #if CONST_HESSIAN == 0
            atomic_local_add_f(gh_hist + addr2, s2_stat2);
            #endif
            // thread 0, 1, 2, 3 now process feature 1, 2, 3, 0's Hessians  for example 0, 1, 2, 3
            // thread 4, 5, 6, 7 now process feature 1, 2, 3, 0's gradients for example 4, 5, 6, 7
            s2_stat1 = stat1;
            s2_stat2 = stat2;
        }
        else {
            // printf("%3d (%4d): acc s2 %d", ltid, i, bin);
            s2_stat1 += stat1;
            s2_stat2 += stat2;
        }


        // prefetch the next iteration variables
        // we don't need boundary check because if it is out of boundary, ind_next = 0
        #ifndef IGNORE_INDICES
        feature4_next = feature_data[ind_next];
        #endif

        bin = feature4.s1;
        offset = (offset + 1) & 0x3;
        if ((bin != feature4_prev.s1) && feature_mask.s1) {
            // printf("%3d (%4d): writing s1 %d %d feature %d", ltid, i, bin, feature4_prev.s1, offset);
            bin = feature4_prev.s1;
            feature4_prev.s1 = feature4.s1;
            addr = bin * 8 + is_hessian_first * 4 + offset;
            addr2 = addr + 4 - 8 * is_hessian_first;
            atomic_local_add_f(gh_hist + addr, s1_stat1);
            // thread 0, 1, 2, 3 now process feature 2, 3, 0, 1's gradients for example 0, 1, 2, 3
            // thread 4, 5, 6, 7 now process feature 2, 3, 0, 1's Hessians  for example 4, 5, 6, 7
            #if CONST_HESSIAN == 0
            atomic_local_add_f(gh_hist + addr2, s1_stat2);
            #endif
            // thread 0, 1, 2, 3 now process feature 2, 3, 0, 1's Hessians  for example 0, 1, 2, 3
            // thread 4, 5, 6, 7 now process feature 2, 3, 0, 1's gradients for example 4, 5, 6, 7
            s1_stat1 = stat1;
            s1_stat2 = stat2;
        }
        else {
            // printf("%3d (%4d): acc s1 %d", ltid, i, bin);
            s1_stat1 += stat1;
            s1_stat2 += stat2;
        }

        bin = feature4.s0;
        offset = (offset + 1) & 0x3;
        if ((bin != feature4_prev.s0) && feature_mask.s0) {
            // printf("%3d (%4d): writing s0 %d %d feature %d", ltid, i, bin, feature4_prev.s0, offset);
            bin = feature4_prev.s0;
            feature4_prev.s0 = feature4.s0;
            addr = bin * 8 + is_hessian_first * 4 + offset;
            addr2 = addr + 4 - 8 * is_hessian_first;
            atomic_local_add_f(gh_hist + addr, s0_stat1);
            // thread 0, 1, 2, 3 now process feature 3, 0, 1, 2's gradients for example 0, 1, 2, 3
            // thread 4, 5, 6, 7 now process feature 3, 0, 1, 2's Hessians  for example 4, 5, 6, 7
            #if CONST_HESSIAN == 0
            atomic_local_add_f(gh_hist + addr2, s0_stat2);
            #endif
            // thread 0, 1, 2, 3 now process feature 3, 0, 1, 2's Hessians  for example 0, 1, 2, 3
            // thread 4, 5, 6, 7 now process feature 3, 0, 1, 2's gradients for example 4, 5, 6, 7
            s0_stat1 = stat1;
            s0_stat2 = stat2;
        }
        else {
            // printf("%3d (%4d): acc s0 %d", ltid, i, bin);
            s0_stat1 += stat1;
            s0_stat2 += stat2;
        }
        #if CONST_HESSIAN == 1
        // STAGE 3: accumulate counter
        // there are 4 counters for 4 features
        // thread 0, 1, 2, 3 now process feature 0, 1, 2, 3's counts for example 0, 1, 2, 3
        offset = (ltid & 0x3);
        if (feature_mask.s3) {
            bin = feature4.s3;
            addr = bin * 4 + offset;
            atom_inc(cnt_hist + addr);
        }
        // thread 0, 1, 2, 3 now process feature 1, 2, 3, 0's counts for example 0, 1, 2, 3
        offset = (offset + 1) & 0x3;
        if (feature_mask.s2) {
            bin = feature4.s2;
            addr = bin * 4 + offset;
            atom_inc(cnt_hist + addr);
        }
        // thread 0, 1, 2, 3 now process feature 2, 3, 0, 1's counts for example 0, 1, 2, 3
        offset = (offset + 1) & 0x3;
        if (feature_mask.s1) {
            bin = feature4.s1;
            addr = bin * 4 + offset;
            atom_inc(cnt_hist + addr);
        }
        // thread 0, 1, 2, 3 now process feature 3, 0, 1, 2's counts for example 0, 1, 2, 3
        offset = (offset + 1) & 0x3;
        if (feature_mask.s0) {
            bin = feature4.s0;
            addr = bin * 4 + offset;
            atom_inc(cnt_hist + addr);
        }
        #endif
        stat1 = stat1_next;
        stat2 = stat2_next;
        feature4 = feature4_next;
    }

    bin = feature4_prev.s3;
    offset = (ltid & 0x3);
    addr = bin * 8 + is_hessian_first * 4 + offset;
    addr2 = addr + 4 - 8 * is_hessian_first;
    atomic_local_add_f(gh_hist + addr, s3_stat1);
    #if CONST_HESSIAN == 0
    atomic_local_add_f(gh_hist + addr2, s3_stat2);
    #endif

    bin = feature4_prev.s2;
    offset = (offset + 1) & 0x3;
    addr = bin * 8 + is_hessian_first * 4 + offset;
    addr2 = addr + 4 - 8 * is_hessian_first;
    atomic_local_add_f(gh_hist + addr, s2_stat1);
    #if CONST_HESSIAN == 0
    atomic_local_add_f(gh_hist + addr2, s2_stat2);
    #endif

    bin = feature4_prev.s1;
    offset = (offset + 1) & 0x3;
    addr = bin * 8 + is_hessian_first * 4 + offset;
    addr2 = addr + 4 - 8 * is_hessian_first;
    atomic_local_add_f(gh_hist + addr, s1_stat1);
    #if CONST_HESSIAN == 0
    atomic_local_add_f(gh_hist + addr2, s1_stat2);
    #endif

    bin = feature4_prev.s0;
    offset = (offset + 1) & 0x3;
    addr = bin * 8 + is_hessian_first * 4 + offset;
    addr2 = addr + 4 - 8 * is_hessian_first;
    atomic_local_add_f(gh_hist + addr, s0_stat1);
    #if CONST_HESSIAN == 0
    atomic_local_add_f(gh_hist + addr2, s0_stat2);
    #endif
    barrier(CLK_LOCAL_MEM_FENCE);

/* Makes MSVC happy with long string literal
)""
R""()
*/
    #if ENABLE_ALL_FEATURES == 0
    // restore feature_mask
    feature_mask = feature_masks[group_feature];
    #endif

    #if CONST_HESSIAN == 1
    barrier(CLK_LOCAL_MEM_FENCE);
    // make a final reduction
    offset = ltid & 0x3; // helps avoid LDS bank conflicts
    gh_hist[ltid * 8 + offset] += gh_hist[ltid * 8 + offset + 4];
    gh_hist[ltid * 8 + offset + 4] = const_hessian * cnt_hist[ltid * 4 + offset];
    offset = (offset + 1) & 0x3;
    gh_hist[ltid * 8 + offset] += gh_hist[ltid * 8 + offset + 4];
    gh_hist[ltid * 8 + offset + 4] = const_hessian * cnt_hist[ltid * 4 + offset];
    offset = (offset + 1) & 0x3;
    gh_hist[ltid * 8 + offset] += gh_hist[ltid * 8 + offset + 4];
    gh_hist[ltid * 8 + offset + 4] = const_hessian * cnt_hist[ltid * 4 + offset];
    offset = (offset + 1) & 0x3;
    gh_hist[ltid * 8 + offset] += gh_hist[ltid * 8 + offset + 4];
    gh_hist[ltid * 8 + offset + 4] = const_hessian * cnt_hist[ltid * 4 + offset];
    barrier(CLK_LOCAL_MEM_FENCE);
    #endif

    // write to output
    // write gradients and hessians histogram for all 4 features
    /* memory layout in gh_hist (total 2 * 4 * 256 * sizeof(float) = 8 KB):
       -----------------------------------------------------------------------------------------------
       g_f0_bin0   g_f1_bin0   g_f2_bin0   g_f3_bin0   h_f0_bin0   h_f1_bin0   h_f2_bin0   h_f3_bin0
       g_f0_bin1   g_f1_bin1   g_f2_bin1   g_f3_bin1   h_f0_bin1   h_f1_bin1   h_f2_bin1   h_f3_bin1
       ...
       g_f0_bin255 g_f1_bin255 g_f2_bin255 g_f3_bin255 h_f0_bin255 h_f1_bin255 h_f2_bin255 h_f3_bin255
       -----------------------------------------------------------------------------------------------
    */
    /* memory layout in cnt_hist (total 4 * 256 * sizeof(uint) = 4 KB):
       -----------------------------------------------
       c_f0_bin0   c_f1_bin0   c_f2_bin0   c_f3_bin0
       c_f0_bin1   c_f1_bin1   c_f2_bin1   c_f3_bin1
       ...
       c_f0_bin255 c_f1_bin255 c_f2_bin255 c_f3_bin255
       -----------------------------------------------
    */
    // output data in linear order for further reduction
    // output size = 4 (features) * 3 (counters) * 256 (bins) * sizeof(float)
    /* memory layout of output:
       --------------------------------------------
       g_f0_bin0   g_f0_bin1   ...   g_f0_bin255   \
       h_f0_bin0   h_f0_bin1   ...   h_f0_bin255    |
       c_f0_bin0   c_f0_bin1   ...   c_f0_bin255    |
       g_f1_bin0   g_f1_bin1   ...   g_f1_bin255    |
       h_f1_bin0   h_f1_bin1   ...   h_f1_bin255    |
       c_f1_bin0   c_f1_bin1   ...   c_f1_bin255    |--- 1 sub-histogram block
       g_f2_bin0   g_f2_bin1   ...   g_f2_bin255    |
       h_f2_bin0   h_f2_bin1   ...   h_f2_bin255    |
       c_f2_bin0   c_f2_bin1   ...   c_f2_bin255    |
       g_f3_bin0   g_f3_bin1   ...   g_f3_bin255    |
       h_f3_bin0   h_f3_bin1   ...   h_f3_bin255    |
       c_f3_bin0   c_f3_bin1   ...   c_f3_bin255   /
       --------------------------------------------
    */
    uint feature4_id = (group_id >> POWER_FEATURE_WORKGROUPS);
    // if there is only one workgroup processing this feature4, don't even need to write
    #if POWER_FEATURE_WORKGROUPS != 0
    __global acc_type * restrict output = (__global acc_type * restrict)output_buf + group_id * 4 * 2 * NUM_BINS;
    // write gradients and hessians
    __global acc_type * restrict ptr_f = output;
    for (ushort j = 0; j < 4; ++j) {
        for (ushort i = ltid; i < 2 * NUM_BINS; i += lsize) {
            // even threads read gradients, odd threads read hessians
            // FIXME: 2-way bank conflict
            acc_type value = gh_hist[i * 4 + j];
            ptr_f[(i & 1) * NUM_BINS + (i >> 1)] = value;
        }
        ptr_f += 2 * NUM_BINS;
    }
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
    __local uint * counter_val = (__local uint *)(gh_hist + 2 * 4 * NUM_BINS);;
    // backup the old value
    uint old_val = *counter_val;
    if (ltid == 0) {
        // all workgroups processing the same feature add this counter
        *counter_val = atom_inc(sync_counters + feature4_id);
    }
    // make sure everyone in this workgroup is here
    barrier(CLK_LOCAL_MEM_FENCE);
    // everyone in this workgroup: if we are the last workgroup, then do reduction!
    if (*counter_val == (1 << POWER_FEATURE_WORKGROUPS) - 1) {
        if (ltid == 0) {
            // printf("workgroup %d: %g %g %g %g %g %g %g %g\n", group_id, gh_hist[0], gh_hist[1], gh_hist[2], gh_hist[3], gh_hist[4], gh_hist[5], gh_hist[6], gh_hist[7]);
            // printf("feature_data[0] = %d %d %d %d", feature_data[0].s0, feature_data[0].s1, feature_data[0].s2, feature_data[0].s3);
            // clear the sync counter for using it next time
            sync_counters[feature4_id] = 0;
        }
    #else
    // only 1 work group, no need to increase counter
    // the reduction will become a simple copy
    if (1) {
        uint old_val; // dummy
    #endif
        // locate our feature4's block in output memory
        uint output_offset = (feature4_id << POWER_FEATURE_WORKGROUPS);
        __global acc_type const * restrict feature4_subhists =
                 (__global acc_type *)output_buf + output_offset * 4 * 2 * NUM_BINS;
        // skip reading the data already in local memory
        uint skip_id = group_id ^ output_offset;
        // locate output histogram location for this feature4
        __global acc_type* restrict hist_buf = hist_buf_base + feature4_id * 4 * 2 * NUM_BINS;
        within_kernel_reduction256x4(feature_mask, feature4_subhists, skip_id, old_val, 1 << POWER_FEATURE_WORKGROUPS,
                                     hist_buf, (__local acc_type *)shared_array);
        // if (ltid == 0)
        //    printf("workgroup %d reduction done, %g %g %g %g %g %g %g %g\n", group_id, hist_buf[0], hist_buf[3*NUM_BINS], hist_buf[2*3*NUM_BINS], hist_buf[3*3*NUM_BINS], hist_buf[1], hist_buf[3*NUM_BINS+1], hist_buf[2*3*NUM_BINS+1], hist_buf[3*3*NUM_BINS+1]);
    }
}

// The following line ends the string literal, adds an extra #endif at the end
// )"" "\n#endif"
#endif
