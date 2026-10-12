/*!
 * Copyright (c) 2017-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2017-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_SRC_TREELEARNER_GPU_TREE_LEARNER_H_
#define LIGHTGBM_SRC_TREELEARNER_GPU_TREE_LEARNER_H_

#include <LightGBM/dataset.h>
#include <LightGBM/feature_group.h>
#include <LightGBM/tree.h>
#include <LightGBM/utils/array_args.h>
#include <LightGBM/utils/random.h>

#include <string>
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

#include "data_partition.hpp"
#include "feature_histogram.hpp"
#include "leaf_splits.hpp"
#include "serial_tree_learner.h"
#include "split_info.hpp"

#ifdef USE_GPU

#define BOOST_COMPUTE_THREAD_SAFE
#define BOOST_COMPUTE_HAVE_THREAD_LOCAL
// Use Boost.Compute on-disk kernel cache
#define BOOST_COMPUTE_USE_OFFLINE_CACHE
#include <boost/compute/core.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/align/aligned_allocator.hpp>

namespace LightGBM {

using json11_internal_lightgbm::Json;

/*!
* \brief GPU-based parallel learning algorithm.
*/
class GPUTreeLearner: public SerialTreeLearner {
 public:
  explicit GPUTreeLearner(const Config* tree_config);
  ~GPUTreeLearner();
  void Init(const Dataset* train_data, bool is_constant_hessian) override;
  void ResetTrainingDataInner(const Dataset* train_data, bool is_constant_hessian, bool reset_multi_val_bin) override;
  void ResetIsConstantHessian(bool is_constant_hessian) override;
  Tree* Train(const score_t* gradients, const score_t *hessians, bool is_first_tree) override;

  void SetBaggingData(const Dataset* subset, const data_size_t* used_indices, data_size_t num_data) override {
    SerialTreeLearner::SetBaggingData(subset, used_indices, num_data);
    if (subset == nullptr && used_indices != nullptr) {
      // determine if we are using bagging before we construct the data partition
      // thus we can start data movement to GPU earlier
      if (num_data != num_data_) {
        use_bagging_ = true;
        return;
      }
    }
    use_bagging_ = false;
  }

 protected:
  void BeforeTrain() override;
  bool BeforeFindBestSplit(const Tree* tree, int left_leaf, int right_leaf) override;
  void FindBestSplits(const Tree* tree) override;
  void Split(Tree* tree, int best_Leaf, int* left_leaf, int* right_leaf) override;
  void ConstructHistograms(const std::vector<int8_t>& is_feature_used, bool use_subtract) override;

 private:
  /*! \brief 4-byte feature tuple used by GPU kernels */
  struct Feature4 {
      uint8_t s[4];
  };

  typedef float gpu_hist_t;

  /*!
  * \brief Find the best number of workgroups processing one feature for maximizing efficiency
  * \param leaf_num_data The number of data examples on the current leaf being processed
  * \return Log2 of the best number for workgroups per feature, in range 0...kMaxLogWorkgroupsPerFeature
  */
  int GetNumWorkgroupsPerFeature(data_size_t leaf_num_data);

  /*!
  * \brief Initialize GPU device, context and command queues
  *        Also compiles the OpenCL kernel
  * \param platform_id OpenCL platform ID
  * \param device_id OpenCL device ID
  */
  void InitGPU(int platform_id, int device_id);

  /*!
  * \brief Allocate memory for GPU computation
  */
  void AllocateGPUMemory();

  /*!
  * \brief Compile OpenCL GPU source code to kernel binaries
  */
  void BuildGPUKernels();

  /*!
   * \brief Returns OpenCL kernel build log when compiled with option opts
   * \param opts OpenCL build options
   * \return OpenCL build log
  */
  std::string GetBuildLog(const std::string &opts);

  /*!
  * \brief Setup GPU kernel arguments, preparing for launching
  */
  void SetupKernelArguments();

  /*!
   * \brief Compute GPU feature histogram for the current leaf.
   *        Indices, gradients and Hessians have been copied to the device.
   * \param leaf_num_data Number of data on current leaf
   * \param use_all_features Set to true to not use feature masks, with a faster kernel
  */
  void GPUHistogram(data_size_t leaf_num_data, bool use_all_features);

  /*!
   * \brief Wait for GPU kernel execution and read histogram
   * \param histograms Destination of histogram results from GPU.
  */
  template <typename HistType>
  void WaitAndGetHistograms(hist_t* histograms);

  /*!
   * \brief Construct GPU histogram asynchronously.
   *        Interface is similar to Dataset::ConstructHistograms().
   * \param is_feature_used A predicate vector for enabling each feature
   * \param data_indices Array of data example IDs to be included in histogram, will be copied to GPU.
   *                     Set to nullptr to skip copy to GPU.
   * \param num_data Number of data examples to be included in histogram
   * \param gradients Array of gradients for all examples.
   * \param hessians Array of Hessians for all examples.
   * \param ordered_gradients Ordered gradients will be generated and copied to GPU when gradients is not nullptr,
   *                     Set gradients to nullptr to skip copy to GPU.
   * \param ordered_hessians Ordered Hessians will be generated and copied to GPU when hessians is not nullptr,
   *                     Set hessians to nullptr to skip copy to GPU.
   * \return true if GPU kernel is launched, false if GPU is not used
  */
  bool ConstructGPUHistogramsAsync(
    const std::vector<int8_t>& is_feature_used,
    const data_size_t* data_indices, data_size_t num_data,
    const score_t* gradients, const score_t* hessians,
    score_t* ordered_gradients, score_t* ordered_hessians);


  /*! brief Log2 of max number of workgroups per feature*/
  const int kMaxLogWorkgroupsPerFeature = 10;  // 2^10
  /*! brief Max total number of workgroups with preallocated workspace.
   *        If we use more than this number of workgroups, we have to reallocate subhistograms */
  int preallocd_max_num_wg_ = 1024;

  /*! \brief True if bagging is used */
  bool use_bagging_;

  /*! \brief GPU device object */
  boost::compute::device dev_;
  /*! \brief GPU context object */
  boost::compute::context ctx_;
  /*! \brief GPU command queue object */
  boost::compute::command_queue queue_;
  /*! \brief GPU kernel for 256 bins */
  const char *kernel256_src_ = {
  #include "ocl/histogram256.cl"
  };
  /*! \brief GPU kernel for 64 bins */
  const char *kernel64_src_ = {
  #include "ocl/histogram64.cl"
  };
  /*! \brief GPU kernel for 16 bins */
  const char *kernel16_src_ = {
  #include "ocl/histogram16.cl"
  };
  /*! \brief Currently used kernel source */
  std::string kernel_source_;
  /*! \brief Currently used kernel name */
  std::string kernel_name_;

  /*! \brief an array of histogram kernels with different number
     of workgroups per feature */
  std::vector<boost::compute::kernel> histogram_kernels_;
  /*! \brief an array of histogram kernels with different number
     of workgroups per feature, with all features enabled to avoid branches */
  std::vector<boost::compute::kernel> histogram_allfeats_kernels_;
  /*! \brief an array of histogram kernels with different number
     of workgroups per feature, and processing the whole dataset */
  std::vector<boost::compute::kernel> histogram_fulldata_kernels_;
  /*! \brief total number of feature-groups */
  int num_feature_groups_;
  /*! \brief total number of dense feature-groups, which will be processed on GPU */
  int num_dense_feature_groups_;
  /*! \brief On GPU we read one DWORD (4-byte) of features of one example once.
   *  With bin size > 16, there are 4 features per DWORD.
   *  With bin size <=16, there are 8 features per DWORD.
   * */
  int dword_features_;
  /*! \brief total number of dense feature-group tuples on GPU.
   * Each feature tuple is 4-byte (4 features if each feature takes a byte) */
  int num_dense_feature4_;
  /*! \brief Max number of bins of training data, used to determine
   * which GPU kernel to use */
  int max_num_bin_;
  /*! \brief Used GPU kernel bin size (64, 256) */
  int device_bin_size_;
  /*! \brief Size of histogram bin entry, depending if single or double precision is used */
  size_t hist_bin_entry_sz_;
  /*! \brief Indices of all dense feature-groups */
  std::vector<int> dense_feature_group_map_;
  /*! \brief Indices of all sparse feature-groups */
  std::vector<int> sparse_feature_group_map_;
  /*! \brief Multipliers of all dense feature-groups, used for redistributing bins */
  std::vector<int> device_bin_mults_;
  /*! \brief GPU memory object holding the training data */
  std::unique_ptr<boost::compute::vector<Feature4>> device_features_;
  /*! \brief GPU memory object holding the ordered gradient */
  boost::compute::buffer device_gradients_;
  /*! \brief Pinned memory object for ordered gradient */
  boost::compute::buffer pinned_gradients_;
  /*! \brief Pointer to pinned memory of ordered gradient */
  void * ptr_pinned_gradients_ = nullptr;
  /*! \brief GPU memory object holding the ordered hessian */
  boost::compute::buffer device_hessians_;
  /*! \brief Pinned memory object for ordered hessian */
  boost::compute::buffer pinned_hessians_;
  /*! \brief Pointer to pinned memory of ordered hessian */
  void * ptr_pinned_hessians_ = nullptr;
  /*! \brief A vector of feature mask. 1 = feature used, 0 = feature not used */
  std::vector<char, boost::alignment::aligned_allocator<char, 4096>> feature_masks_;
  /*! \brief GPU memory object holding the feature masks */
  boost::compute::buffer device_feature_masks_;
  /*! \brief Pinned memory object for feature masks */
  boost::compute::buffer pinned_feature_masks_;
  /*! \brief Pointer to pinned memory of feature masks */
  void * ptr_pinned_feature_masks_ = nullptr;
  /*! \brief GPU memory object holding indices of the leaf being processed */
  std::unique_ptr<boost::compute::vector<data_size_t>> device_data_indices_;
  /*! \brief GPU memory object holding counters for workgroup coordination */
  std::unique_ptr<boost::compute::vector<int>> sync_counters_;
  /*! \brief GPU memory object holding temporary sub-histograms per workgroup */
  std::unique_ptr<boost::compute::vector<char>> device_subhistograms_;
  /*! \brief Host memory object for histogram output (GPU will write to Host memory directly) */
  boost::compute::buffer device_histogram_outputs_;
  /*! \brief Host memory pointer for histogram outputs */
  void * host_histogram_outputs_;
  /*! \brief OpenCL waitlist object for waiting for data transfer before kernel execution */
  boost::compute::wait_list kernel_wait_obj_;
  /*! \brief OpenCL waitlist object for reading output histograms after kernel execution */
  boost::compute::wait_list histograms_wait_obj_;
  /*! \brief Asynchronous waiting object for copying indices */
  boost::compute::future<void> indices_future_;
  /*! \brief Asynchronous waiting object for copying gradients */
  boost::compute::event gradients_future_;
  /*! \brief Asynchronous waiting object for copying Hessians */
  boost::compute::event hessians_future_;
};

}  // namespace LightGBM
#else

// When GPU support is not compiled in, quit with an error message

namespace LightGBM {

class GPUTreeLearner: public SerialTreeLearner {
 public:
  #ifdef _MSC_VER
    #pragma warning(disable : 4702)
  #endif
  explicit GPUTreeLearner(const Config* tree_config) : SerialTreeLearner(tree_config) {
    Log::Fatal("GPU Tree Learner was not enabled in this build.\n"
               "Please recompile with CMake option -DUSE_GPU=1");
  }
};

}  // namespace LightGBM

#endif   // USE_GPU

#endif   // LIGHTGBM_SRC_TREELEARNER_GPU_TREE_LEARNER_H_
