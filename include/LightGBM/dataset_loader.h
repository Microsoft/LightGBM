/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_DATASET_LOADER_H_
#define LIGHTGBM_DATASET_LOADER_H_

#include <LightGBM/dataset.h>

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace LightGBM {

class DatasetLoader {
 public:
  LIGHTGBM_EXPORT DatasetLoader(const Config& io_config, const PredictFunction& predict_fun, int num_class, const char* filename);

  LIGHTGBM_EXPORT ~DatasetLoader();

  LIGHTGBM_EXPORT Dataset* LoadFromFile(const char* filename, int rank, int num_machines);

  LIGHTGBM_EXPORT Dataset* LoadFromFile(const char* filename) {
    return LoadFromFile(filename, 0, 1);
  }

  LIGHTGBM_EXPORT Dataset* LoadFromFileAlignWithOtherDataset(const char* filename, const Dataset* train_data);

  LIGHTGBM_EXPORT Dataset* ConstructFromSampleData(double** sample_values,
    int** sample_indices, int num_col, const int* num_per_col,
    size_t total_sample_size, data_size_t num_data);

  /*! \brief Disable copy */
  DatasetLoader& operator=(const DatasetLoader&) = delete;
  /*! \brief Disable copy */
  DatasetLoader(const DatasetLoader&) = delete;

  static std::vector<std::vector<double>> GetForcedBins(std::string forced_bins_path, int num_total_features,
                                                        const std::unordered_set<int>& categorical_features);

 private:
  Dataset* LoadFromBinFile(const char* data_filename, const char* bin_filename, int rank, int num_machines, int* num_global_data, std::vector<data_size_t>* used_data_indices);

  void SetHeader(const char* filename);

  void CheckDataset(const Dataset* dataset, bool is_load_from_binary);

  std::vector<std::string> LoadTextDataToMemory(const char* filename, const Metadata& metadata, int rank, int num_machines, int* num_global_data, std::vector<data_size_t>* used_data_indices);

  std::vector<std::string> SampleTextDataFromMemory(const std::vector<std::string>& data);

  std::vector<std::string> SampleTextDataFromFile(const char* filename, const Metadata& metadata, int rank, int num_machines, int* num_global_data, std::vector<data_size_t>* used_data_indices);

  void ConstructBinMappersFromTextData(int rank, int num_machines, const std::vector<std::string>& sample_data, const Parser* parser, Dataset* dataset);

  /*! \brief Extract local features from memory */
  void ExtractFeaturesFromMemory(std::vector<std::string>* text_data, const Parser* parser, Dataset* dataset);

  /*! \brief Extract local features from file */
  void ExtractFeaturesFromFile(const char* filename, const Parser* parser, const std::vector<data_size_t>& used_data_indices, Dataset* dataset);

  /*! \brief Check can load from binary file */
  std::string CheckCanLoadFromBin(const char* filename);

  /*! \brief Check the number of bins for categorical features.
   * The number of bins for categorical features may exceed the configured maximum value.
   * Log warnings when such cases happen.
   *
   * \param bin_mappers the bin_mappers of all features
   * \param max_bin max_bin from Config
   * \param max_bin_by_feature max_bin_by_feature from Config
   */
  void CheckCategoricalFeatureNumBin(const std::vector<std::unique_ptr<BinMapper>>& bin_mappers, const int max_bin, const std::vector<int>& max_bin_by_feature) const;

  const Config& config_;
  /*! \brief Random generator*/
  Random random_;
  /*! \brief prediction function for initial model */
  const PredictFunction& predict_fun_;
  /*! \brief number of classes */
  int num_class_;
  /*! \brief index of label column */
  int label_idx_;
  /*! \brief index of weight column */
  int weight_idx_;
  /*! \brief index of group column */
  int group_idx_;
  /*! \brief Mapper from real feature index to used index*/
  std::unordered_set<int> ignore_features_;
  /*! \brief store feature names */
  std::vector<std::string> feature_names_;
  /*! \brief Mapper from real feature index to used index*/
  std::unordered_set<int> categorical_features_;
  /*! \brief Whether to store raw feature values */
  bool store_raw_;
};

}  // namespace LightGBM

#endif  // LIGHTGBM_DATASET_LOADER_H_
