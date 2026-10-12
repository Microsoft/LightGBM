/*!
 * Copyright (c) 2016-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2016-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_INCLUDE_LIGHTGBM_DATASET_H_
#define LIGHTGBM_INCLUDE_LIGHTGBM_DATASET_H_

#include <LightGBM/arrow.h>
#include <LightGBM/config.h>
#include <LightGBM/feature_group.h>
#include <LightGBM/meta.h>
#include <LightGBM/train_share_states.h>
#include <LightGBM/utils/byte_buffer.h>
#include <LightGBM/utils/openmp_wrapper.h>
#include <LightGBM/utils/random.h>
#include <LightGBM/utils/text_reader.h>

#include <cstdint>
#include <string>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_set>
#include <utility>
#include <vector>

#include <LightGBM/cuda/cuda_column_data.hpp>
#include <LightGBM/cuda/cuda_metadata.hpp>

namespace LightGBM {

/*! \brief forward declaration */
class DatasetLoader;
/*!
* \brief This class is used to store some meta(non-feature) data for training data,
*        e.g. labels, weights, initial scores, query level information.
*
*        Some details:
*        1. Label, used for training.
*        2. Weights, weighs of records, optional
*        3. Query Boundaries, necessary for LambdaRank.
*           The documents of i-th query is in [ query_boundaries[i], query_boundaries[i+1] )
*        4. Query Weights, auto calculate by weights and query_boundaries(if both of them are existed)
*           the weight for i-th query is sum(query_boundaries[i] , .., query_boundaries[i+1]) / (query_boundaries[i + 1] -  query_boundaries[i+1])
*        5. Initial score. optional. if existing, the model will boost from this score, otherwise will start from 0.
*/
class Metadata {
 public:
  /*!
  * \brief Null constructor
  */
  Metadata();
  /*!
  * \brief Initialization will load query level information, since it is need for sampling data
  * \param data_filename Filename of data
  */
  void Init(const char* data_filename);
  /*!
  * \brief init as subset
  * \param metadata Filename of data
  * \param used_indices
  * \param num_used_indices
  */
  void Init(const Metadata& metadata, const data_size_t* used_indices, data_size_t num_used_indices);
  /*!
  * \brief Initial with binary memory
  * \param memory Pointer to memory
  */
  void LoadFromMemory(const void* memory);
  /*! \brief Destructor */
  ~Metadata();

  /*!
  * \brief Initial work, will allocate space for label, weight (if exists) and query (if exists)
  * \param num_data Number of training data
  * \param weight_idx Index of weight column, < 0 means doesn't exists
  * \param query_idx Index of query id column, < 0 means doesn't exists
  */
  void Init(data_size_t num_data, int weight_idx, int query_idx);

  /*!
  * \brief Allocate space for label, weight (if exists), initial score (if exists) and query (if exists)
  * \param num_data Number of data
  * \param reference Reference metadata
  */
  void InitByReference(data_size_t num_data, const Metadata* reference);

  /*!
  * \brief Allocate space for label, weight (if exists), initial score (if exists) and query (if exists)
  * \param num_data Number of data rows
  * \param has_weights Whether the metadata has weights
  * \param has_init_scores Whether the metadata has initial scores
  * \param has_queries Whether the metadata has queries
  * \param nclasses Number of classes for initial scores
  */
  void Init(data_size_t num_data, int32_t has_weights, int32_t has_init_scores, int32_t has_queries, int32_t nclasses);

  /*!
  * \brief Partition label by used indices
  * \param used_indices Indices of local used
  */
  void PartitionLabel(const std::vector<data_size_t>& used_indices);

  /*!
  * \brief Partition meta data according to local used indices if need
  * \param num_all_data Number of total training data, including other machines' data on distributed learning
  * \param used_data_indices Indices of local used training data
  */
  void CheckOrPartition(data_size_t num_all_data,
                        const std::vector<data_size_t>& used_data_indices);

  void SetLabel(const label_t* label, data_size_t len);
  void SetLabel(struct ArrowArrayStream* stream);
  void SetLabel(int64_t n_chunks, struct ArrowArray* chunks, struct ArrowSchema* schema);

  void SetWeights(const label_t* weights, data_size_t len);
  void SetWeights(struct ArrowArrayStream* stream);
  void SetWeights(int64_t n_chunks, struct ArrowArray* chunks, struct ArrowSchema* schema);

  void SetQuery(const data_size_t* query, data_size_t len);
  void SetQuery(struct ArrowArrayStream* stream);
  void SetQuery(int64_t n_chunks, struct ArrowArray* chunks, struct ArrowSchema* schema);

  void SetPosition(const data_size_t* position, data_size_t len);
  void SetPosition(struct ArrowArrayStream* stream);
  void SetPosition(int64_t n_chunks, struct ArrowArray* chunks, struct ArrowSchema* schema);

  /*!
  * \brief Set initial scores
  * \param init_score Initial scores, this class will manage memory for init_score.
  */
  void SetInitScore(const double* init_score, data_size_t len);
  void SetInitScore(struct ArrowArrayStream* stream);
  void SetInitScore(int64_t n_chunks, struct ArrowArray* chunks, struct ArrowSchema* schema);


  /*!
  * \brief Save binary data to file
  * \param file File want to write
  */
  void SaveBinaryToFile(BinaryWriter* writer) const;

  /*!
  * \brief Get sizes in byte of this object
  */
  size_t SizesInByte() const;

  /*!
  * \brief Get pointer of label
  * \return Pointer of label
  */
  inline const label_t* label() const { return label_.data(); }

  /*!
  * \brief Set label for one record
  * \param idx Index of this record
  * \param value Label value of this record
  */
  inline void SetLabelAt(data_size_t idx, label_t value) {
    label_[idx] = value;
  }

  /*!
  * \brief Set Weight for one record
  * \param idx Index of this record
  * \param value Weight value of this record
  */
  inline void SetWeightAt(data_size_t idx, label_t value) {
    weights_[idx] = value;
  }

  /*!
  * \brief Set initial scores for one record.  Note that init_score might have multiple columns and is stored in column format.
  * \param idx Index of this record
  * \param values Initial score values for this record, one per class
  */
  inline void SetInitScoreAt(data_size_t idx, const double* values) {
    const auto nclasses = num_init_score_classes();
    const double* val_ptr = values;
    for (int i = idx; i < nclasses * num_data_; i += num_data_, ++val_ptr) {
      init_score_[i] = *val_ptr;
    }
  }

  /*!
  * \brief Set Query Id for one record
  * \param idx Index of this record
  * \param value Query Id value of this record
  */
  inline void SetQueryAt(data_size_t idx, data_size_t value) {
    queries_[idx] = static_cast<data_size_t>(value);
  }

  /*! \brief Load initial scores from file */
  void LoadInitialScore(const std::string& data_filename);

  /*!
  * \brief Insert data from a given data to the current data at a specified index
  * \param start_index The target index to begin the insertion
  * \param count Number of records to insert
  * \param labels Pointer to label data
  * \param weights Pointer to weight data, or null
  * \param init_scores Pointer to init-score data, or null
  * \param queries Pointer to query data, or null
  */
  void InsertAt(data_size_t start_index,
    data_size_t count,
    const float* labels,
    const float* weights,
    const double* init_scores,
    const int32_t* queries);

  /*!
  * \brief Perform any extra operations after all data has been loaded
  */
  void FinishLoad();
  /*!
  * \brief Get weights, if not exists, will return nullptr
  * \return Pointer of weights
  */
  inline const label_t* weights() const {
    if (!weights_.empty()) {
      return weights_.data();
    } else {
      return nullptr;
    }
  }

  /*!
  * \brief Get positions, if does not exist then return nullptr
  * \return Pointer of positions
  */
  inline const data_size_t* positions() const {
    if (!positions_.empty()) {
      return positions_.data();
    } else {
      return nullptr;
    }
  }

  /*!
  * \brief Get position IDs, if does not exist then return nullptr
  * \return Pointer of position IDs
  */
  inline const std::string* position_ids() const {
    if (!position_ids_.empty()) {
      return position_ids_.data();
    } else {
      return nullptr;
    }
  }

  /*!
  * \brief Get Number of different position IDs
  * \return number of different position IDs
  */
  inline size_t num_position_ids() const {
      return position_ids_.size();
  }

  /*!
  * \brief Get data boundaries on queries, if not exists, will return nullptr
  *        we assume data will order by query,
  *        the interval of [query_boundaris[i], query_boundaris[i+1])
  *        is the data indices for query i.
  * \return Pointer of data boundaries on queries
  */
  inline const data_size_t* query_boundaries() const {
    if (!query_boundaries_.empty()) {
      return query_boundaries_.data();
    } else {
      return nullptr;
    }
  }

  /*!
  * \brief Get Number of queries
  * \return Number of queries
  */
  inline data_size_t num_queries() const { return num_queries_; }

  /*!
  * \brief Get weights for queries, if not exists, will return nullptr
  * \return Pointer of weights for queries
  */
  inline const label_t* query_weights() const {
    if (!query_weights_.empty()) {
      return query_weights_.data();
    } else {
      return nullptr;
    }
  }

  /*!
  * \brief Get initial scores, if not exists, will return nullptr
  * \return Pointer of initial scores
  */
  inline const double* init_score() const {
    if (!init_score_.empty()) {
      return init_score_.data();
    } else {
      return nullptr;
    }
  }

  /*!
  * \brief Get size of initial scores
  */
  inline int64_t num_init_score() const { return num_init_score_; }

  /*!
  * \brief Get number of classes
  */
  inline int32_t num_init_score_classes() const {
    if (num_data_ && num_init_score_) {
      return static_cast<int>(num_init_score_ / num_data_);
    }
    return 1;
  }

  /*! \brief Disable copy */
  Metadata& operator=(const Metadata&) = delete;
  /*! \brief Disable copy */
  Metadata(const Metadata&) = delete;

  #ifdef USE_CUDA

  CUDAMetadata* cuda_metadata() const { return cuda_metadata_.get(); }

  void CreateCUDAMetadata(const int gpu_device_id);

  #endif  // USE_CUDA

 private:
  /*! \brief Load wights from file */
  void LoadWeights();
  /*! \brief Load positions from file */
  void LoadPositions();
  /*! \brief Load query boundaries from file */
  void LoadQueryBoundaries();
  /*! \brief Calculate query weights from queries */
  void CalculateQueryWeights();
  /*! \brief Calculate query boundaries from queries */
  void CalculateQueryBoundaries();
  /*! \brief Insert labels at the given index */
  void InsertLabels(const label_t* labels, data_size_t start_index, data_size_t len);
  /*! \brief Set labels from pointers to the first element and the end of an iterator. */
  template <typename It>
  void SetLabelsFromIterator(It first, It last);
  /*! \brief Insert weights at the given index */
  void InsertWeights(const label_t* weights, data_size_t start_index, data_size_t len);
  /*! \brief Set weights from pointers to the first element and the end of an iterator. */
  template <typename It>
  void SetWeightsFromIterator(It first, It last);
  /*! \brief Insert initial scores at the given index */
  void InsertInitScores(const double* init_scores, data_size_t start_index, data_size_t len, data_size_t source_size);
  /*! \brief Set init scores from pointers to the first element and the end of an iterator. */
  template <typename It>
  void SetInitScoresFromIterator(It first, It last);
  /*! \brief Insert queries at the given index */
  void InsertQueries(const data_size_t* queries, data_size_t start_index, data_size_t len);
  /*! \brief Set queries from pointers to the first element and the end of an iterator. */
  template <typename It>
  void SetQueriesFromIterator(It first, It last);
  /*! \brief Filename of current data */
  std::string data_filename_;
  /*! \brief Number of data */
  data_size_t num_data_;
  /*! \brief Number of weights, used to check correct weight file */
  data_size_t num_weights_;
  /*! \brief Number of positions, used to check correct position file */
  data_size_t num_positions_;
  /*! \brief Label data */
  std::vector<label_t> label_;
  /*! \brief Weights data */
  std::vector<label_t> weights_;
  /*! \brief Positions data */
  std::vector<data_size_t> positions_;
  /*! \brief Position identifiers */
  std::vector<std::string> position_ids_;
  /*! \brief Query boundaries */
  std::vector<data_size_t> query_boundaries_;
  /*! \brief Query weights */
  std::vector<label_t> query_weights_;
  /*! \brief Number of queries */
  data_size_t num_queries_;
  /*! \brief Number of Initial score, used to check correct weight file */
  int64_t num_init_score_;
  /*! \brief Initial score */
  std::vector<double> init_score_;
  /*! \brief Queries data */
  std::vector<data_size_t> queries_;
  /*! \brief mutex for threading safe call */
  std::mutex mutex_;
  bool weight_load_from_file_;
  bool position_load_from_file_;
  bool query_load_from_file_;
  bool init_score_load_from_file_;
  #ifdef USE_CUDA
  std::unique_ptr<CUDAMetadata> cuda_metadata_;
  #endif  // USE_CUDA
};


/*! \brief Interface for Parser */
class Parser {
 public:
  typedef const char* (*AtofFunc)(const char* p, double* out);

  /*! \brief Default constructor */
  Parser() {}

  /*!
  * \brief Constructor for customized parser. The constructor accepts content not path because need to save/load the config along with model string
  */
  explicit Parser(std::string) {}

  /*! \brief virtual destructor */
  virtual ~Parser() {}

  /*!
  * \brief Parse one line with label
  * \param str One line record, string format, should end with '\0'
  * \param out_features Output columns, store in (column_idx, values)
  * \param out_label Label will store to this if exists
  */
  virtual void ParseOneLine(const char* str,
                            std::vector<std::pair<int, double>>* out_features, double* out_label) const = 0;

  virtual int NumFeatures() const = 0;

  /*!
  * \brief Create an object of parser, will auto choose the format depend on file
  * \param filename One Filename of data
  * \param header whether input file contains header
  * \param num_features Pass num_features of this data file if you know, <=0 means don't know
  * \param label_idx index of label column
  * \param precise_float_parser using precise floating point number parsing if true
  * \return Object of parser
  */
  static Parser* CreateParser(const char* filename, bool header, int num_features, int label_idx, bool precise_float_parser);

  /*!
  * \brief Create an object of parser, could use customized parser, or auto choose the format depend on file
  * \param filename One Filename of data
  * \param header whether input file contains header
  * \param num_features Pass num_features of this data file if you know, <=0 means don't know
  * \param label_idx index of label column
  * \param precise_float_parser using precise floating point number parsing if true
  * \param parser_config_str Customized parser config content
  * \return Object of parser
  */
  static Parser* CreateParser(const char* filename, bool header, int num_features, int label_idx, bool precise_float_parser,
                              std::string parser_config_str);

  /*!
  * \brief Generate parser config str used for custom parser initialization, may save values of label id and header
  * \param filename One Filename of data
  * \param parser_config_filename One Filename of parser config
  * \param header whether input file contains header
  * \param label_idx index of label column
  * \return Parser config str
  */
  static std::string GenerateParserConfigStr(const char* filename, const char* parser_config_filename, bool header, int label_idx);
};

/*! \brief Interface for parser factory, used by customized parser */
class ParserFactory {
 private:
  ParserFactory() {}
  std::map<std::string, std::function<Parser*(std::string)>> object_map_;

 public:
  ~ParserFactory() {}
  static ParserFactory& getInstance();
  void Register(std::string class_name, std::function<Parser*(std::string)> objc);
  Parser* getObject(std::string class_name, std::string config_str);
};

/*! \brief Interface for parser reflector, used by customized parser */
class ParserReflector {
 public:
  ParserReflector(std::string class_name, std::function<Parser*(std::string)> objc) {
    ParserFactory::getInstance().Register(class_name, objc);
  }
  virtual ~ParserReflector() {}
};

/*! \brief The main class of data set,
*          which are used to training or validation
*/
class Dataset {
 public:
  friend DatasetLoader;

  LIGHTGBM_EXPORT Dataset();

  LIGHTGBM_EXPORT Dataset(data_size_t num_data);

  void Construct(
    std::vector<std::unique_ptr<BinMapper>>* bin_mappers,
    int num_total_features,
    const std::vector<std::vector<double>>& forced_bins,
    int** sample_non_zero_indices,
    double** sample_values,
    const int* num_per_col,
    int num_sample_col,
    size_t total_sample_cnt,
    const Config& io_config);

  /*! \brief Destructor */
  LIGHTGBM_EXPORT ~Dataset();

  /*!
  * \brief Initialize from the given reference
  * \param num_data Number of data
  * \param reference Reference dataset
  */
  LIGHTGBM_EXPORT void InitByReference(data_size_t num_data, const Dataset* reference) {
    metadata_.InitByReference(num_data, &reference->metadata());
  }

  LIGHTGBM_EXPORT void InitStreaming(data_size_t num_data,
                                     int32_t has_weights,
                                     int32_t has_init_scores,
                                     int32_t has_queries,
                                     int32_t nclasses,
                                     int32_t nthreads,
                                     int32_t omp_max_threads) {
    // Initialize optional max thread count with either parameter or OMP setting
    if (omp_max_threads > 0) {
      omp_max_threads_ = omp_max_threads;
    } else if (omp_max_threads_ <= 0) {
      omp_max_threads_ = OMP_NUM_THREADS();
    }

    metadata_.Init(num_data, has_weights, has_init_scores, has_queries, nclasses);
    for (int i = 0; i < num_groups_; ++i) {
      feature_groups_[i]->InitStreaming(nthreads, omp_max_threads_);
    }
  }

  LIGHTGBM_EXPORT bool CheckAlign(const Dataset& other) const {
    if (num_features_ != other.num_features_) {
      return false;
    }
    if (num_total_features_ != other.num_total_features_) {
      return false;
    }
    if (label_idx_ != other.label_idx_) {
      return false;
    }
    for (int i = 0; i < num_features_; ++i) {
      if (!FeatureBinMapper(i)->CheckAlign(*(other.FeatureBinMapper(i)))) {
        return false;
      }
    }
    return true;
  }

  inline void FinishOneRow(int tid, data_size_t row_idx, const std::vector<bool>& is_feature_added) {
    if (is_finish_load_) {
      return;
    }
    for (auto fidx : feature_need_push_zeros_) {
      if (is_feature_added[fidx]) {
        continue;
      }
      const int group = feature2group_[fidx];
      const int sub_feature = feature2subfeature_[fidx];
      feature_groups_[group]->PushData(tid, sub_feature, row_idx, 0.0f);
    }
  }

  inline void PushOneValue(int tid, data_size_t row_idx, size_t col_idx, double value) {
    if (this->is_finish_load_)
      return;
    auto feature_idx = this->used_feature_map_[col_idx];
    if (feature_idx >= 0) {
      auto group = this->feature2group_[feature_idx];
      auto sub_feature = this->feature2subfeature_[feature_idx];
      this->feature_groups_[group]->PushData(tid, sub_feature, row_idx, value);
      if (this->has_raw_) {
        auto feat_ind = numeric_feature_map_[feature_idx];
        if (feat_ind >= 0) {
          raw_data_[feat_ind][row_idx] = static_cast<float>(value);
        }
      }
    }
  }

  inline void PushOneRow(int tid, data_size_t row_idx, const std::vector<double>& feature_values) {
    for (size_t i = 0; i < feature_values.size() && i < static_cast<size_t>(num_total_features_); ++i) {
      this->PushOneValue(tid, row_idx, i, feature_values[i]);
    }
  }

  inline void PushOneRow(int tid, data_size_t row_idx, const std::vector<std::pair<int, double>>& feature_values) {
    if (is_finish_load_) {
      return;
    }
    std::vector<bool> is_feature_added(num_features_, false);
    for (auto& inner_data : feature_values) {
      if (inner_data.first >= num_total_features_) {
        continue;
      }
      int feature_idx = used_feature_map_[inner_data.first];
      if (feature_idx >= 0) {
        is_feature_added[feature_idx] = true;
        const int group = feature2group_[feature_idx];
        const int sub_feature = feature2subfeature_[feature_idx];
        feature_groups_[group]->PushData(tid, sub_feature, row_idx, inner_data.second);
        if (has_raw_) {
          int feat_ind = numeric_feature_map_[feature_idx];
          if (feat_ind >= 0) {
            raw_data_[feat_ind][row_idx] = static_cast<float>(inner_data.second);
          }
        }
      }
    }
    FinishOneRow(tid, row_idx, is_feature_added);
  }

  inline void PushOneData(int tid, data_size_t row_idx, int group, int feature_idx, int sub_feature, double value) {
    feature_groups_[group]->PushData(tid, sub_feature, row_idx, value);
    if (has_raw_) {
      int feat_ind = numeric_feature_map_[feature_idx];
      if (feat_ind >= 0) {
        raw_data_[feat_ind][row_idx] = static_cast<float>(value);
      }
    }
  }

  inline void InsertMetadataAt(data_size_t start_index,
    data_size_t count,
    const label_t* labels,
    const label_t* weights,
    const double* init_scores,
    const data_size_t* queries) {
    metadata_.InsertAt(start_index, count, labels, weights, init_scores, queries);
  }

  inline int RealFeatureIndex(int fidx) const {
    return real_feature_idx_[fidx];
  }

  inline int InnerFeatureIndex(int col_idx) const {
    return used_feature_map_[col_idx];
  }
  inline int Feature2Group(int feature_idx) const {
    return feature2group_[feature_idx];
  }
  inline int Feature2SubFeature(int feature_idx) const {
    return feature2subfeature_[feature_idx];
  }
  inline uint64_t GroupBinBoundary(int group_idx) const {
    return group_bin_boundaries_[group_idx];
  }
  inline uint64_t NumTotalBin() const {
    return group_bin_boundaries_.back();
  }

  inline std::vector<int> ValidFeatureIndices() const {
    std::vector<int> ret;
    for (int i = 0; i < num_total_features_; ++i) {
      if (used_feature_map_[i] >= 0) {
        ret.push_back(i);
      }
    }
    return ret;
  }
  void ReSize(data_size_t num_data);

  void CopySubrow(const Dataset* fullset, const data_size_t* used_indices, data_size_t num_used_indices, bool need_meta_data);

  void CopySubrowToDevice(const Dataset* fullset, const data_size_t* used_indices, data_size_t num_used_indices, bool need_meta_data, int gpu_device_id);

  MultiValBin* GetMultiBinFromSparseFeatures(const std::vector<uint32_t>& offsets) const;

  MultiValBin* GetMultiBinFromAllFeatures(const std::vector<uint32_t>& offsets) const;

  template <bool USE_QUANT_GRAD, int HIST_BITS>
  TrainingShareStates* GetShareStates(
      score_t* gradients, score_t* hessians,
      const std::vector<int8_t>& is_feature_used, bool is_constant_hessian,
      bool force_col_wise, bool force_row_wise, const int num_grad_quant_bins) const;

  LIGHTGBM_EXPORT void FinishLoad();

  bool SetFieldFromArrow(const char* field_name, struct ArrowArrayStream* stream);

  bool SetFieldFromArrow(const char* field_name, int64_t n_chunks,
                         struct ArrowArray* chunks, struct ArrowSchema* schema);

  LIGHTGBM_EXPORT bool SetFloatField(const char* field_name, const float* field_data, data_size_t num_element);

  LIGHTGBM_EXPORT bool SetDoubleField(const char* field_name, const double* field_data, data_size_t num_element);

  LIGHTGBM_EXPORT bool SetIntField(const char* field_name, const int* field_data, data_size_t num_element);

  LIGHTGBM_EXPORT bool GetFloatField(const char* field_name, data_size_t* out_len, const float** out_ptr);

  LIGHTGBM_EXPORT bool GetDoubleField(const char* field_name, data_size_t* out_len, const double** out_ptr);

  LIGHTGBM_EXPORT bool GetIntField(const char* field_name, data_size_t* out_len, const int** out_ptr);

  /*!
  * \brief Save current dataset into binary file, will save to "filename.bin"
  */
  LIGHTGBM_EXPORT void SaveBinaryFile(const char* bin_filename);

  /*!
   * \brief Serialize the overall Dataset definition/schema to a binary buffer (i.e., without data)
   */
  LIGHTGBM_EXPORT void SerializeReference(ByteBuffer* out);

  LIGHTGBM_EXPORT void DumpTextFile(const char* text_filename);

  LIGHTGBM_EXPORT void CopyFeatureMapperFrom(const Dataset* dataset);

  LIGHTGBM_EXPORT void CreateValid(const Dataset* dataset);

  void InitTrain(const std::vector<int8_t>& is_feature_used,
                 TrainingShareStates* share_state) const;

  template <bool USE_INDICES, bool USE_HESSIAN, bool USE_QUANT_GRAD, int HIST_BITS>
  void ConstructHistogramsInner(const std::vector<int8_t>& is_feature_used,
                                const data_size_t* data_indices,
                                data_size_t num_data, const score_t* gradients,
                                const score_t* hessians,
                                score_t* ordered_gradients,
                                score_t* ordered_hessians,
                                TrainingShareStates* share_state,
                                hist_t* hist_data) const;

  template <bool USE_INDICES, bool ORDERED, bool USE_QUANT_GRAD, int HIST_BITS>
  void ConstructHistogramsMultiVal(const data_size_t* data_indices,
                                   data_size_t num_data,
                                   const score_t* gradients,
                                   const score_t* hessians,
                                   TrainingShareStates* share_state,
                                   hist_t* hist_data) const;

  template <bool USE_QUANT_GRAD, int HIST_BITS>
  inline void ConstructHistograms(
      const std::vector<int8_t>& is_feature_used,
      const data_size_t* data_indices, data_size_t num_data,
      const score_t* gradients, const score_t* hessians,
      score_t* ordered_gradients, score_t* ordered_hessians,
      TrainingShareStates* share_state, hist_t* hist_data) const {
    if (num_data <= 0) {
      return;
    }
    bool use_indices = data_indices != nullptr && (num_data < num_data_);
    if (share_state->is_constant_hessian) {
      if (use_indices) {
        ConstructHistogramsInner<true, false, USE_QUANT_GRAD, HIST_BITS>(
            is_feature_used, data_indices, num_data, gradients, hessians,
            ordered_gradients, ordered_hessians, share_state, hist_data);
      } else {
        ConstructHistogramsInner<false, false, USE_QUANT_GRAD, HIST_BITS>(
            is_feature_used, data_indices, num_data, gradients, hessians,
            ordered_gradients, ordered_hessians, share_state, hist_data);
      }
    } else {
      if (use_indices) {
        ConstructHistogramsInner<true, true, USE_QUANT_GRAD, HIST_BITS>(
            is_feature_used, data_indices, num_data, gradients, hessians,
            ordered_gradients, ordered_hessians, share_state, hist_data);
      } else {
        ConstructHistogramsInner<false, true, USE_QUANT_GRAD, HIST_BITS>(
            is_feature_used, data_indices, num_data, gradients, hessians,
            ordered_gradients, ordered_hessians, share_state, hist_data);
      }
    }
  }

  void FixHistogram(int feature_idx, double sum_gradient, double sum_hessian, hist_t* data) const;

  template <typename PACKED_HIST_BIN_T, typename PACKED_HIST_ACC_T, int HIST_BITS_BIN, int HIST_BITS_ACC>
  void FixHistogramInt(int feature_idx, int64_t sum_gradient_and_hessian, hist_t* data) const;

  inline data_size_t Split(int feature, const uint32_t* threshold,
                           int num_threshold, bool default_left,
                           const data_size_t* data_indices,
                           data_size_t cnt, data_size_t* lte_indices,
                           data_size_t* gt_indices) const {
    const int group = feature2group_[feature];
    const int sub_feature = feature2subfeature_[feature];
    return feature_groups_[group]->Split(
        sub_feature, threshold, num_threshold, default_left, data_indices,
        cnt, lte_indices, gt_indices);
  }

  inline int SubFeatureBinOffset(int i) const {
    const int sub_feature = feature2subfeature_[i];
    if (sub_feature == 0) {
      return 1;
    } else {
      return 0;
    }
  }

  inline int FeatureNumBin(int i) const {
    const int group = feature2group_[i];
    const int sub_feature = feature2subfeature_[i];
    return feature_groups_[group]->bin_mappers_[sub_feature]->num_bin();
  }

  inline int FeatureGroupNumBin(int group) const {
    return feature_groups_[group]->num_total_bin_;
  }

  inline const BinMapper* FeatureBinMapper(int i) const {
    const int group = feature2group_[i];
    const int sub_feature = feature2subfeature_[i];
    return feature_groups_[group]->bin_mappers_[sub_feature].get();
  }

  inline const Bin* FeatureGroupBin(int group) const {
    return feature_groups_[group]->bin_data_.get();
  }

  inline BinIterator* FeatureIterator(int i) const {
    const int group = feature2group_[i];
    const int sub_feature = feature2subfeature_[i];
    return feature_groups_[group]->SubFeatureIterator(sub_feature);
  }

  inline BinIterator* FeatureGroupIterator(int group) const {
    return feature_groups_[group]->FeatureGroupIterator();
  }

  inline bool IsMultiGroup(int i) const {
    return feature_groups_[i]->is_multi_val_;
  }

  inline size_t FeatureGroupSizesInByte(int group) const {
    return feature_groups_[group]->FeatureGroupSizesInByte();
  }

  inline void* FeatureGroupData(int group) const {
    return feature_groups_[group]->FeatureGroupData();
  }

  const void* GetColWiseData(
    const int feature_group_index,
    const int sub_feature_index,
    uint8_t* bit_type,
    bool* is_sparse,
    std::vector<BinIterator*>* bin_iterator,
    const int num_threads) const;

  const void* GetColWiseData(
    const int feature_group_index,
    const int sub_feature_index,
    uint8_t* bit_type,
    bool* is_sparse,
    BinIterator** bin_iterator) const;

  inline double RealThreshold(int i, uint32_t threshold) const {
    const int group = feature2group_[i];
    const int sub_feature = feature2subfeature_[i];
    return feature_groups_[group]->bin_mappers_[sub_feature]->BinToValue(threshold);
  }

  // given a real threshold, find the closest threshold bin
  inline uint32_t BinThreshold(int i, double threshold_double) const {
    const int group = feature2group_[i];
    const int sub_feature = feature2subfeature_[i];
    return feature_groups_[group]->bin_mappers_[sub_feature]->ValueToBin(threshold_double);
  }

  inline int MaxRealCatValue(int i) const {
    const int group = feature2group_[i];
    const int sub_feature = feature2subfeature_[i];
    return feature_groups_[group]->bin_mappers_[sub_feature]->MaxCatValue();
  }

  /*!
  * \brief Get meta data pointer
  * \return Pointer of meta data
  */
  inline const Metadata& metadata() const { return metadata_; }

  /*! \brief Get Number of used features */
  inline int num_features() const { return num_features_; }

  /*! \brief Get number of numeric features */
  inline int num_numeric_features() const { return num_numeric_features_; }

  /*! \brief Get Number of feature groups */
  inline int num_feature_groups() const { return num_groups_;}

  /*! \brief Get Number of total features */
  inline int num_total_features() const { return num_total_features_; }

  /*! \brief Get the index of label column */
  inline int label_idx() const { return label_idx_; }

  /*! \brief Get names of current data set */
  inline const std::vector<std::string>& feature_names() const { return feature_names_; }

  /*! \brief Get content of parser config file */
  inline const std::string parser_config_str() const { return parser_config_str_; }

  inline void set_feature_names(const std::vector<std::string>& feature_names) {
    if (feature_names.size() != static_cast<size_t>(num_total_features_)) {
      Log::Fatal("Size of feature_names error, should equal with total number of features");
    }
    feature_names_ = std::vector<std::string>(feature_names);
    std::unordered_set<std::string> feature_name_set;
    // replace ' ' in feature_names with '_'
    bool spaceInFeatureName = false;
    for (auto& feature_name : feature_names_) {
      // check JSON
      if (!Common::CheckAllowedJSON(feature_name)) {
        Log::Fatal("Do not support special JSON characters in feature name.");
      }
      if (feature_name.find(' ') != std::string::npos) {
        spaceInFeatureName = true;
        std::replace(feature_name.begin(), feature_name.end(), ' ', '_');
      }
      if (feature_name_set.count(feature_name) > 0) {
        Log::Fatal("After preprocessing (including replacing whitespace with '_'), multiple features named '%s' found in Dataset. "
                   "Ensure that feature names are unique and do not contain whitespace.", feature_name.c_str());
      }
      feature_name_set.insert(feature_name);
    }
    if (spaceInFeatureName) {
      Log::Warning("Found whitespace in feature_names, replace with underlines");
    }
  }

  inline std::vector<std::string> feature_infos() const {
    std::vector<std::string> bufs;
    for (int i = 0; i < num_total_features_; ++i) {
      int fidx = used_feature_map_[i];
      if (fidx < 0) {
        bufs.push_back("none");
      } else {
        const auto bin_mapper = FeatureBinMapper(fidx);
        bufs.push_back(bin_mapper->bin_info_string());
      }
    }
    return bufs;
  }

  /*! \brief Get Number of data */
  inline data_size_t num_data() const { return num_data_; }

  /*! \brief Get whether FinishLoad is automatically called when pushing last row. */
  inline bool wait_for_manual_finish() const { return wait_for_manual_finish_; }

  /*! \brief Get the maximum number of OpenMP threads to allocate for. */
  inline int omp_max_threads() const { return omp_max_threads_; }

  /*! \brief Set whether the Dataset is finished automatically when last row is pushed or with a manual
   *         MarkFinished API call.  Set to true for thread-safe streaming and/or if will be coalesced later.
   *         FinishLoad should not be called on any Dataset that will be coalesced.
   */
  inline void set_wait_for_manual_finish(bool value) {
    std::lock_guard<std::mutex> lock(mutex_);
    wait_for_manual_finish_ = value;
  }

  /*! \brief Disable copy */
  Dataset& operator=(const Dataset&) = delete;
  /*! \brief Disable copy */
  Dataset(const Dataset&) = delete;

  void AddFeaturesFrom(Dataset* other);

  /*! \brief Get has_raw_ */
  inline bool has_raw() const { return has_raw_; }

  /*! \brief Set has_raw_ */
  inline void SetHasRaw(bool has_raw) { has_raw_ = has_raw; }

  /*! \brief Resize raw_data_ */
  inline void ResizeRaw(int num_rows) {
    if (static_cast<int>(raw_data_.size()) > num_numeric_features_) {
      raw_data_.resize(num_numeric_features_);
    }
    for (size_t i = 0; i < raw_data_.size(); ++i) {
      raw_data_[i].resize(num_rows);
    }
    int curr_size = static_cast<int>(raw_data_.size());
    for (int i = curr_size; i < num_numeric_features_; ++i) {
      raw_data_.push_back(std::vector<float>(num_rows, 0));
    }
  }

  /*! \brief Get pointer to raw_data_ feature */
  inline const float* raw_index(int feat_ind) const {
    return raw_data_[numeric_feature_map_[feat_ind]].data();
  }

  inline uint32_t feature_max_bin(const int inner_feature_index) const {
    const int feature_group_index = Feature2Group(inner_feature_index);
    const int sub_feature_index = feature2subfeature_[inner_feature_index];
    return feature_groups_[feature_group_index]->feature_max_bin(sub_feature_index);
  }

  inline uint32_t feature_min_bin(const int inner_feature_index) const {
    const int feature_group_index = Feature2Group(inner_feature_index);
    const int sub_feature_index = feature2subfeature_[inner_feature_index];
    return feature_groups_[feature_group_index]->feature_min_bin(sub_feature_index);
  }

  #ifdef USE_CUDA

  const CUDAColumnData* cuda_column_data() const {
    return cuda_column_data_.get();
  }

  #endif  // USE_CUDA

 private:
  void SerializeHeader(BinaryWriter* serializer);

  size_t GetSerializedHeaderSize();

  void CreateCUDAColumnData();

  void CopySubrowHostPart(const Dataset* fullset, const data_size_t* used_indices, data_size_t num_used_indices, bool need_meta_data);

  std::string data_filename_;
  /*! \brief Store used features */
  std::vector<std::unique_ptr<FeatureGroup>> feature_groups_;
  /*! \brief Mapper from real feature index to used index*/
  std::vector<int> used_feature_map_;
  /*! \brief Number of used features*/
  int num_features_;
  /*! \brief Number of total features*/
  int num_total_features_;
  /*! \brief Number of total data*/
  data_size_t num_data_;
  /*! \brief Store some label level data*/
  Metadata metadata_;
  /*! \brief index of label column */
  int label_idx_ = 0;
  /*! \brief store feature names */
  std::vector<std::string> feature_names_;
  /*! \brief serialized versions */
  static const int kSerializedReferenceVersionLength;
  static const char* serialized_reference_version;
  static const char* binary_file_token;
  static const char* binary_serialized_reference_token;
  int num_groups_;
  std::vector<int> real_feature_idx_;
  std::vector<int> feature2group_;
  std::vector<int> feature2subfeature_;
  std::vector<uint64_t> group_bin_boundaries_;
  std::vector<int> group_feature_start_;
  std::vector<int> group_feature_cnt_;
  bool is_finish_load_;
  int max_bin_;
  std::vector<int32_t> max_bin_by_feature_;
  std::vector<std::vector<double>> forced_bin_bounds_;
  int bin_construct_sample_cnt_;
  int min_data_in_bin_;
  bool use_missing_;
  bool zero_as_missing_;
  std::vector<int> feature_need_push_zeros_;
  std::vector<std::vector<float>> raw_data_;
  bool wait_for_manual_finish_;
  int omp_max_threads_ = -1;
  bool has_raw_;
  /*! map feature (inner index) to its index in the list of numeric (non-categorical) features */
  std::vector<int> numeric_feature_map_;
  int num_numeric_features_;
  std::string device_type_;
  int gpu_device_id_;
  /*! \brief mutex for threading safe call */
  std::mutex mutex_;

  #ifdef USE_CUDA
  std::unique_ptr<CUDAColumnData> cuda_column_data_;
  #endif  // USE_CUDA

  std::string parser_config_str_;
};

}  // namespace LightGBM

#endif   // LIGHTGBM_INCLUDE_LIGHTGBM_DATASET_H_
