/*!
 * Copyright (c) 2016-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2016-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_SRC_APPLICATION_PREDICTOR_HPP_
#define LIGHTGBM_SRC_APPLICATION_PREDICTOR_HPP_

#include <LightGBM/boosting.h>
#include <LightGBM/dataset.h>
#include <LightGBM/meta.h>
#include <LightGBM/utils/common.h>
#include <LightGBM/utils/openmp_wrapper.h>
#include <LightGBM/utils/text_reader.h>
#include <LightGBM/utils/common.h>

#include <string>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>
#include <numeric>

namespace LightGBM {

/*!
* \brief Used to predict data with input model
*/
class Predictor {
 public:
  /*!
  * \brief Constructor
  * \param boosting Input boosting model
  * \param start_iteration Start index of the iteration to predict
  * \param num_iteration Number of boosting round
  * \param is_raw_score True if need to predict result with raw score
  * \param predict_leaf_index True to output leaf index instead of prediction score
  * \param predict_contrib True to output feature contributions instead of prediction score
  */
  Predictor(Boosting* boosting, int start_iteration, int num_iteration, bool is_raw_score,
            bool predict_leaf_index, bool predict_contrib, bool early_stop,
            int early_stop_freq, double early_stop_margin) {
    early_stop_ = CreatePredictionEarlyStopInstance(
        "none", LightGBM::PredictionEarlyStopConfig());
    if (early_stop && !boosting->NeedAccuratePrediction()) {
      PredictionEarlyStopConfig pred_early_stop_config;
      CHECK_GT(early_stop_freq, 0);
      CHECK_GE(early_stop_margin, 0);
      pred_early_stop_config.margin_threshold = early_stop_margin;
      pred_early_stop_config.round_period = early_stop_freq;
      if (boosting->NumberOfClasses() == 1) {
        early_stop_ =
            CreatePredictionEarlyStopInstance("binary", pred_early_stop_config);
      } else {
        early_stop_ = CreatePredictionEarlyStopInstance("multiclass",
                                                        pred_early_stop_config);
      }
    }

    boosting->InitPredict(start_iteration, num_iteration, predict_contrib);
    boosting_ = boosting;
    num_pred_one_row_ = boosting_->NumPredictOneRow(start_iteration,
        num_iteration, predict_leaf_index, predict_contrib);
    num_feature_ = boosting_->MaxFeatureIdx() + 1;
    predict_buf_.resize(
        OMP_NUM_THREADS(),
        std::vector<double, Common::AlignmentAllocator<double, kAlignedSize>>(
            num_feature_, 0.0f));
    const int kFeatureThreshold = 100000;
    const size_t KSparseThreshold = static_cast<size_t>(0.01 * num_feature_);
    if (predict_leaf_index) {
      predict_fun_ = [=](const std::vector<std::pair<int, double>>& features,
                         double* output) {
        int tid = omp_get_thread_num();
        if (num_feature_ > kFeatureThreshold &&
            features.size() < KSparseThreshold) {
          auto buf = CopyToPredictMap(features);
          boosting_->PredictLeafIndexByMap(buf, output);
        } else {
          CopyToPredictBuffer(predict_buf_[tid].data(), features);
          // get result for leaf index
          boosting_->PredictLeafIndex(predict_buf_[tid].data(), output);
          ClearPredictBuffer(predict_buf_[tid].data(), predict_buf_[tid].size(),
                             features);
        }
      };
    } else if (predict_contrib) {
      if (boosting_->IsLinear()) {
        Log::Fatal("Predicting SHAP feature contributions is not implemented for linear trees.");
      }
      predict_fun_ = [=](const std::vector<std::pair<int, double>>& features,
                         double* output) {
        int tid = omp_get_thread_num();
        CopyToPredictBuffer(predict_buf_[tid].data(), features);
        // get feature importances
        boosting_->PredictContrib(predict_buf_[tid].data(), output);
        ClearPredictBuffer(predict_buf_[tid].data(), predict_buf_[tid].size(),
                           features);
      };
      predict_sparse_fun_ = [=](const std::vector<std::pair<int, double>>& features,
                                std::vector<std::unordered_map<int, double>>* output) {
        auto buf = CopyToPredictMap(features);
        // get sparse feature importances
        boosting_->PredictContribByMap(buf, output);
      };

    } else {
      if (is_raw_score) {
        predict_fun_ = [=](const std::vector<std::pair<int, double>>& features,
                           double* output) {
          int tid = omp_get_thread_num();
          if (num_feature_ > kFeatureThreshold &&
              features.size() < KSparseThreshold) {
            auto buf = CopyToPredictMap(features);
            boosting_->PredictRawByMap(buf, output, &early_stop_);
          } else {
            CopyToPredictBuffer(predict_buf_[tid].data(), features);
            boosting_->PredictRaw(predict_buf_[tid].data(), output,
                                  &early_stop_);
            ClearPredictBuffer(predict_buf_[tid].data(),
                               predict_buf_[tid].size(), features);
          }
        };
      } else {
        predict_fun_ = [=](const std::vector<std::pair<int, double>>& features,
                           double* output) {
          int tid = omp_get_thread_num();
          if (num_feature_ > kFeatureThreshold &&
              features.size() < KSparseThreshold) {
            auto buf = CopyToPredictMap(features);
            boosting_->PredictByMap(buf, output, &early_stop_);
          } else {
            CopyToPredictBuffer(predict_buf_[tid].data(), features);
            boosting_->Predict(predict_buf_[tid].data(), output, &early_stop_);
            ClearPredictBuffer(predict_buf_[tid].data(),
                               predict_buf_[tid].size(), features);
          }
        };
      }
    }
  }

  /*!
  * \brief Destructor
  */
  ~Predictor() {
  }

  inline const PredictFunction& GetPredictFunction() const {
    return predict_fun_;
  }


  inline const PredictSparseFunction& GetPredictSparseFunction() const {
    return predict_sparse_fun_;
  }

  /*!
  * \brief predicting on data, then saving result to disk
  * \param data_filename Filename of data
  * \param result_filename Filename of output result
  */
  void Predict(const char* data_filename, const char* result_filename, bool header, bool disable_shape_check, bool precise_float_parser) {
    auto writer = VirtualFileWriter::Make(result_filename);
    if (!writer->Init()) {
      Log::Fatal("Prediction results file %s cannot be created", result_filename);
    }
    auto label_idx = header ? -1 : boosting_->LabelIdx();
    auto parser = std::unique_ptr<Parser>(Parser::CreateParser(data_filename, header, boosting_->MaxFeatureIdx() + 1, label_idx,
                                                               precise_float_parser, boosting_->ParserConfigStr()));

    if (parser == nullptr) {
      Log::Fatal("Could not recognize the data format of data file %s", data_filename);
    }
    if (!header && !disable_shape_check && parser->NumFeatures() != boosting_->MaxFeatureIdx() + 1) {
      Log::Fatal("The number of features in data (%d) is not the same as it was in training data (%d).\n" \
                 "You can set ``predict_disable_shape_check=true`` to discard this error, but please be aware what you are doing.", parser->NumFeatures(), boosting_->MaxFeatureIdx() + 1);
    }
    TextReader<data_size_t> predict_data_reader(data_filename, header);
    std::vector<int> feature_remapper(parser->NumFeatures(), -1);
    bool need_adjust = false;
    // skip raw feature remapping if trained model has parser config str which may contain actual feature names.
    if (header && boosting_->ParserConfigStr().empty()) {
      std::string first_line = predict_data_reader.first_line();
      std::vector<std::string> header_words = Common::Split(first_line.c_str(), "\t,");
      std::unordered_map<std::string, int> header_mapper;
      for (int i = 0; i < static_cast<int>(header_words.size()); ++i) {
        if (header_mapper.count(header_words[i]) > 0) {
          Log::Fatal("Feature (%s) appears more than one time.", header_words[i].c_str());
        }
        header_mapper[header_words[i]] = i;
      }
      const auto& fnames = boosting_->FeatureNames();
      for (int i = 0; i < static_cast<int>(fnames.size()); ++i) {
        if (header_mapper.count(fnames[i]) <= 0) {
          Log::Warning("Feature (%s) is missed in data file. If it is weight/query/group/ignore_column, you can ignore this warning.", fnames[i].c_str());
        } else {
          feature_remapper[header_mapper.at(fnames[i])] = i;
        }
      }
      for (int i = 0; i < static_cast<int>(feature_remapper.size()); ++i) {
        if (feature_remapper[i] >= 0 && i != feature_remapper[i]) {
          need_adjust = true;
          break;
        }
      }
    }
    // function for parse data
    std::function<void(const char*, std::vector<std::pair<int, double>>*)> parser_fun;
    double tmp_label;
    parser_fun = [&parser, &feature_remapper, &tmp_label, need_adjust]
    (const char* buffer, std::vector<std::pair<int, double>>* feature) {
      parser->ParseOneLine(buffer, feature, &tmp_label);
      if (need_adjust) {
        int i = 0, j = static_cast<int>(feature->size());
        while (i < j) {
          if (feature_remapper[(*feature)[i].first] >= 0) {
            (*feature)[i].first = feature_remapper[(*feature)[i].first];
            ++i;
          } else {
            // move the non-used features to the end of the feature vector
            std::swap((*feature)[i], (*feature)[--j]);
          }
        }
        feature->resize(i);
      }
    };

    std::function<void(data_size_t, const std::vector<std::string>&)>
        process_fun = [&parser_fun, &writer, this](
                          data_size_t, const std::vector<std::string>& lines) {
      std::vector<std::pair<int, double>> oneline_features;
      std::vector<std::string> result_to_write(lines.size());
      OMP_INIT_EX();
      #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) firstprivate(oneline_features)
      for (data_size_t i = 0; i < static_cast<data_size_t>(lines.size()); ++i) {
        OMP_LOOP_EX_BEGIN();
        oneline_features.clear();
        // parser
        parser_fun(lines[i].c_str(), &oneline_features);
        // predict
        std::vector<double> result(num_pred_one_row_);
        predict_fun_(oneline_features, result.data());
        auto str_result = Common::Join<double>(result, "\t");
        result_to_write[i] = str_result;
        OMP_LOOP_EX_END();
      }
      OMP_THROW_EX();
      for (data_size_t i = 0; i < static_cast<data_size_t>(result_to_write.size()); ++i) {
        writer->Write(result_to_write[i].c_str(), result_to_write[i].size());
        writer->Write("\n", 1);
      }
    };
    predict_data_reader.ReadAllAndProcessParallel(process_fun);
  }


  // Produces all unique directed pairs (x, y), x != y,
  // such that at least one of x or y is in the selected top_n set.
  // Selection of the top_n set is random via 'rng' (shuffle + truncate).
  void InitializeRandomPairs(data_size_t num_items_in_query,
    std::vector<std::pair<int, int>>& random_pairs,
    int top_n,
    std::mt19937& rng) {
    const std::size_t N = num_items_in_query;
    if (N < 2 || top_n <= 0) {
      return;  // no pairs possible
    }

    // Build indices 0..N-1 and shuffle
    std::vector<int> nums(static_cast<std::size_t>(N));
    std::iota(nums.begin(), nums.end(), 0);
    std::shuffle(nums.begin(), nums.end(), rng);

    // Clamp top_n to N and take the first top_n as the selected set S
    const int M = std::min<int>(top_n, static_cast<int>(N));
    nums.resize(static_cast<std::size_t>(M));

    // Mark membership in S for O(1) checks
    std::vector<unsigned char> in_top(N, 0);
    for (int x : nums) {
      in_top[static_cast<std::size_t>(x)] = 1;
    }

    // Reserve exact capacity to avoid reallocations:
    // total = N*(N-1) - (K*(K-1)), where K = N - M
    const std::size_t K = N - static_cast<std::size_t>(M);
    const std::size_t total_pairs =
      N * (N - 1) - K * (K - 1);  // directed pairs with at least one endpoint in S
    random_pairs.reserve(random_pairs.size() + total_pairs);

    // 1) Emit S × U (minus diagonal)
    for (int x : nums) {  // x in S
      const std::size_t xs = static_cast<std::size_t>(x);
      for (std::size_t y = 0; y < N; ++y) {
        if (y == xs) continue;  // skip diagonal
        random_pairs.emplace_back(x, static_cast<int>(y));
      }
    }

    // 2) Emit (U \ S) × S (these are not duplicates of step 1)
    for (std::size_t x = 0; x < N; ++x) {
      if (in_top[x]) continue;  // only x outside S
      for (int y : nums) {      // y in S
        if (static_cast<std::size_t>(y) == x) continue;  // diagonal guard (redundant because x !in S)
        random_pairs.emplace_back(static_cast<int>(x), y);
      }
    }

    // At this point, random_pairs contains exactly all (x, y), x != y,
    // with (x in S) OR (y in S) — no duplicates.
  }

  void UpdateMapsForNewPairs(data_size_t previous_pairs_count, std::vector<std::pair<int, int>>& pairs, std::vector<std::vector<std::pair<short, data_size_t>>>& right2left2pair_map,
    std::vector<std::vector<std::pair<short, data_size_t>>>& left2right2pair_map) {
    for (data_size_t i = previous_pairs_count; i < pairs.size(); ++i) {
      //data_size_t current_pair = selected_pairs[i];
      short index_left = pairs[i].first;
      short index_right = pairs[i].second;

      left2right2pair_map[index_left].emplace_back(index_right, i);   // neighbor, pair index
      right2left2pair_map[index_right].emplace_back(index_left, i);   // reverse mapping
    }

    // sort inner vectors by neighbor index for binary search
    auto sort_by_neighbor = [](std::vector<std::pair<short, data_size_t>>& vec) {
      std::sort(vec.begin(), vec.end(),
        [](const auto& a, const auto& b) { return a.first < b.first; });
      };

    for (auto& vec : left2right2pair_map) sort_by_neighbor(vec);
    for (auto& vec : right2left2pair_map) sort_by_neighbor(vec);
  }

  std::vector<int> sort_by_noisy_scores(const std::vector<double>& scores_pointwise, double sigma, std::mt19937& rng) {
    const int n = scores_pointwise.size();
    std::vector<int> indices(n);
    for (int i = 0; i < n; ++i) indices[i] = i; // [0..n-1]

    std::uniform_real_distribution<double> dist(0.0, sigma);

    // Precompute noisy scores (O(n))
    std::vector<double> noisy_scores(n);
    for (int i = 0; i < n; ++i) {
      noisy_scores[i] = scores_pointwise[i] + dist(rng);
    }

    // Sort indices by noisy score, descending
    std::sort(indices.begin(), indices.end(), [&](int a, int b) { return noisy_scores[a] > noisy_scores[b]; });
    return indices;
  }

  void ComputeNewPairs(std::vector<std::pair<data_size_t, data_size_t>>& pairs,
    const std::vector<double>& scores_pointwise, const std::vector<std::vector<std::pair<short, data_size_t>>>& left2right2pair_map,
    double sigma, std::mt19937& rng, int top_n, int top_pairs_k) {


    // 1. Get indices sorted by score + noise
    auto sorted_indices = sort_by_noisy_scores(scores_pointwise, sigma, rng);

    // 2. Iterate over top_n left elements

    if (top_n > static_cast<int>(sorted_indices.size())) {
      top_n = static_cast<int>(sorted_indices.size());
    }

    int pairs_old_size = pairs.size();

    for (int i = 0; i < top_n; ++i) {
      data_size_t left = sorted_indices[i];

      const auto& paired_rights = left2right2pair_map[left]; // sorted by right index
      int collected = 0;      

      for (int idx = 0; idx < (int)sorted_indices.size() && collected < top_pairs_k; ++idx) {
        data_size_t right = sorted_indices[idx];
        if (right == left) continue; // skip self

        // Binary search in paired_rights to check if 'right' is already paired
        if (get_pair_index(paired_rights, right) != static_cast<data_size_t>(-1)) continue; // forbidden

        bool already_added = false;
        for (int j = pairs_old_size; j < pairs.size(); j++) {
          if ((pairs[j].first == right && pairs[j].second == left) || (pairs[j].first == left && pairs[j].second == right)) {
            already_added = true;
            break;
          }
        }
        if (already_added) continue; // forbidden

        // Accept this pair
        pairs.emplace_back(left, right);
        pairs.emplace_back(right, left);
        ++collected;
      }
    }
  }


  // Sort indices by raw scores (descending). No NaN per your assumption.
  static inline std::vector<int>
    sort_by_scores_desc(const std::vector<double>& scores_pointwise) {
    const std::size_t n = scores_pointwise.size();
    std::vector<int> idx(static_cast<int>(n));
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
      [&](int a, int b) {
        return scores_pointwise[static_cast<std::size_t>(a)]
        > scores_pointwise[static_cast<std::size_t>(b)];
      });
    return idx;
  }

  /**
   * ComputeNewPairs:
   * - Scan docs by descending scores_pointwise.
   * - Take up to 'top_n' docs that are NOT fully connected (degree < N-1).
   * - For each selected doc 'left', add ALL missing pairs involving it.
   * - No "seen" set; only rely on left2right2pair_map lookups.
   * - To avoid intra-call duplicates when inserting the reverse orientation,
   *   only add (right, left) if rank_pos[left] < rank_pos[right].
   */
  void ComputeNewPairs(std::vector<std::pair<data_size_t, data_size_t>>& pairs,
    const std::vector<double>& scores_pointwise,
    const std::vector<std::vector<std::pair<short, data_size_t>>>& left2right2pair_map,
    int top_n) {
    const std::size_t N = scores_pointwise.size();
    if (N <= 1 || top_n <= 0) return;

    // Safety: helper uses short keys; ensure indices fit (adjust if needed).
    if (N > static_cast<std::size_t>(std::numeric_limits<short>::max())) {
      // If this can happen in your environment, widen the helper key type.
      // For now, clamp top_n to prevent out-of-range keys.
      top_n = std::min<int>(top_n, static_cast<int>(std::numeric_limits<short>::max()));
    }

    // 1) Rank by raw scores (descending).
    std::vector<int> ranked = sort_by_scores_desc(scores_pointwise);

    // Build rank positions for duplicate-safe reverse insertion:
    // rank_pos[doc_id] = position in 'ranked'
    std::vector<int> rank_pos(N, -1);
    for (int pos = 0; pos < static_cast<int>(ranked.size()); ++pos) {
      rank_pos[static_cast<std::size_t>(ranked[pos])] = pos;
    }

    // 2) Select up to 'top_n' docs that are NOT fully connected.
    std::vector<data_size_t> selected;
    selected.reserve(static_cast<std::size_t>(std::min<int>(top_n, static_cast<int>(N))));
    for (int pos = 0; pos < static_cast<int>(ranked.size()) && static_cast<int>(selected.size()) < top_n; ++pos) {
      const data_size_t left = static_cast<data_size_t>(ranked[pos]);
      const auto& adj = left2right2pair_map[left];
      if (adj.size() < (N - 1)) {
        selected.push_back(left);
      }
    }
    if (selected.empty()) return;

    // 3) For each selected 'left', add all missing pairs.
    for (data_size_t left : selected) {
      const auto& adj_left = left2right2pair_map[left];

      for (data_size_t right = 0; right < N; ++right) {
        if (right == left) continue;

        // (left, right): add if missing in adjacency
        const data_size_t lr_idx =
          get_pair_index(adj_left, static_cast<short>(right));
        if (lr_idx == static_cast<data_size_t>(-1)) {
          pairs.emplace_back(left, right);
        }

        // (right, left): add if missing in adjacency on 'right' AND
        // honor rank-order rule to avoid intra-call duplicates.
        const auto& adj_right = left2right2pair_map[right];
        const data_size_t rl_idx =
          get_pair_index(adj_right, static_cast<short>(left));
        if (rl_idx == static_cast<data_size_t>(-1)) {
          if (rank_pos[left] < rank_pos[right]) {
            pairs.emplace_back(right, left);
          }
          // If rank_pos[right] < rank_pos[left], the symmetric insertion
          // will happen when 'right' is (or becomes) a selected 'left'.
        }
      }
    }
  }

  inline std::string VectorPairsToString(const std::vector<std::pair<data_size_t, data_size_t>>& pair_indices) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < pair_indices.size(); ++i) {
      oss << "(" << pair_indices[i].first << "," << pair_indices[i].second << ")";
      if (i + 1 < pair_indices.size()) {
        oss << ", ";
      }
    }
    oss << "]";
    return oss.str();
  }


  inline std::string MapToString(const std::vector<std::vector<std::pair<short, data_size_t>>>& map) {
    std::ostringstream oss;
    oss << "{";
    for (size_t i = 0; i < map.size(); ++i) {
      if (!map[i].empty()) {
        oss << "\n  [" << i << "] -> {";
        for (size_t j = 0; j < map[i].size(); ++j) {
          oss << "(" << map[i][j].first << "," << map[i][j].second << ")";
          if (j + 1 < map[i].size()) oss << ", ";
        }
        oss << "}";
      }
    }
    oss << "\n}";
    return oss.str();
  }


  inline std::string VectorToString(const std::vector<double>& vec) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
      oss << vec[i];
      if (i + 1 < vec.size()) {
        oss << ", ";
      }
    }
    oss << "]";
    return oss.str();
  }




  /*!
  * \brief predicting on data, then saving result to disk
  * \brief used only in ``pairwise_lambdarank`` objective
  * \param data_filename Filename of data
  * \param result_filename Filename of output result
  */
  void PredictPairwise(const Config& config) {

    const double indirect_comparison_weight = config.pairwise_lambdarank_model_indirect_comparison ? config.pairwise_lambdarank_indirect_comparison_weight : 0.0;

    const char* data_filename = config.data.c_str();
    const char* result_filename = config.output_result.c_str();
    std::unique_ptr<Metadata> metadata(new Metadata());
    metadata->Init(data_filename);
    auto writer = VirtualFileWriter::Make(result_filename);
    if (!writer->Init()) {
      Log::Fatal("Prediction results file %s cannot be created", result_filename);
    }

    TextReader<data_size_t> predict_data_reader(data_filename, config.header);
    std::string first_line = predict_data_reader.first_line();
    std::vector<std::string> header_words;
    std::unordered_map<std::string, int> pointwise_header_mapper;
    if (config.header) {
      header_words = Common::Split(first_line.c_str(), "\t,");
      for (int i = 0; i < static_cast<int>(header_words.size()); ++i) {
        if (pointwise_header_mapper.count(header_words[i]) > 0) {
          Log::Fatal("Feature (%s) appears more than one time.", header_words[i].c_str());
        }
        pointwise_header_mapper[header_words[i]] = i;
      }
    }

    auto label_idx = -1;

    if (boosting_->ParserConfigStr().empty()) {
      if (!config.header) {
        label_idx = boosting_->LabelIdx();
      } else {
        std::string label_name = std::string("");
        ParseLabelColumnIdx(pointwise_header_mapper, config.label_column, &label_idx, &label_name);

        if (label_idx >= 0) {
          // label will be separated with features when parsing a line
          std::string label_name_in_header = std::string("");
          for (auto& pair : pointwise_header_mapper) {
            if (pair.second > label_idx) {
              --pair.second;
            } else if (pair.second == label_idx) {
              label_name_in_header = pair.first;
            }
          }
          CHECK(pointwise_header_mapper.count(label_name_in_header) > 0);
          pointwise_header_mapper.erase(label_name_in_header);
          if (!label_name.empty()) {
            CHECK_EQ(label_name, label_name_in_header);
          }
        }

      }
    } else {
      Log::Fatal("Unsupported with the combination of pairwise ranking and parser config str.");
    }

    auto parser = std::unique_ptr<Parser>(Parser::CreateParser(data_filename, config.header, boosting_->MaxFeatureIdx() + 1, label_idx,
                                                               config.precise_float_parser, boosting_->ParserConfigStr()));

    if (parser == nullptr) {
      Log::Fatal("Could not recognize the data format of data file %s", data_filename);
    }

    int num_total_features = 0;
    if (config.use_differential_feature_in_pairwise_ranking) {
      num_total_features = 3 * parser->NumFeatures();
    } else {
      num_total_features = 2 * parser->NumFeatures();
    }

    if (!config.header && !config.predict_disable_shape_check && num_total_features != boosting_->MaxFeatureIdx() + 1) {
      Log::Fatal("The number of features in data (%d) is not the same as it was in training data (%d).\n" \
                 "You can set ``predict_disable_shape_check=true`` to discard this error, but please be aware what you are doing.", num_total_features, boosting_->MaxFeatureIdx() + 1);
    }
    std::vector<int> feature_remapper(num_total_features, -1);
    bool need_adjust = false;
    // skip raw feature remapping if trained model has parser config str which may contain actual feature names.
    if (config.header && boosting_->ParserConfigStr().empty()) {
      std::unordered_map<std::string, int> header_mapper;
      for (const auto& pair : pointwise_header_mapper) {
        header_mapper[pair.first + std::string("_i")] = pair.second;
      }
      for (const auto& pair : pointwise_header_mapper) {
        header_mapper[pair.first + std::string("_j")] = pair.second + parser->NumFeatures();
      }
      if (config.use_differential_feature_in_pairwise_ranking) {
        for (const auto& pair : pointwise_header_mapper) {
          header_mapper[pair.first + std::string("_k")] = pair.second + 2 * parser->NumFeatures();
        }
      }

      const auto& fnames = boosting_->FeatureNames();
      for (int i = 0; i < static_cast<int>(fnames.size()); ++i) {
        if (header_mapper.count(fnames[i]) <= 0) {
          Log::Warning("Feature (%s) is missed in data file. If it is weight/query/group/ignore_column, you can ignore this warning.", fnames[i].c_str());
        } else {
          feature_remapper[header_mapper.at(fnames[i])] = i;
        }
      }
      for (int i = 0; i < static_cast<int>(feature_remapper.size()); ++i) {
        if (feature_remapper[i] >= 0 && i != feature_remapper[i]) {
          need_adjust = true;
          break;
        }
      }
    }

    // try to parse query_id from data file
    ParseQueryGroupBoundaries(pointwise_header_mapper, config.group_column, metadata, config, data_filename, parser);

    // function for parse data
    std::function<void(const char*, std::vector<std::pair<int, double>>*)> parser_fun;
    double tmp_label;
    parser_fun = [&parser, &feature_remapper, &tmp_label, need_adjust, &config]
    (const char* buffer, std::vector<std::pair<int, double>>* feature) {
      parser->ParseOneLine(buffer, feature, &tmp_label);
      if (need_adjust) {
        int i = 0, j = static_cast<int>(feature->size());
        while (i < j) {
          if (feature_remapper[(*feature)[i].first] >= 0) {
            (*feature)[i].first = feature_remapper[(*feature)[i].first];
            ++i;
          } else {
            // move the non-used features to the end of the feature vector
            std::swap((*feature)[i], (*feature)[--j]);
          }
        }
        feature->resize(i);

        if (config.use_differential_feature_in_pairwise_ranking) {
          // need sorting, since differential feature calculation requires feature indices in ascending ordering
          std::sort(feature->begin(), feature->end(), [] (std::pair<int, double> pair_1, std::pair<int, double> pair_2) {
            return pair_1.first < pair_2.first;
          });
        }

      }
    };

    const data_size_t* query_boundaries = metadata->query_boundaries();
    const data_size_t num_queries = metadata->num_queries();
    const int num_features = parser->NumFeatures();

    CommonC::SigmoidCache sigmoid_cache;
    sigmoid_cache.Init(config.sigmoid);

    auto label_gain_copy = config.label_gain;
    DCGCalculator::DefaultLabelGain(&label_gain_copy);
    DCGCalculator::Init(label_gain_copy);


    // Use query-aware processing for ranking tasks
    std::function<void(data_size_t, data_size_t, const std::vector<std::string>&)>
        process_fun_by_query = [&config, sigmoid_cache, indirect_comparison_weight, num_features, &parser_fun, &writer, this](
                                  data_size_t query_idx, data_size_t query_start, const std::vector<std::string>& lines) {

      // Get num items in this query
      const data_size_t num_items_in_query = static_cast<data_size_t>(lines.size());

      // Parse all the data in the current query
      std::vector<std::vector<std::pair<int, double>>> oneline_features_in_query(lines.size());
      OMP_INIT_EX();
      #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
      for (data_size_t i = 0; i < num_items_in_query; ++i) {
        OMP_LOOP_EX_BEGIN();
        // parser
        parser_fun(lines[i].c_str(), &oneline_features_in_query[i]);
        OMP_LOOP_EX_END();
      }
      OMP_THROW_EX();

      // Prepare for final prediction results
      std::vector<std::string> result_to_write(lines.size());

      // result buffer in each iteration
      std::vector<double> result;

      // list of paired indices in this query, in range [0, lines.size() - 1]
      std::vector<std::pair<data_size_t, data_size_t>> pair_indices;
      data_size_t old_num_pairs = static_cast<data_size_t>(pair_indices.size()); // 0 at first iteration
      
      std::vector<double> scores_pointwise(num_items_in_query, 0.0);
      std::vector<std::vector<std::pair<short, data_size_t>>> right2left2pair_map(num_items_in_query);
      std::vector<std::vector<std::pair<short, data_size_t>>> left2right2pair_map(num_items_in_query);

      // RNG setup
      static thread_local std::mt19937 rng(std::random_device{}());
      //Log::Info((std::string("pair_indices = ") + VectorPairsToString(pair_indices)).c_str());
      //Log::Info("InitializeRandomPairs");
      InitializeRandomPairs(num_items_in_query, pair_indices, config.pairwise_lambdarank_prediction_pairing_top_n, rng);
      //Log::Info((std::string("pair_indices = ") + VectorPairsToString(pair_indices)).c_str());
      UpdateMapsForNewPairs(0, pair_indices, right2left2pair_map, left2right2pair_map);
      //Log::Info((std::string("right2left2pair_map = ") + MapToString(right2left2pair_map)).c_str());
      //Log::Info((std::string("left2right2pair_map = ") + MapToString(left2right2pair_map)).c_str());


      for (int k = 0; k < config.pairwise_lambdarank_prediction_num_iteration; ++k) {
        //Log::Info("iteration=" + k);
        if (k > 0) {
          old_num_pairs = static_cast<data_size_t>(pair_indices.size());
          //Log::Info("ComputeNewPairs");
          //ComputeNewPairs(pair_indices, scores_pointwise, left2right2pair_map, config.pairwise_lambdarank_prediction_shuffle_sigma, rng, config.pairwise_lambdarank_prediction_pairing_top_n, config.pairwise_lambdarank_prediction_pairing_top_pairs_k);
          ComputeNewPairs(pair_indices, scores_pointwise, left2right2pair_map, config.pairwise_lambdarank_prediction_pairing_top_n);
          //Log::Info((std::string("pair_indices = ") + VectorPairsToString(pair_indices)).c_str());
          UpdateMapsForNewPairs(old_num_pairs, pair_indices, right2left2pair_map, left2right2pair_map);
          //Log::Info((std::string("right2left2pair_map = ") + MapToString(right2left2pair_map)).c_str());
          //Log::Info((std::string("left2right2pair_map = ") + MapToString(left2right2pair_map)).c_str());
        }

        // resize result vector
        // Log::Warning("pair_indices[%d].first = %d, pair_indices[%d].second = %d", i, pair_indices[i].first, i, pair_indices[i].second);
        result.resize(pair_indices.size() * num_pred_one_row_);
        OMP_INIT_EX();
        #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
        for (data_size_t i = old_num_pairs; i < static_cast<data_size_t>(pair_indices.size()); ++i) {
          OMP_LOOP_EX_BEGIN();

          // concatenate features from the paired instances
          std::vector<std::pair<int, double>> oneline_features;
          for (const auto& pair : oneline_features_in_query[pair_indices[i].first]) {
            oneline_features.push_back(std::make_pair(pair.first, pair.second));
          }
          for (const auto& pair : oneline_features_in_query[pair_indices[i].second]) {
            oneline_features.push_back(std::make_pair(pair.first + num_features, pair.second));
          }
          if (config.use_differential_feature_in_pairwise_ranking) {
            std::set<int> feature_set;
            for (const auto& pair : oneline_features_in_query[pair_indices[i].first]) {
              feature_set.insert(pair.first);
            }
            for (const auto& pair : oneline_features_in_query[pair_indices[i].second]) {
              feature_set.insert(pair.first);
            }
            std::vector<std::pair<int, double>>::iterator first_feature_iterator = oneline_features_in_query[pair_indices[i].first].begin();
            std::vector<std::pair<int, double>>::iterator second_feature_iterator = oneline_features_in_query[pair_indices[i].second].begin();
            const std::vector<std::pair<int, double>>::iterator first_feature_iterator_end = oneline_features_in_query[pair_indices[i].first].end();
            const std::vector<std::pair<int, double>>::iterator second_feature_iterator_end = oneline_features_in_query[pair_indices[i].second].end();
            // elements are sorted in order in feature_set, which is an ordered set
            for (const int feature : feature_set) {
              double val1 = 0.0f, val2 = 0.0f;
              while (first_feature_iterator != first_feature_iterator_end && first_feature_iterator->first < feature) {
                ++first_feature_iterator;
              }
              if (first_feature_iterator < first_feature_iterator_end && first_feature_iterator->first == feature) {
                val1 = first_feature_iterator->second;
              }
              while (second_feature_iterator != second_feature_iterator_end && second_feature_iterator->first < feature) {
                ++second_feature_iterator;
              }
              if (second_feature_iterator < second_feature_iterator_end && second_feature_iterator->first == feature) {
                val2 = second_feature_iterator->second;
              }
              // Log::Warning("feature = %d, val1 = %f, val2 = %f", feature, val1, val2);
              oneline_features.push_back(std::make_pair(feature + 2 * num_features, val1 - val2));
            }
          }

          predict_fun_(oneline_features, result.data() + i * num_pred_one_row_);
          OMP_LOOP_EX_END();          
        }
        OMP_THROW_EX();

        for (int i = 0; i < config.pairwise_lambdarank_prediction_pointwise_updates_per_iteration; i++) {
          UpdatePointwiseScoresForOneQuery(scores_pointwise.data(), result.data(), scores_pointwise.size(),
            pair_indices.data(), right2left2pair_map, left2right2pair_map, config.lambdarank_truncation_level, config.sigmoid, sigmoid_cache, config.pairwise_lambdarank_model_indirect_comparison,
            config.pairwise_lambdarank_model_conditional_rel, config.pairwise_lambdarank_indirect_comparison_above_only, config.pairwise_lambdarank_logarithmic_discounts, config.pairwise_lambdarank_hard_pairwise_preference,
            config.pairwise_lambdarank_indirect_comparison_max_rank, indirect_comparison_weight, config.pairwise_lambdarank_l2_pairwise_diff_weight);
        }
        //Log::Info((std::string("result = ") + VectorToString(result)).c_str());
        //Log::Info((std::string("scores_pointwise = ") + VectorToString(scores_pointwise)).c_str());
      }


      #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
      for (data_size_t i = 0; i < static_cast<data_size_t>(result_to_write.size()); ++i) {   
        result_to_write[i] = std::to_string(scores_pointwise[i]);
      }

      // Write results for this query
      for (data_size_t i = 0; i < static_cast<data_size_t>(result_to_write.size()); ++i) {
        writer->Write(result_to_write[i].c_str(), result_to_write[i].size());
        writer->Write("\n", 1);
      }
    };

    predict_data_reader.ReadAllAndProcessParallelByQuery(process_fun_by_query, query_boundaries, num_queries);
  }

 private:
  int ParseGroupColumnIdx(const std::unordered_map<std::string, int>& header_mapper, std::string group_column) {
    std::string name_prefix("name:");
    int group_idx = -1;
    if (group_column.size() > 0) {
      if (Common::StartsWith(group_column, name_prefix)) {
        std::string name = group_column.substr(name_prefix.size());
        if (header_mapper.count(name) > 0) {
          group_idx = header_mapper.at(name);
          Log::Info("Using column %s as group/query id", name.c_str());
        } else {
          Log::Fatal("Could not find group/query column %s in data file", name.c_str());
        }
      } else {
        if (!Common::AtoiAndCheck(group_column.c_str(), &group_idx)) {
          Log::Fatal("group_column is not a number,\n"
                     "if you want to use a column name,\n"
                     "please add the prefix \"name:\" to the column name");
        }
        Log::Info("Using column number %d as group/query id", group_idx);
      }
    }

    return group_idx;
  }

  void ParseLabelColumnIdx(const std::unordered_map<std::string, int>& header_mapper, std::string label_column, int* label_idx, std::string* name) {
    const std::string name_prefix("name:");
    *label_idx = -1;
    if (label_column.size() > 0) {
      if (Common::StartsWith(label_column, name_prefix)) {
        *name = label_column.substr(name_prefix.size());
        *label_idx = -1;
        if (header_mapper.count(*name) > 0) {
          *label_idx = header_mapper.at(*name);
          Log::Info("Using column %s as label", name->c_str());
        } else {
          Log::Fatal("Could not find label column %s in data file \n"
                      "or data file doesn't contain header", name->c_str());
        }
      } else {
        if (!Common::AtoiAndCheck(label_column.c_str(), label_idx)) {
          Log::Fatal("label_column is not a number,\n"
                      "if you want to use a column name,\n"
                      "please add the prefix \"name:\" to the column name");
        }
        Log::Info("Using column number %d as label", label_idx);
      }
    }
  }


  void ParseQueryGroupBoundaries(const std::unordered_map<std::string, int>& header_mapper, const std::string group_column, const std::unique_ptr<Metadata>& metadata, const Config& config, const char* data_filename, const std::unique_ptr<Parser>& parser) {
    const int group_column_idx = ParseGroupColumnIdx(header_mapper, group_column);

    if (group_column_idx >= 0) {
      if (metadata->query_load_from_file()) {
        Log::Info("Using query id in data file, ignoring the additional query file");
      }

      // use dataset loader
      DatasetLoader loader(config, nullptr, config.num_class, data_filename);

      if (Network::num_machines() == 1) {
        data_size_t global_num_data = 0;
        std::vector<data_size_t> used_data_indices;
        const std::vector<std::string> text_lines = loader.LoadTextDataToMemory(data_filename, *metadata.get(), 0, 1, &global_num_data, &used_data_indices);
        const int num_threads = OMP_NUM_THREADS();
        const data_size_t num_lines = static_cast<data_size_t>(text_lines.size());
        metadata->Init(num_lines, -1, group_column_idx);
        #pragma omp parallel for schedule(static) num_threads(num_threads)
        for (data_size_t i = 0; i < num_lines; ++i) { // skipping header
          const std::string& line = text_lines[i];
          std::vector<std::pair<int, double>> oneline_feature;
          double out_label = 0.0;
          parser->ParseOneLine(line.c_str(), &oneline_feature, &out_label);
          bool query_id_found = false;
          int query_id = -1;
          for (const auto& pair : oneline_feature) {
            if (pair.first == group_column_idx) {
              query_id_found = true;
              query_id = static_cast<int>(pair.second);
              if (query_id < 0) {
                Log::Fatal("Invalid query_id %d found", query_id);
              }
              break;
            }
          }

          if (!query_id_found) {
            query_id = 0;
          }

          metadata->SetQueryAt(i, query_id);
        }

        metadata->FinishLoad();
      } else {
        Log::Fatal("Pairwise ranking prediction now supports only single machine mode.");
      }
    }
  }


  void CopyToPredictBuffer(double* pred_buf, const std::vector<std::pair<int, double>>& features) {
    for (const auto &feature : features) {
      if (feature.first < num_feature_) {
        pred_buf[feature.first] = feature.second;
      }
    }
  }

  void ClearPredictBuffer(double* pred_buf, size_t buf_size, const std::vector<std::pair<int, double>>& features) {
    if (features.size() > static_cast<size_t>(buf_size / 2)) {
      std::memset(pred_buf, 0, sizeof(double)*(buf_size));
    } else {
      for (const auto &feature : features) {
        if (feature.first < num_feature_) {
          pred_buf[feature.first] = 0.0f;
        }
      }
    }
  }

  std::unordered_map<int, double> CopyToPredictMap(const std::vector<std::pair<int, double>>& features) {
    std::unordered_map<int, double> buf;
    for (const auto &feature : features) {
      if (feature.first < num_feature_) {
        buf[feature.first] = feature.second;
      }
    }
    return buf;
  }

  /*! \brief Boosting model */
  const Boosting* boosting_;
  /*! \brief function for prediction */
  PredictFunction predict_fun_;
  PredictSparseFunction predict_sparse_fun_;
  PredictionEarlyStopInstance early_stop_;
  int num_feature_;
  int num_pred_one_row_;
  std::vector<std::vector<double, Common::AlignmentAllocator<double, kAlignedSize>>> predict_buf_;
};

}  // namespace LightGBM

#endif  // LIGHTGBM_SRC_APPLICATION_PREDICTOR_HPP_
