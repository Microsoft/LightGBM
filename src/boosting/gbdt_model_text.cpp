/*!
 * Copyright (c) 2017 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <LightGBM/config.h>
#include <LightGBM/metric.h>
#include <LightGBM/objective_function.h>
#include <LightGBM/utils/array_args.h>
#include <LightGBM/utils/common.h>

#include <string>
#include <sstream>
#include <vector>

#include "gbdt.h"

namespace LightGBM {

const char* kModelVersion = "v4";

std::string GBDT::DumpModel(int start_iteration, int num_iteration, int feature_importance_type) const {
  std::stringstream str_buf;
  Common::C_stringstream(str_buf);

  str_buf << "{";
  str_buf << "\"name\":\"" << SubModelName() << "\"," << '\n';
  str_buf << "\"version\":\"" << kModelVersion << "\"," << '\n';
  str_buf << "\"num_class\":" << num_class_ << "," << '\n';
  str_buf << "\"num_tree_per_iteration\":" << num_tree_per_iteration_ << "," << '\n';
  str_buf << "\"label_index\":" << label_idx_ << "," << '\n';
  str_buf << "\"max_feature_idx\":" << max_feature_idx_ << "," << '\n';
  if (objective_function_ != nullptr) {
    str_buf << "\"objective\":\"" << objective_function_->ToString() << "\",\n";
  }

  str_buf << "\"average_output\":" << (average_output_ ? "true" : "false") << ",\n";

  str_buf << "\"feature_names\":[\"" << CommonC::Join(feature_names_, "\",\"")
          << "\"]," << '\n';

  str_buf << "\"monotone_constraints\":["
          << CommonC::Join(monotone_constraints_, ",") << "]," << '\n';

  if (category_encoding_provider_ != nullptr) {
    str_buf << "\"category_encoding_provider\":"
            << category_encoding_provider_->DumpToJSON() << "," << '\n';
  } else {
    str_buf << "\"category_encoding_provider\":null,\n";
  }

  str_buf << "\"feature_infos\":" << "{";
  bool first_obj = true;
  for (size_t i = 0; i < feature_infos_.size(); ++i) {
    std::stringstream json_str_buf;
    Common::C_stringstream(json_str_buf);
    auto strs = Common::Split(feature_infos_[i].c_str(), ":");
    if (strs[0][0] == '[') {
      strs[0].erase(0, 1);  // remove '['
      strs[1].erase(strs[1].size() - 1);  // remove ']'
      double max_, min_;
      Common::Atof(strs[0].c_str(), &min_);
      Common::Atof(strs[1].c_str(), &max_);
      json_str_buf << std::setprecision(std::numeric_limits<double>::digits10 + 2);
      json_str_buf << "{\"min_value\":" << Common::AvoidInf(min_) << ",";
      json_str_buf << "\"max_value\":" << Common::AvoidInf(max_) << ",";
      json_str_buf << "\"values\":[]}";
    } else if (strs[0] != "none") {  // categorical feature
      auto vals = CommonC::StringToArray<int>(feature_infos_[i], ':');
      auto max_idx = ArrayArgs<int>::ArgMax(vals);
      auto min_idx = ArrayArgs<int>::ArgMin(vals);
      json_str_buf << "{\"min_value\":" << vals[min_idx] << ",";
      json_str_buf << "\"max_value\":" << vals[max_idx] << ",";
      json_str_buf << "\"values\":[" << CommonC::Join(vals, ",") << "]}";
    } else {  // unused feature
      continue;
    }
    if (!first_obj) {
      str_buf << ",";
    }
    str_buf << "\"" << feature_names_[i] << "\":";
    str_buf << json_str_buf.str();
    first_obj = false;
  }
  str_buf << "}," << '\n';

  str_buf << "\"tree_info\":[";
  int num_used_model = static_cast<int>(models_.size());
  int total_iteration = num_used_model / num_tree_per_iteration_;
  start_iteration = std::max(start_iteration, 0);
  start_iteration = std::min(start_iteration, total_iteration);
  if (num_iteration > 0) {
    int end_iteration = start_iteration + num_iteration;
    num_used_model = std::min(end_iteration * num_tree_per_iteration_ , num_used_model);
  }
  int start_model = start_iteration * num_tree_per_iteration_;
  for (int i = start_model; i < num_used_model; ++i) {
    if (i > start_model) {
      str_buf << ",";
    }
    str_buf << "{";
    str_buf << "\"tree_index\":" << i << ",";
    str_buf << models_[i]->ToJSON();
    str_buf << "}";
  }
  str_buf << "]," << '\n';

  std::vector<double> feature_importances = FeatureImportance(
      num_iteration, feature_importance_type);
  // store the importance first
  std::vector<std::pair<size_t, std::string>> pairs;
  for (size_t i = 0; i < feature_importances.size(); ++i) {
    size_t feature_importances_int = static_cast<size_t>(feature_importances[i]);
    if (feature_importances_int > 0) {
      pairs.emplace_back(feature_importances_int, feature_names_[i]);
    }
  }
  str_buf << '\n' << "\"feature_importances\":" << "{";
  for (size_t i = 0; i < pairs.size(); ++i) {
    if (i > 0) {
      str_buf << ",";
    }
    str_buf << "\"" << pairs[i].second << "\":" << std::to_string(pairs[i].first);
  }
  str_buf << "}" << '\n';

  str_buf << "}" << '\n';

  return str_buf.str();
}

std::string GBDT::ModelToIfElse(int num_iteration) const {
  std::stringstream str_buf;
  Common::C_stringstream(str_buf);

  str_buf << "#include \"gbdt.h\"" << '\n';
  str_buf << "#include <LightGBM/utils/common.h>" << '\n';
  str_buf << "#include <LightGBM/objective_function.h>" << '\n';
  str_buf << "#include <LightGBM/metric.h>" << '\n';
  str_buf << "#include <LightGBM/prediction_early_stop.h>" << '\n';
  str_buf << "#include <ctime>" << '\n';
  str_buf << "#include <sstream>" << '\n';
  str_buf << "#include <chrono>" << '\n';
  str_buf << "#include <string>" << '\n';
  str_buf << "#include <vector>" << '\n';
  str_buf << "#include <utility>" << '\n';
  str_buf << "namespace LightGBM {" << '\n';

  int num_used_model = static_cast<int>(models_.size());
  if (num_iteration > 0) {
    num_used_model = std::min(num_iteration * num_tree_per_iteration_, num_used_model);
  }

  // PredictRaw
  for (int i = 0; i < num_used_model; ++i) {
    str_buf << models_[i]->ToIfElse(i, false) << '\n';
  }

  str_buf << "double (*PredictTreePtr[])(const double*) = { ";
  for (int i = 0; i < num_used_model; ++i) {
    if (i > 0) {
      str_buf << " , ";
    }
    str_buf << "PredictTree" << i;
  }
  str_buf << " };" << '\n' << '\n';

  std::stringstream pred_str_buf;
  Common::C_stringstream(pred_str_buf);

  pred_str_buf << "\t" << "int early_stop_round_counter = 0;" << '\n';
  pred_str_buf << "\t" << "std::memset(output, 0, sizeof(double) * num_tree_per_iteration_);" << '\n';
  pred_str_buf << "\t" << "for (int i = 0; i < num_iteration_for_pred_; ++i) {" << '\n';
  pred_str_buf << "\t\t" << "for (int k = 0; k < num_tree_per_iteration_; ++k) {" << '\n';
  pred_str_buf << "\t\t\t" << "output[k] += (*PredictTreePtr[i * num_tree_per_iteration_ + k])(features);" << '\n';
  pred_str_buf << "\t\t" << "}" << '\n';
  pred_str_buf << "\t\t" << "++early_stop_round_counter;" << '\n';
  pred_str_buf << "\t\t" << "if (early_stop->round_period == early_stop_round_counter) {" << '\n';
  pred_str_buf << "\t\t\t" << "if (early_stop->callback_function(output, num_tree_per_iteration_))" << '\n';
  pred_str_buf << "\t\t\t\t" << "return;" << '\n';
  pred_str_buf << "\t\t\t" << "early_stop_round_counter = 0;" << '\n';
  pred_str_buf << "\t\t" << "}" << '\n';
  pred_str_buf << "\t" << "}" << '\n';

  str_buf << "void GBDT::PredictRaw(const double* features, double *output, const PredictionEarlyStopInstance* early_stop) const {" << '\n';
  str_buf << pred_str_buf.str();
  str_buf << "}" << '\n';
  str_buf << '\n';

  // PredictRawByMap
  str_buf << "double (*PredictTreeByMapPtr[])(const std::unordered_map<int, double>&) = { ";
  for (int i = 0; i < num_used_model; ++i) {
    if (i > 0) {
      str_buf << " , ";
    }
    str_buf << "PredictTree" << i << "ByMap";
  }
  str_buf << " };" << '\n' << '\n';

  std::stringstream pred_str_buf_map;
  Common::C_stringstream(pred_str_buf_map);

  pred_str_buf_map << "\t" << "int early_stop_round_counter = 0;" << '\n';
  pred_str_buf_map << "\t" << "std::memset(output, 0, sizeof(double) * num_tree_per_iteration_);" << '\n';
  pred_str_buf_map << "\t" << "for (int i = 0; i < num_iteration_for_pred_; ++i) {" << '\n';
  pred_str_buf_map << "\t\t" << "for (int k = 0; k < num_tree_per_iteration_; ++k) {" << '\n';
  pred_str_buf_map << "\t\t\t" << "output[k] += (*PredictTreeByMapPtr[i * num_tree_per_iteration_ + k])(features);" << '\n';
  pred_str_buf_map << "\t\t" << "}" << '\n';
  pred_str_buf_map << "\t\t" << "++early_stop_round_counter;" << '\n';
  pred_str_buf_map << "\t\t" << "if (early_stop->round_period == early_stop_round_counter) {" << '\n';
  pred_str_buf_map << "\t\t\t" << "if (early_stop->callback_function(output, num_tree_per_iteration_))" << '\n';
  pred_str_buf_map << "\t\t\t\t" << "return;" << '\n';
  pred_str_buf_map << "\t\t\t" << "early_stop_round_counter = 0;" << '\n';
  pred_str_buf_map << "\t\t" << "}" << '\n';
  pred_str_buf_map << "\t" << "}" << '\n';

  str_buf << "void GBDT::PredictRawByMap(const std::unordered_map<int, double>& features, double* output, const PredictionEarlyStopInstance* early_stop) const {" << '\n';
  str_buf << pred_str_buf_map.str();
  str_buf << "}" << '\n';
  str_buf << '\n';

  // Predict
  str_buf << "void GBDT::Predict(const double* features, double *output, const PredictionEarlyStopInstance* early_stop) const {" << '\n';
  str_buf << "\t" << "PredictRaw(features, output, early_stop);" << '\n';
  str_buf << "\t" << "if (average_output_) {" << '\n';
  str_buf << "\t\t" << "for (int k = 0; k < num_tree_per_iteration_; ++k) {" << '\n';
  str_buf << "\t\t\t" << "output[k] /= num_iteration_for_pred_;" << '\n';
  str_buf << "\t\t" << "}" << '\n';
  str_buf << "\t" << "}" << '\n';
  str_buf << "\t" << "if (objective_function_ != nullptr) {" << '\n';
  str_buf << "\t\t" << "objective_function_->ConvertOutput(output, output);" << '\n';
  str_buf << "\t" << "}" << '\n';
  str_buf << "}" << '\n';
  str_buf << '\n';

  // PredictByMap
  str_buf << "void GBDT::PredictByMap(const std::unordered_map<int, double>& features, double* output, const PredictionEarlyStopInstance* early_stop) const {" << '\n';
  str_buf << "\t" << "PredictRawByMap(features, output, early_stop);" << '\n';
  str_buf << "\t" << "if (average_output_) {" << '\n';
  str_buf << "\t\t" << "for (int k = 0; k < num_tree_per_iteration_; ++k) {" << '\n';
  str_buf << "\t\t\t" << "output[k] /= num_iteration_for_pred_;" << '\n';
  str_buf << "\t\t" << "}" << '\n';
  str_buf << "\t" << "}" << '\n';
  str_buf << "\t" << "if (objective_function_ != nullptr) {" << '\n';
  str_buf << "\t\t" << "objective_function_->ConvertOutput(output, output);" << '\n';
  str_buf << "\t" << "}" << '\n';
  str_buf << "}" << '\n';
  str_buf << '\n';


  // PredictLeafIndex
  for (int i = 0; i < num_used_model; ++i) {
    str_buf << models_[i]->ToIfElse(i, true) << '\n';
  }

  str_buf << "double (*PredictTreeLeafPtr[])(const double*) = { ";
  for (int i = 0; i < num_used_model; ++i) {
    if (i > 0) {
      str_buf << " , ";
    }
    str_buf << "PredictTree" << i << "Leaf";
  }
  str_buf << " };" << '\n' << '\n';

  str_buf << "void GBDT::PredictLeafIndex(const double* features, double *output) const {" << '\n';
  str_buf << "\t" << "int total_tree = num_iteration_for_pred_ * num_tree_per_iteration_;" << '\n';
  str_buf << "\t" << "for (int i = 0; i < total_tree; ++i) {" << '\n';
  str_buf << "\t\t" << "output[i] = (*PredictTreeLeafPtr[i])(features);" << '\n';
  str_buf << "\t" << "}" << '\n';
  str_buf << "}" << '\n';

  // PredictLeafIndexByMap
  str_buf << "double (*PredictTreeLeafByMapPtr[])(const std::unordered_map<int, double>&) = { ";
  for (int i = 0; i < num_used_model; ++i) {
    if (i > 0) {
      str_buf << " , ";
    }
    str_buf << "PredictTree" << i << "LeafByMap";
  }
  str_buf << " };" << '\n' << '\n';

  str_buf << "void GBDT::PredictLeafIndexByMap(const std::unordered_map<int, double>& features, double* output) const {" << '\n';
  str_buf << "\t" << "int total_tree = num_iteration_for_pred_ * num_tree_per_iteration_;" << '\n';
  str_buf << "\t" << "for (int i = 0; i < total_tree; ++i) {" << '\n';
  str_buf << "\t\t" << "output[i] = (*PredictTreeLeafByMapPtr[i])(features);" << '\n';
  str_buf << "\t" << "}" << '\n';
  str_buf << "}" << '\n';

  str_buf << "}  // namespace LightGBM" << '\n';

  return str_buf.str();
}

bool GBDT::SaveModelToIfElse(int num_iteration, const char* filename) const {
  /*! \brief File to write models */
  std::ofstream output_file;
  std::ifstream ifs(filename);
  if (ifs.good()) {
    std::string origin((std::istreambuf_iterator<char>(ifs)),
      (std::istreambuf_iterator<char>()));
    output_file.open(filename);
    output_file << "#define USE_HARD_CODE 0" << '\n';
    output_file << "#ifndef USE_HARD_CODE" << '\n';
    output_file << origin << '\n';
    output_file << "#else" << '\n';
    output_file << ModelToIfElse(num_iteration);
    output_file << "#endif" << '\n';
  } else {
    output_file.open(filename);
    output_file << ModelToIfElse(num_iteration);
  }

  ifs.close();
  output_file.close();

  return static_cast<bool>(output_file);
}

std::string GBDT::SaveModelToString(int start_iteration, int num_iteration, int feature_importance_type) const {
  std::stringstream ss;
  Common::C_stringstream(ss);

  // output model type
  ss << SubModelName() << '\n';
  ss << "version=" << kModelVersion << '\n';
  // output number of class
  ss << "num_class=" << num_class_ << '\n';
  ss << "num_tree_per_iteration=" << num_tree_per_iteration_ << '\n';
  // output label index
  ss << "label_index=" << label_idx_ << '\n';
  // output max_feature_idx
  ss << "max_feature_idx=" << max_feature_idx_ << '\n';
  // output objective
  if (objective_function_ != nullptr) {
    ss << "objective=" << objective_function_->ToString() << '\n';
  }

  if (average_output_) {
    ss << "average_output" << '\n';
  }

  ss << "feature_names=" << CommonC::Join(feature_names_, " ") << '\n';

  if (monotone_constraints_.size() != 0) {
    ss << "monotone_constraints=" << CommonC::Join(monotone_constraints_, " ")
       << '\n';
  }

  ss << "feature_infos=" << CommonC::Join(feature_infos_, " ") << '\n';
  // dump target encoding information
  ss << "\n";
  if (category_encoding_provider_ != nullptr) {
    ss << "has_category_encoding_provider=1\n" << category_encoding_provider_->DumpToString();
  } else {
    ss << "has_category_encoding_provider=0\n";
  }
  ss << "\n";

  int num_used_model = static_cast<int>(models_.size());
  int total_iteration = num_used_model / num_tree_per_iteration_;
  start_iteration = std::max(start_iteration, 0);
  start_iteration = std::min(start_iteration, total_iteration);
  if (num_iteration > 0) {
    int end_iteration = start_iteration + num_iteration;
    num_used_model = std::min(end_iteration * num_tree_per_iteration_, num_used_model);
  }

  int start_model = start_iteration * num_tree_per_iteration_;

  std::vector<std::string> tree_strs(num_used_model - start_model);
  std::vector<size_t> tree_sizes(num_used_model - start_model);
  // output tree models
  #pragma omp parallel for schedule(static)
  for (int i = start_model; i < num_used_model; ++i) {
    const int idx = i - start_model;
    tree_strs[idx] = "Tree=" + std::to_string(idx) + '\n';
    tree_strs[idx] += models_[i]->ToString() + '\n';
    tree_sizes[idx] = tree_strs[idx].size();
  }

  ss << "tree_sizes=" << CommonC::Join(tree_sizes, " ") << '\n';
  ss << '\n';

  for (int i = 0; i < num_used_model - start_model; ++i) {
    ss << tree_strs[i];
    tree_strs[i].clear();
  }
  ss << "end of trees" << "\n";
  std::vector<double> feature_importances = FeatureImportance(
      num_iteration, feature_importance_type);
  // store the importance first
  std::vector<std::pair<size_t, std::string>> pairs;
  for (size_t i = 0; i < feature_importances.size(); ++i) {
    size_t feature_importances_int = static_cast<size_t>(feature_importances[i]);
    if (feature_importances_int > 0) {
      pairs.emplace_back(feature_importances_int, feature_names_[i]);
    }
  }
  // sort the importance
  std::stable_sort(pairs.begin(), pairs.end(),
                   [](const std::pair<size_t, std::string>& lhs,
                      const std::pair<size_t, std::string>& rhs) {
    return lhs.first > rhs.first;
  });
  ss << '\n' << "feature_importances:" << '\n';
  for (size_t i = 0; i < pairs.size(); ++i) {
    ss << pairs[i].second << "=" << std::to_string(pairs[i].first) << '\n';
  }
  if (config_ != nullptr) {
    ss << "\nparameters:" << '\n';
    ss << config_->ToString() << "\n";
    ss << "end of parameters" << '\n';
  } else if (!loaded_parameter_.empty()) {
    ss << "\nparameters:" << '\n';
    ss << loaded_parameter_ << "\n";
    ss << "end of parameters" << '\n';
  }
  return ss.str();
}

bool GBDT::SaveModelToFile(int start_iteration, int num_iteration, int feature_importance_type, const char* filename) const {
  /*! \brief File to write models */
  auto writer = VirtualFileWriter::Make(filename);
  if (!writer->Init()) {
    Log::Fatal("Model file %s is not available for writes", filename);
  }
  std::string str_to_write = SaveModelToString(start_iteration, num_iteration, feature_importance_type);
  auto size = writer->Write(str_to_write.c_str(), str_to_write.size());
  return size > 0;
}

bool GBDT::LoadModelFromString(const char* buffer, size_t len) {
  // use serialized string to restore this object
  models_.clear();
  auto c_str = buffer;
  auto p = c_str;
  auto end = p + len;
  std::unordered_map<std::string, std::string> key_vals;
  int has_category_encoding_provider = 0;
  while (p < end) {
    size_t used_len = 0;
    auto line_len = Common::GetLine(p);
    if (line_len > 0) {
      std::string cur_line(p, line_len);
      if (Common::StartsWith(cur_line, "has_category_encoding_provider")) {
        Common::Atoi(Common::Split(cur_line.c_str(), "=")[1].c_str(), &has_category_encoding_provider);
        if (has_category_encoding_provider == 0) {
          category_encoding_provider_.reset(nullptr);
          used_len = line_len;
        } else {
          p += line_len;
          p = Common::SkipNewLine(p);
          category_encoding_provider_.reset(CategoryEncodingProvider::RecoverFromCharPointer(p, &used_len));
        }
      } else if (!Common::StartsWith(cur_line, "Tree=")) {
        auto strs = Common::Split(cur_line.c_str(), '=');
        if (strs.size() == 1) {
          key_vals[strs[0]] = "";
        } else if (strs.size() == 2) {
          key_vals[strs[0]] = strs[1];
        } else if (strs.size() > 2) {
          if (strs[0] == "feature_names") {
            key_vals[strs[0]] = cur_line.substr(std::strlen("feature_names="));
          } else if (strs[0] == "monotone_constraints") {
            key_vals[strs[0]] = cur_line.substr(std::strlen("monotone_constraints="));
          } else {
            // Use first 128 chars to avoid exceed the message buffer.
            Log::Fatal("Wrong line at model file: %s", cur_line.substr(0, std::min<size_t>(128, cur_line.size())).c_str());
          }
        }
        used_len = line_len;
      } else {
        break;
      }
    }
    p += used_len;
    p = Common::SkipNewLine(p);
  }

  // for backward compatibility, since old version model files has no `has_category_encoding_provider=` entry.
  if (has_category_encoding_provider == 0) {
    category_encoding_provider_.reset(nullptr);
  }

  // get number of classes
  if (key_vals.count("num_class")) {
    Common::Atoi(key_vals["num_class"].c_str(), &num_class_);
  } else {
    Log::Fatal("Model file doesn't specify the number of classes");
    return false;
  }

  if (key_vals.count("num_tree_per_iteration")) {
    Common::Atoi(key_vals["num_tree_per_iteration"].c_str(), &num_tree_per_iteration_);
  } else {
    num_tree_per_iteration_ = num_class_;
  }

  // get index of label
  if (key_vals.count("label_index")) {
    Common::Atoi(key_vals["label_index"].c_str(), &label_idx_);
  } else {
    Log::Fatal("Model file doesn't specify the label index");
    return false;
  }

  // get max_feature_idx first
  if (key_vals.count("max_feature_idx")) {
    Common::Atoi(key_vals["max_feature_idx"].c_str(), &max_feature_idx_);
  } else {
    Log::Fatal("Model file doesn't specify max_feature_idx");
    return false;
  }

  // get average_output
  if (key_vals.count("average_output")) {
    average_output_ = true;
  }

  // get feature names
  if (key_vals.count("feature_names")) {
    feature_names_ = Common::Split(key_vals["feature_names"].c_str(), ' ');
    if (feature_names_.size() != static_cast<size_t>(max_feature_idx_ + 1)) {
      Log::Fatal("Wrong size of feature_names");
      return false;
    }
  } else {
    Log::Fatal("Model file doesn't contain feature_names");
    return false;
  }

  // get monotone_constraints
  if (key_vals.count("monotone_constraints")) {
    monotone_constraints_ = CommonC::StringToArray<int8_t>(key_vals["monotone_constraints"].c_str(), ' ');
    if (monotone_constraints_.size() != static_cast<size_t>(max_feature_idx_ + 1)) {
      Log::Fatal("Wrong size of monotone_constraints");
      return false;
    }
  }

  if (key_vals.count("feature_infos")) {
    feature_infos_ = Common::Split(key_vals["feature_infos"].c_str(), ' ');
    if (feature_infos_.size() != static_cast<size_t>(max_feature_idx_ + 1)) {
      Log::Fatal("Wrong size of feature_infos");
      return false;
    }
  } else {
    Log::Fatal("Model file doesn't contain feature_infos");
    return false;
  }

  if (key_vals.count("objective")) {
    auto str = key_vals["objective"];
    loaded_objective_.reset(ObjectiveFunction::CreateObjectiveFunction(ParseObjectiveAlias(str)));
    objective_function_ = loaded_objective_.get();
  }

  if (!key_vals.count("tree_sizes")) {
    while (p < end) {
      auto line_len = Common::GetLine(p);
      if (line_len > 0) {
        std::string cur_line(p, line_len);
        if (Common::StartsWith(cur_line, "Tree=")) {
          p += line_len;
          p = Common::SkipNewLine(p);
          size_t used_len = 0;
          models_.emplace_back(new Tree(p, &used_len));
          p += used_len;
        } else {
          break;
        }
      }
      p = Common::SkipNewLine(p);
    }
  } else {
    std::vector<size_t> tree_sizes = CommonC::StringToArray<size_t>(key_vals["tree_sizes"].c_str(), ' ');
    std::vector<size_t> tree_boundries(tree_sizes.size() + 1, 0);
    int num_trees = static_cast<int>(tree_sizes.size());
    for (int i = 0; i < num_trees; ++i) {
      tree_boundries[i + 1] = tree_boundries[i] + tree_sizes[i];
      models_.emplace_back(nullptr);
    }
    OMP_INIT_EX();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_trees; ++i) {
      OMP_LOOP_EX_BEGIN();
      auto cur_p = p + tree_boundries[i];
      auto line_len = Common::GetLine(cur_p);
      std::string cur_line(cur_p, line_len);
      if (Common::StartsWith(cur_line, "Tree=")) {
        cur_p += line_len;
        cur_p = Common::SkipNewLine(cur_p);
        size_t used_len = 0;
        models_[i].reset(new Tree(cur_p, &used_len));
      } else {
        Log::Fatal("Model format error, expect a tree here. met %s", cur_line.c_str());
      }
      OMP_LOOP_EX_END();
    }
    OMP_THROW_EX();
  }
  num_iteration_for_pred_ = static_cast<int>(models_.size()) / num_tree_per_iteration_;
  num_init_iteration_ = num_iteration_for_pred_;
  iter_ = 0;
  bool is_inparameter = false;
  std::stringstream ss;
  Common::C_stringstream(ss);
  while (p < end) {
    auto line_len = Common::GetLine(p);
    if (line_len > 0) {
      std::string cur_line(p, line_len);
      if (cur_line == std::string("parameters:")) {
        is_inparameter = true;
      } else if (cur_line == std::string("end of parameters")) {
        break;
      } else if (is_inparameter) {
        ss << cur_line << "\n";
        if (Common::StartsWith(cur_line, "[linear_tree: ")) {
          int is_linear = 0;
          Common::Atoi(cur_line.substr(14, 1).c_str(), &is_linear);
          linear_tree_ = static_cast<bool>(is_linear);
        }
      }
    }
    p += line_len;
    p = Common::SkipNewLine(p);
  }
  if (!ss.str().empty()) {
    loaded_parameter_ = ss.str();
  }
  return true;
}

std::vector<double> GBDT::FeatureImportance(int num_iteration, int importance_type) const {
  int num_used_model = static_cast<int>(models_.size());
  if (num_iteration > 0) {
    num_iteration += 0;
    num_used_model = std::min(num_iteration * num_tree_per_iteration_, num_used_model);
  }

  std::vector<double> feature_importances(max_feature_idx_ + 1, 0.0);
  if (importance_type == 0) {
    for (int iter = 0; iter < num_used_model; ++iter) {
      for (int split_idx = 0; split_idx < models_[iter]->num_leaves() - 1; ++split_idx) {
        if (models_[iter]->split_gain(split_idx) > 0) {
#ifdef DEBUG
          CHECK_GE(models_[iter]->split_feature(split_idx), 0);
#endif
          feature_importances[models_[iter]->split_feature(split_idx)] += 1.0;
        }
      }
    }
  } else if (importance_type == 1) {
    for (int iter = 0; iter < num_used_model; ++iter) {
      for (int split_idx = 0; split_idx < models_[iter]->num_leaves() - 1; ++split_idx) {
        if (models_[iter]->split_gain(split_idx) > 0) {
#ifdef DEBUG
          CHECK_GE(models_[iter]->split_feature(split_idx), 0);
#endif
          feature_importances[models_[iter]->split_feature(split_idx)] += models_[iter]->split_gain(split_idx);
        }
      }
    }
  } else {
    Log::Fatal("Unknown importance type: only support split=0 and gain=1");
  }
  return feature_importances;
}

}  // namespace LightGBM
