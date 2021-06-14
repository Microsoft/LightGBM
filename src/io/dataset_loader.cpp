/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <LightGBM/dataset_loader.h>

#include <LightGBM/network.h>
#include <LightGBM/utils/array_args.h>
#include <LightGBM/utils/common.h>
#include <LightGBM/utils/json11.h>
#include <LightGBM/utils/log.h>
#include <LightGBM/utils/openmp_wrapper.h>

#include <chrono>
#include <fstream>

namespace LightGBM {

using json11::Json;

DatasetLoader::DatasetLoader(Config* io_config, const PredictFunction& predict_fun, int num_class, const char* filename)
  :config_(*io_config), random_(config_.data_random_seed), predict_fun_(predict_fun), num_class_(num_class) {
  label_idx_ = 0;
  weight_idx_ = NO_SPECIFIC;
  group_idx_ = NO_SPECIFIC;
  SetHeader(filename);
  store_raw_ = false;
  if (io_config->linear_tree) {
    store_raw_ = true;
  }
}

DatasetLoader::~DatasetLoader() {
}

void DatasetLoader::SetHeader(const char* filename) {
  std::unordered_map<std::string, int> name2idx;
  std::string name_prefix("name:");
  if (filename != nullptr) {
    TextReader<data_size_t> text_reader(filename, config_.header);

    // get column names
    if (config_.header) {
      std::string first_line = text_reader.first_line();
      feature_names_ = Common::Split(first_line.c_str(), "\t,");
    }

    // load label idx first
    if (config_.label_column.size() > 0) {
      if (Common::StartsWith(config_.label_column, name_prefix)) {
        std::string name = config_.label_column.substr(name_prefix.size());
        label_idx_ = -1;
        for (int i = 0; i < static_cast<int>(feature_names_.size()); ++i) {
          if (name == feature_names_[i]) {
            label_idx_ = i;
            break;
          }
        }
        if (label_idx_ >= 0) {
          Log::Info("Using column %s as label", name.c_str());
        } else {
          Log::Fatal("Could not find label column %s in data file \n"
                     "or data file doesn't contain header", name.c_str());
        }
      } else {
        if (!Common::AtoiAndCheck(config_.label_column.c_str(), &label_idx_)) {
          Log::Fatal("label_column is not a number,\n"
                     "if you want to use a column name,\n"
                     "please add the prefix \"name:\" to the column name");
        }
        Log::Info("Using column number %d as label", label_idx_);
      }
    }

    if (!feature_names_.empty()) {
      // erase label column name
      feature_names_.erase(feature_names_.begin() + label_idx_);
      for (size_t i = 0; i < feature_names_.size(); ++i) {
        name2idx[feature_names_[i]] = static_cast<int>(i);
      }
    }

    // load ignore columns
    if (config_.ignore_column.size() > 0) {
      if (Common::StartsWith(config_.ignore_column, name_prefix)) {
        std::string names = config_.ignore_column.substr(name_prefix.size());
        for (auto name : Common::Split(names.c_str(), ',')) {
          if (name2idx.count(name) > 0) {
            int tmp = name2idx[name];
            ignore_features_.emplace(tmp);
          } else {
            Log::Fatal("Could not find ignore column %s in data file", name.c_str());
          }
        }
      } else {
        for (auto token : Common::Split(config_.ignore_column.c_str(), ',')) {
          int tmp = 0;
          if (!Common::AtoiAndCheck(token.c_str(), &tmp)) {
            Log::Fatal("ignore_column is not a number,\n"
                       "if you want to use a column name,\n"
                       "please add the prefix \"name:\" to the column name");
          }
          ignore_features_.emplace(tmp);
        }
      }
    }
    // load weight idx
    if (config_.weight_column.size() > 0) {
      if (Common::StartsWith(config_.weight_column, name_prefix)) {
        std::string name = config_.weight_column.substr(name_prefix.size());
        if (name2idx.count(name) > 0) {
          weight_idx_ = name2idx[name];
          Log::Info("Using column %s as weight", name.c_str());
        } else {
          Log::Fatal("Could not find weight column %s in data file", name.c_str());
        }
      } else {
        if (!Common::AtoiAndCheck(config_.weight_column.c_str(), &weight_idx_)) {
          Log::Fatal("weight_column is not a number,\n"
                     "if you want to use a column name,\n"
                     "please add the prefix \"name:\" to the column name");
        }
        Log::Info("Using column number %d as weight", weight_idx_);
      }
      ignore_features_.emplace(weight_idx_);
    }
    // load group idx
    if (config_.group_column.size() > 0) {
      if (Common::StartsWith(config_.group_column, name_prefix)) {
        std::string name = config_.group_column.substr(name_prefix.size());
        if (name2idx.count(name) > 0) {
          group_idx_ = name2idx[name];
          Log::Info("Using column %s as group/query id", name.c_str());
        } else {
          Log::Fatal("Could not find group/query column %s in data file", name.c_str());
        }
      } else {
        if (!Common::AtoiAndCheck(config_.group_column.c_str(), &group_idx_)) {
          Log::Fatal("group_column is not a number,\n"
                     "if you want to use a column name,\n"
                     "please add the prefix \"name:\" to the column name");
        }
        Log::Info("Using column number %d as group/query id", group_idx_);
      }
      ignore_features_.emplace(group_idx_);
    }
  }
  if (config_.categorical_feature.size() > 0) {
    if (Common::StartsWith(config_.categorical_feature, name_prefix)) {
      std::string names = config_.categorical_feature.substr(name_prefix.size());
      for (auto name : Common::Split(names.c_str(), ',')) {
        if (name2idx.count(name) > 0) {
          int tmp = name2idx[name];
          categorical_features_.emplace(tmp);
        } else {
          Log::Fatal("Could not find categorical_feature %s in data file", name.c_str());
        }
      }
    } else {
      for (auto token : Common::Split(config_.categorical_feature.c_str(), ',')) {
        int tmp = 0;
        if (!Common::AtoiAndCheck(token.c_str(), &tmp)) {
          Log::Fatal("categorical_feature is not a number,\n"
                     "if you want to use a column name,\n"
                     "please add the prefix \"name:\" to the column name");
        }
        categorical_features_.emplace(tmp);
      }
    }
  }
}

void CheckSampleSize(size_t sample_cnt, size_t num_data) {
  if (static_cast<double>(sample_cnt) / num_data < 0.2f &&
      sample_cnt < 100000) {
    Log::Warning(
        "Using too small ``bin_construct_sample_cnt`` may encounter "
        "unexpected "
        "errors and poor accuracy.");
  }
}

Dataset* DatasetLoader::LoadFromFile(const char* filename, int rank, int num_machines,
  CategoryEncodingProvider* category_encoding_provider) {
  // don't support query id in data file when using distributed training
  if (num_machines > 1 && !config_.pre_partition) {
    if (group_idx_ > 0) {
      Log::Fatal("Using a query id without pre-partitioning the data file is not supported for distributed training.\n"
                 "Please use an additional query file or pre-partition the data");
    }
  }
  auto dataset = std::unique_ptr<Dataset>(new Dataset());
  if (store_raw_) {
    dataset->SetHasRaw(true);
  }
  data_size_t num_global_data = 0;
  std::vector<data_size_t> used_data_indices;

  auto bin_filename = Parser::CheckCanLoadFromBin(filename);
  bool is_load_from_binary = false;

  if (bin_filename.size() == 0) {
    auto parser = std::unique_ptr<Parser>(Parser::CreateParser(filename, config_.header, 0, label_idx_,
                                                               config_.precise_float_parser));
    if (parser == nullptr) {
      Log::Fatal("Could not recognize data format of %s", filename);
    }
    dataset->data_filename_ = filename;
    dataset->label_idx_ = label_idx_;
    dataset->metadata_.Init(filename);
    if (!config_.two_round) {
      // read data to memory
      if (category_encoding_provider != nullptr) {
        category_encoding_provider->InitFromParser(&config_, parser.release(), num_machines, &categorical_features_);
      }
      auto text_data = LoadTextDataToMemory(filename, dataset->metadata_, rank, num_machines,
        &num_global_data, &used_data_indices, category_encoding_provider);
      if (category_encoding_provider != nullptr) {
        Parser* parser_ptr = category_encoding_provider->FinishProcess(num_machines, &config_);
        if (category_encoding_provider->GetNumCatConverters() == 0) {
          category_encoding_provider = nullptr;
          parser.reset(parser_ptr);
        } else {
          parser.reset(new CategoryEncodingParser(parser_ptr, category_encoding_provider, false));
        }
      }
      dataset->num_data_ = static_cast<data_size_t>(text_data.size());
      // sample data
      std::vector<std::string> sample_data;
      std::vector<data_size_t> sampled_indices;
      if (category_encoding_provider == nullptr) {
        sample_data = SampleTextDataFromMemory<false>(text_data, &sampled_indices);
      } else {
        sample_data = SampleTextDataFromMemory<true>(text_data, &sampled_indices);
        if (category_encoding_provider != nullptr) {
          category_encoding_provider->ExtendFeatureNames(&feature_names_);
        }
      }
      CheckSampleSize(sample_data.size(),
                      static_cast<size_t>(dataset->num_data_));
      // construct feature bin mappers
      ConstructBinMappersFromTextData(rank, num_machines, sample_data, parser.get(), dataset.get(), sampled_indices);
      if (dataset->has_raw()) {
        dataset->ResizeRaw(dataset->num_data_);
      }
      // initialize label
      dataset->metadata_.Init(dataset->num_data_, weight_idx_, group_idx_);
      // extract features
      ExtractFeaturesFromMemory(&text_data, parser.get(), dataset.get());
      text_data.clear();
    } else {
      // sample data from file
      std::vector<std::string> sample_data;
      std::vector<data_size_t> sampled_indices;
      if (category_encoding_provider != nullptr) {
        category_encoding_provider->InitFromParser(&config_, parser.release(), num_machines, &categorical_features_);
      }
      sample_data = SampleTextDataFromFile(filename, dataset->metadata_, rank, num_machines,
        &num_global_data, &used_data_indices, &sampled_indices, category_encoding_provider);
      if (category_encoding_provider != nullptr) {
        Parser* parser_ptr = category_encoding_provider->FinishProcess(num_machines, &config_);
        if (category_encoding_provider->GetNumCatConverters() == 0) {
          category_encoding_provider = nullptr;
          parser.reset(parser_ptr);
        } else {
          parser.reset(new CategoryEncodingParser(parser_ptr, category_encoding_provider, false));
        }
      }
      if (used_data_indices.size() > 0) {
        dataset->num_data_ = static_cast<data_size_t>(used_data_indices.size());
      } else {
        dataset->num_data_ = num_global_data;
      }
      CheckSampleSize(sample_data.size(),
                      static_cast<size_t>(dataset->num_data_));
      // construct feature bin mappers
      ConstructBinMappersFromTextData(rank, num_machines, sample_data, parser.get(), dataset.get(), sampled_indices);
      if (dataset->has_raw()) {
        dataset->ResizeRaw(dataset->num_data_);
      }
      // initialize label
      dataset->metadata_.Init(dataset->num_data_, weight_idx_, group_idx_);
      Log::Info("Making second pass...");
      // extract features
      ExtractFeaturesFromFile(filename, parser.get(), used_data_indices, dataset.get());
    }
    if (category_encoding_provider != nullptr) {
      dataset->SetCategoryEncodingProvider(CategoryEncodingProvider::RecoverFromModelString(category_encoding_provider->DumpToString()));
    }
  } else {
    // load data from binary file
    is_load_from_binary = true;
    Log::Info("Load from binary file %s", bin_filename.c_str());
    dataset.reset(LoadFromBinFile(filename, bin_filename.c_str(), rank, num_machines, &num_global_data, &used_data_indices));
  }
  // check meta data
  dataset->metadata_.CheckOrPartition(num_global_data, used_data_indices);
  // need to check training data

  CheckDataset(dataset.get(), is_load_from_binary);


  return dataset.release();
}



Dataset* DatasetLoader::LoadFromFileAlignWithOtherDataset(const char* filename, const Dataset* train_data) {
  data_size_t num_global_data = 0;
  std::vector<data_size_t> used_data_indices;
  auto dataset = std::unique_ptr<Dataset>(new Dataset());
  auto bin_filename = Parser::CheckCanLoadFromBin(filename);
  const CategoryEncodingProvider* category_encoding_provider = train_data->category_encoding_provider();
  if (store_raw_) {
    dataset->SetHasRaw(true);
  }
  if (bin_filename.size() == 0) {
    auto parser = std::unique_ptr<Parser>(Parser::CreateParser(filename, config_.header, 0, label_idx_,
                                                               config_.precise_float_parser));
    if (parser == nullptr) {
      Log::Fatal("Could not recognize data format of %s", filename);
    }
    if (category_encoding_provider != nullptr) {
      std::unique_ptr<Parser> inner_parser(nullptr);
      inner_parser.reset(parser.release());
      parser.reset(new CategoryEncodingParser(inner_parser.release(), category_encoding_provider, true));
    }
    dataset->data_filename_ = filename;
    dataset->label_idx_ = label_idx_;
    dataset->metadata_.Init(filename);
    if (!config_.two_round) {
      // read data in memory
      auto text_data = LoadTextDataToMemory(filename, dataset->metadata_, 0, 1, &num_global_data, &used_data_indices, nullptr);
      dataset->num_data_ = static_cast<data_size_t>(text_data.size());
      // initialize label
      dataset->metadata_.Init(dataset->num_data_, weight_idx_, group_idx_);
      dataset->CreateValid(train_data);
      if (dataset->has_raw()) {
        dataset->ResizeRaw(dataset->num_data_);
      }
      // extract features
      ExtractFeaturesFromMemory(&text_data, parser.get(), dataset.get());
      text_data.clear();
    } else {
      TextReader<data_size_t> text_reader(filename, config_.header);
      // Get number of lines of data file
      dataset->num_data_ = static_cast<data_size_t>(text_reader.CountLine());
      num_global_data = dataset->num_data_;
      // initialize label
      dataset->metadata_.Init(dataset->num_data_, weight_idx_, group_idx_);
      dataset->CreateValid(train_data);
      if (dataset->has_raw()) {
        dataset->ResizeRaw(dataset->num_data_);
      }
      // extract features
      ExtractFeaturesFromFile(filename, parser.get(), used_data_indices, dataset.get());
    }
  } else {
    // load data from binary file
    dataset.reset(LoadFromBinFile(filename, bin_filename.c_str(), 0, 1, &num_global_data, &used_data_indices));
  }
  // not need to check validation data
  // check meta data
  dataset->metadata_.CheckOrPartition(num_global_data, used_data_indices);
  return dataset.release();
}

Dataset* DatasetLoader::LoadFromBinFile(const char* data_filename, const char* bin_filename,
                                        int rank, int num_machines, int* num_global_data,
                                        std::vector<data_size_t>* used_data_indices) {
  auto dataset = std::unique_ptr<Dataset>(new Dataset());
  auto reader = VirtualFileReader::Make(bin_filename);
  dataset->data_filename_ = data_filename;
  if (!reader->Init()) {
    Log::Fatal("Could not read binary data from %s", bin_filename);
  }

  // buffer to read binary file
  size_t buffer_size = 16 * 1024 * 1024;
  auto buffer = std::vector<char>(buffer_size);

  // check token
  size_t size_of_token = std::strlen(Parser::binary_file_token);
  size_t read_cnt = reader->Read(
      buffer.data(),
      VirtualFileWriter::AlignedSize(sizeof(char) * size_of_token));
  if (read_cnt < sizeof(char) * size_of_token) {
    Log::Fatal("Binary file error: token has the wrong size");
  }
  if (std::string(buffer.data()) != std::string(Parser::binary_file_token)) {
    Log::Fatal("Input file is not LightGBM binary file");
  }

  // read size of header
  read_cnt = reader->Read(buffer.data(), sizeof(size_t));

  if (read_cnt != sizeof(size_t)) {
    Log::Fatal("Binary file error: header has the wrong size");
  }

  size_t size_of_head = *(reinterpret_cast<size_t*>(buffer.data()));

  // re-allocmate space if not enough
  if (size_of_head > buffer_size) {
    buffer_size = size_of_head;
    buffer.resize(buffer_size);
  }
  // read header
  read_cnt = reader->Read(buffer.data(), size_of_head);

  if (read_cnt != size_of_head) {
    Log::Fatal("Binary file error: header is incorrect");
  }
  // get header
  const char* mem_ptr = buffer.data();
  dataset->num_data_ = *(reinterpret_cast<const data_size_t*>(mem_ptr));
  mem_ptr += VirtualFileWriter::AlignedSize(sizeof(dataset->num_data_));
  dataset->num_features_ = *(reinterpret_cast<const int*>(mem_ptr));
  mem_ptr += VirtualFileWriter::AlignedSize(sizeof(dataset->num_features_));
  dataset->num_total_features_ = *(reinterpret_cast<const int*>(mem_ptr));
  mem_ptr +=
      VirtualFileWriter::AlignedSize(sizeof(dataset->num_total_features_));
  dataset->label_idx_ = *(reinterpret_cast<const int*>(mem_ptr));
  mem_ptr += VirtualFileWriter::AlignedSize(sizeof(dataset->label_idx_));
  dataset->max_bin_ = *(reinterpret_cast<const int*>(mem_ptr));
  mem_ptr += VirtualFileWriter::AlignedSize(sizeof(dataset->max_bin_));
  dataset->bin_construct_sample_cnt_ = *(reinterpret_cast<const int*>(mem_ptr));
  mem_ptr += VirtualFileWriter::AlignedSize(
      sizeof(dataset->bin_construct_sample_cnt_));
  dataset->min_data_in_bin_ = *(reinterpret_cast<const int*>(mem_ptr));
  mem_ptr += VirtualFileWriter::AlignedSize(sizeof(dataset->min_data_in_bin_));
  dataset->use_missing_ = *(reinterpret_cast<const bool*>(mem_ptr));
  mem_ptr += VirtualFileWriter::AlignedSize(sizeof(dataset->use_missing_));
  dataset->zero_as_missing_ = *(reinterpret_cast<const bool*>(mem_ptr));
  mem_ptr += VirtualFileWriter::AlignedSize(sizeof(dataset->zero_as_missing_));
  dataset->has_raw_ = *(reinterpret_cast<const bool*>(mem_ptr));
  mem_ptr += VirtualFileWriter::AlignedSize(sizeof(dataset->has_raw_));
  const int* tmp_feature_map = reinterpret_cast<const int*>(mem_ptr);
  dataset->used_feature_map_.clear();
  for (int i = 0; i < dataset->num_total_features_; ++i) {
    dataset->used_feature_map_.push_back(tmp_feature_map[i]);
  }
  mem_ptr += VirtualFileWriter::AlignedSize(sizeof(int) *
                                            dataset->num_total_features_);
  // num_groups
  dataset->num_groups_ = *(reinterpret_cast<const int*>(mem_ptr));
  mem_ptr += VirtualFileWriter::AlignedSize(sizeof(dataset->num_groups_));
  // real_feature_idx_
  const int* tmp_ptr_real_feature_idx_ = reinterpret_cast<const int*>(mem_ptr);
  dataset->real_feature_idx_.clear();
  for (int i = 0; i < dataset->num_features_; ++i) {
    dataset->real_feature_idx_.push_back(tmp_ptr_real_feature_idx_[i]);
  }
  mem_ptr +=
      VirtualFileWriter::AlignedSize(sizeof(int) * dataset->num_features_);
  // feature2group
  const int* tmp_ptr_feature2group = reinterpret_cast<const int*>(mem_ptr);
  dataset->feature2group_.clear();
  for (int i = 0; i < dataset->num_features_; ++i) {
    dataset->feature2group_.push_back(tmp_ptr_feature2group[i]);
  }
  mem_ptr +=
      VirtualFileWriter::AlignedSize(sizeof(int) * dataset->num_features_);
  // feature2subfeature
  const int* tmp_ptr_feature2subfeature = reinterpret_cast<const int*>(mem_ptr);
  dataset->feature2subfeature_.clear();
  for (int i = 0; i < dataset->num_features_; ++i) {
    dataset->feature2subfeature_.push_back(tmp_ptr_feature2subfeature[i]);
  }
  mem_ptr +=
      VirtualFileWriter::AlignedSize(sizeof(int) * dataset->num_features_);
  // group_bin_boundaries
  const uint64_t* tmp_ptr_group_bin_boundaries = reinterpret_cast<const uint64_t*>(mem_ptr);
  dataset->group_bin_boundaries_.clear();
  for (int i = 0; i < dataset->num_groups_ + 1; ++i) {
    dataset->group_bin_boundaries_.push_back(tmp_ptr_group_bin_boundaries[i]);
  }
  mem_ptr += sizeof(uint64_t) * (dataset->num_groups_ + 1);

  // group_feature_start_
  const int* tmp_ptr_group_feature_start = reinterpret_cast<const int*>(mem_ptr);
  dataset->group_feature_start_.clear();
  for (int i = 0; i < dataset->num_groups_; ++i) {
    dataset->group_feature_start_.push_back(tmp_ptr_group_feature_start[i]);
  }
  mem_ptr +=
      VirtualFileWriter::AlignedSize(sizeof(int) * (dataset->num_groups_));

  // group_feature_cnt_
  const int* tmp_ptr_group_feature_cnt = reinterpret_cast<const int*>(mem_ptr);
  dataset->group_feature_cnt_.clear();
  for (int i = 0; i < dataset->num_groups_; ++i) {
    dataset->group_feature_cnt_.push_back(tmp_ptr_group_feature_cnt[i]);
  }
  mem_ptr +=
      VirtualFileWriter::AlignedSize(sizeof(int) * (dataset->num_groups_));

  if (!config_.max_bin_by_feature.empty()) {
    CHECK_EQ(static_cast<size_t>(dataset->num_total_features_), config_.max_bin_by_feature.size());
    CHECK_GT(*(std::min_element(config_.max_bin_by_feature.begin(), config_.max_bin_by_feature.end())), 1);
    dataset->max_bin_by_feature_.resize(dataset->num_total_features_);
    dataset->max_bin_by_feature_.assign(config_.max_bin_by_feature.begin(), config_.max_bin_by_feature.end());
  } else {
    const int32_t* tmp_ptr_max_bin_by_feature = reinterpret_cast<const int32_t*>(mem_ptr);
    dataset->max_bin_by_feature_.clear();
    for (int i = 0; i < dataset->num_total_features_; ++i) {
      dataset->max_bin_by_feature_.push_back(tmp_ptr_max_bin_by_feature[i]);
    }
  }
  mem_ptr += VirtualFileWriter::AlignedSize(sizeof(int32_t) *
                                            (dataset->num_total_features_));
  if (ArrayArgs<int32_t>::CheckAll(dataset->max_bin_by_feature_, -1)) {
    dataset->max_bin_by_feature_.clear();
  }

  // get feature names
  dataset->feature_names_.clear();
  // write feature names
  for (int i = 0; i < dataset->num_total_features_; ++i) {
    int str_len = *(reinterpret_cast<const int*>(mem_ptr));
    mem_ptr += VirtualFileWriter::AlignedSize(sizeof(int));
    std::stringstream str_buf;
    auto tmp_arr = reinterpret_cast<const char*>(mem_ptr);
    for (int j = 0; j < str_len; ++j) {
      char tmp_char = tmp_arr[j];
      str_buf << tmp_char;
    }
    mem_ptr += VirtualFileWriter::AlignedSize(sizeof(char) * str_len);
    dataset->feature_names_.emplace_back(str_buf.str());
  }
  // get forced_bin_bounds_
  dataset->forced_bin_bounds_ = std::vector<std::vector<double>>(dataset->num_total_features_, std::vector<double>());
  for (int i = 0; i < dataset->num_total_features_; ++i) {
    int num_bounds = *(reinterpret_cast<const int*>(mem_ptr));
    mem_ptr += VirtualFileWriter::AlignedSize(sizeof(int));
    dataset->forced_bin_bounds_[i] = std::vector<double>();
    const double* tmp_ptr_forced_bounds =
        reinterpret_cast<const double*>(mem_ptr);
    for (int j = 0; j < num_bounds; ++j) {
      double bound = tmp_ptr_forced_bounds[j];
      dataset->forced_bin_bounds_[i].push_back(bound);
    }
    mem_ptr += num_bounds * sizeof(double);
  }

  // read size of meta data
  read_cnt = reader->Read(buffer.data(), sizeof(size_t));

  if (read_cnt != sizeof(size_t)) {
    Log::Fatal("Binary file error: meta data has the wrong size");
  }

  size_t size_of_metadata = *(reinterpret_cast<size_t*>(buffer.data()));

  // re-allocate space if not enough
  if (size_of_metadata > buffer_size) {
    buffer_size = size_of_metadata;
    buffer.resize(buffer_size);
  }
  //  read meta data
  read_cnt = reader->Read(buffer.data(), size_of_metadata);

  if (read_cnt != size_of_metadata) {
    Log::Fatal("Binary file error: meta data is incorrect");
  }
  // load meta data
  dataset->metadata_.LoadFromMemory(buffer.data());

  *num_global_data = dataset->num_data_;
  used_data_indices->clear();
  // sample local used data if need to partition
  if (num_machines > 1 && !config_.pre_partition) {
    const data_size_t* query_boundaries = dataset->metadata_.query_boundaries();
    if (query_boundaries == nullptr) {
      // if not contain query file, minimal sample unit is one record
      for (data_size_t i = 0; i < dataset->num_data_; ++i) {
        if (random_.NextShort(0, num_machines) == rank) {
          used_data_indices->push_back(i);
        }
      }
    } else {
      // if contain query file, minimal sample unit is one query
      data_size_t num_queries = dataset->metadata_.num_queries();
      data_size_t qid = -1;
      bool is_query_used = false;
      for (data_size_t i = 0; i < dataset->num_data_; ++i) {
        if (qid >= num_queries) {
          Log::Fatal("Current query exceeds the range of the query file,\n"
                     "please ensure the query file is correct");
        }
        if (i >= query_boundaries[qid + 1]) {
          // if is new query
          is_query_used = false;
          if (random_.NextShort(0, num_machines) == rank) {
            is_query_used = true;
          }
          ++qid;
        }
        if (is_query_used) {
          used_data_indices->push_back(i);
        }
      }
    }
    dataset->num_data_ = static_cast<data_size_t>((*used_data_indices).size());
  }
  dataset->metadata_.PartitionLabel(*used_data_indices);
  // read feature data
  for (int i = 0; i < dataset->num_groups_; ++i) {
    // read feature size
    read_cnt = reader->Read(buffer.data(), sizeof(size_t));
    if (read_cnt != sizeof(size_t)) {
      Log::Fatal("Binary file error: feature %d has the wrong size", i);
    }
    size_t size_of_feature = *(reinterpret_cast<size_t*>(buffer.data()));
    // re-allocate space if not enough
    if (size_of_feature > buffer_size) {
      buffer_size = size_of_feature;
      buffer.resize(buffer_size);
    }

    read_cnt = reader->Read(buffer.data(), size_of_feature);

    if (read_cnt != size_of_feature) {
      Log::Fatal("Binary file error: feature %d is incorrect, read count: %d", i, read_cnt);
    }
    dataset->feature_groups_.emplace_back(std::unique_ptr<FeatureGroup>(
      new FeatureGroup(buffer.data(),
                       *num_global_data,
                       *used_data_indices, i)));
  }
  dataset->feature_groups_.shrink_to_fit();

  // read size of category encoding provider
  read_cnt = reader->Read(buffer.data(), sizeof(size_t));

  if (read_cnt != sizeof(size_t)) {
    Log::Fatal("Binary file error: category encoding provider data has the wrong size");
  }

  size_t size_of_category_encoding_provider = *(reinterpret_cast<size_t*>(buffer.data()));
  // re-allocate space if not enough
  if (size_of_category_encoding_provider > buffer_size) {
    buffer_size = size_of_category_encoding_provider;
    buffer.resize(buffer_size);
  }
  //  read category encoding provider
  read_cnt = reader->Read(buffer.data(), size_of_category_encoding_provider);

  if (read_cnt != size_of_category_encoding_provider) {
    Log::Fatal("Binary file error: category_encoding_provider is incorrect");
  }
  // load category encoding provider
  if (size_of_category_encoding_provider > 0) {
    dataset->category_encoding_provider_.reset(CategoryEncodingProvider::RecoverFromModelString(
      std::string(buffer.data(), size_of_category_encoding_provider / sizeof(char))));
  } else {
    dataset->category_encoding_provider_ = nullptr;
  }

  // raw data
  dataset->numeric_feature_map_ = std::vector<int>(dataset->num_features_, false);
  dataset->num_numeric_features_ = 0;
  for (int i = 0; i < dataset->num_features_; ++i) {
    if (dataset->FeatureBinMapper(i)->bin_type() == BinType::CategoricalBin) {
      dataset->numeric_feature_map_[i] = -1;
    } else {
      dataset->numeric_feature_map_[i] = dataset->num_numeric_features_;
      ++dataset->num_numeric_features_;
    }
  }
  if (dataset->has_raw()) {
    dataset->ResizeRaw(dataset->num_data());
      size_t row_size = dataset->num_numeric_features_ * sizeof(float);
      if (row_size > buffer_size) {
        buffer_size = row_size;
        buffer.resize(buffer_size);
      }
    for (int i = 0; i < dataset->num_data(); ++i) {
      read_cnt = reader->Read(buffer.data(), row_size);
      if (read_cnt != row_size) {
        Log::Fatal("Binary file error: row %d of raw data is incorrect, read count: %d", i, read_cnt);
      }
      mem_ptr = buffer.data();
      const float* tmp_ptr_raw_row = reinterpret_cast<const float*>(mem_ptr);
      for (int j = 0; j < dataset->num_features(); ++j) {
        int feat_ind = dataset->numeric_feature_map_[j];
        if (feat_ind >= 0) {
          dataset->raw_data_[feat_ind][i] = tmp_ptr_raw_row[feat_ind];
        }
      }
      mem_ptr += row_size;
    }
  }

  dataset->is_finish_load_ = true;
  return dataset.release();
}

Dataset* DatasetLoader::ConstructFromSampleData(double** sample_values,
                                                int** sample_indices, int num_col, const int* num_per_col,
                                                size_t total_sample_size, data_size_t num_data,
                                                const CategoryEncodingProvider* category_encoding_provider) {
  CheckSampleSize(total_sample_size, static_cast<size_t>(num_data));
  int num_total_features = num_col;
  if (Network::num_machines() > 1) {
    num_total_features = Network::GlobalSyncUpByMax(num_total_features);
  }
  // fill feature_names_ if not header
  if (category_encoding_provider != nullptr) {
    category_encoding_provider->ExtendFeatureNames(&feature_names_);
  }
  if (feature_names_.empty()) {
    for (int i = 0; i < num_total_features; ++i) {
      std::stringstream str_buf;
      str_buf << "Column_" << i;
      feature_names_.push_back(str_buf.str());
    }
  }
  std::vector<std::unique_ptr<BinMapper>> bin_mappers(num_total_features);
  if (!config_.max_bin_by_feature.empty()) {
    CHECK_EQ(static_cast<size_t>(num_col), config_.max_bin_by_feature.size());
    CHECK_GT(*(std::min_element(config_.max_bin_by_feature.begin(), config_.max_bin_by_feature.end())), 1);
  }

  // get forced split
  std::string forced_bins_path = config_.forcedbins_filename;
  std::vector<std::vector<double>> forced_bin_bounds = DatasetLoader::GetForcedBins(forced_bins_path, num_col, categorical_features_);
  const data_size_t filter_cnt = static_cast<data_size_t>(
    static_cast<double>(config_.min_data_in_leaf * total_sample_size) / num_data);
  if (Network::num_machines() == 1) {
    // if only one machine, find bin locally
    OMP_INIT_EX();
    #pragma omp parallel for schedule(guided)
    for (int i = 0; i < num_col; ++i) {
      OMP_LOOP_EX_BEGIN();
      if (ignore_features_.count(i) > 0) {
        bin_mappers[i] = nullptr;
        continue;
      }
      BinType bin_type = BinType::NumericalBin;
      if (categorical_features_.count(i)) {
        bin_type = BinType::CategoricalBin;
        bool feat_is_unconstrained = ((config_.monotone_constraints.size() == 0) || (config_.monotone_constraints[i] == 0));
        if (!feat_is_unconstrained) {
            Log::Fatal("The output cannot be monotone with respect to categorical features");
        }
      }
      bin_mappers[i].reset(new BinMapper());
      if (config_.max_bin_by_feature.empty()) {
        bin_mappers[i]->FindBin(sample_values[i], num_per_col[i], total_sample_size,
                                config_.max_bin, config_.min_data_in_bin, filter_cnt, config_.feature_pre_filter,
                                bin_type, config_.use_missing, config_.zero_as_missing,
                                forced_bin_bounds[i]);
      } else {
        bin_mappers[i]->FindBin(sample_values[i], num_per_col[i], total_sample_size,
                                config_.max_bin_by_feature[i], config_.min_data_in_bin,
                                filter_cnt, config_.feature_pre_filter, bin_type, config_.use_missing,
                                config_.zero_as_missing, forced_bin_bounds[i]);
      }
      OMP_LOOP_EX_END();
    }
    OMP_THROW_EX();
  } else {
    // if have multi-machines, need to find bin distributed
    // different machines will find bin for different features
    int num_machines = Network::num_machines();
    int rank = Network::rank();
    // start and len will store the process feature indices for different machines
    // machine i will find bins for features in [ start[i], start[i] + len[i] )
    std::vector<int> start(num_machines);
    std::vector<int> len(num_machines);
    int step = (num_total_features + num_machines - 1) / num_machines;
    if (step < 1) { step = 1; }

    start[0] = 0;
    for (int i = 0; i < num_machines - 1; ++i) {
      len[i] = std::min(step, num_total_features - start[i]);
      start[i + 1] = start[i] + len[i];
    }
    len[num_machines - 1] = num_total_features - start[num_machines - 1];
    OMP_INIT_EX();
    #pragma omp parallel for schedule(guided)
    for (int i = 0; i < len[rank]; ++i) {
      OMP_LOOP_EX_BEGIN();
      if (ignore_features_.count(start[rank] + i) > 0) {
        continue;
      }
      BinType bin_type = BinType::NumericalBin;
      if (categorical_features_.count(start[rank] + i)) {
        bin_type = BinType::CategoricalBin;
      }
      bin_mappers[i].reset(new BinMapper());
      if (num_col <= start[rank] + i) {
        continue;
      }
      if (config_.max_bin_by_feature.empty()) {
        bin_mappers[i]->FindBin(sample_values[start[rank] + i], num_per_col[start[rank] + i],
                                total_sample_size, config_.max_bin, config_.min_data_in_bin,
                                filter_cnt, config_.feature_pre_filter, bin_type, config_.use_missing, config_.zero_as_missing,
                                forced_bin_bounds[i]);
      } else {
        bin_mappers[i]->FindBin(sample_values[start[rank] + i], num_per_col[start[rank] + i],
                                total_sample_size, config_.max_bin_by_feature[start[rank] + i],
                                config_.min_data_in_bin, filter_cnt, config_.feature_pre_filter, bin_type, config_.use_missing,
                                config_.zero_as_missing, forced_bin_bounds[i]);
      }
      OMP_LOOP_EX_END();
    }
    OMP_THROW_EX();
    comm_size_t self_buf_size = 0;
    for (int i = 0; i < len[rank]; ++i) {
      if (ignore_features_.count(start[rank] + i) > 0) {
        continue;
      }
      self_buf_size += static_cast<comm_size_t>(bin_mappers[i]->SizesInByte());
    }
    std::vector<char> input_buffer(self_buf_size);
    auto cp_ptr = input_buffer.data();
    for (int i = 0; i < len[rank]; ++i) {
      if (ignore_features_.count(start[rank] + i) > 0) {
        continue;
      }
      bin_mappers[i]->CopyTo(cp_ptr);
      cp_ptr += bin_mappers[i]->SizesInByte();
      // free
      bin_mappers[i].reset(nullptr);
    }
    std::vector<comm_size_t> size_len = Network::GlobalArray(self_buf_size);
    std::vector<comm_size_t> size_start(num_machines, 0);
    for (int i = 1; i < num_machines; ++i) {
      size_start[i] = size_start[i - 1] + size_len[i - 1];
    }
    comm_size_t total_buffer_size = size_start[num_machines - 1] + size_len[num_machines - 1];
    std::vector<char> output_buffer(total_buffer_size);
    // gather global feature bin mappers
    Network::Allgather(input_buffer.data(), size_start.data(), size_len.data(), output_buffer.data(), total_buffer_size);
    cp_ptr = output_buffer.data();
    // restore features bins from buffer
    for (int i = 0; i < num_total_features; ++i) {
      if (ignore_features_.count(i) > 0) {
        bin_mappers[i] = nullptr;
        continue;
      }
      bin_mappers[i].reset(new BinMapper());
      bin_mappers[i]->CopyFrom(cp_ptr);
      cp_ptr += bin_mappers[i]->SizesInByte();
    }
  }
  auto dataset = std::unique_ptr<Dataset>(new Dataset(num_data));
  dataset->Construct(&bin_mappers, num_total_features, forced_bin_bounds, sample_indices, sample_values, num_per_col, num_col, total_sample_size, config_);
  if (dataset->has_raw()) {
    dataset->ResizeRaw(num_data);
  }
  dataset->set_feature_names(feature_names_);
  return dataset.release();
}


// ---- private functions ----

void DatasetLoader::CheckDataset(const Dataset* dataset, bool is_load_from_binary) {
  if (dataset->num_data_ <= 0) {
    Log::Fatal("Data file %s is empty", dataset->data_filename_.c_str());
  }
  if (dataset->feature_names_.size() != static_cast<size_t>(dataset->num_total_features_)) {
    Log::Fatal("Size of feature name error, should be %d, got %d", dataset->num_total_features_,
               static_cast<int>(dataset->feature_names_.size()));
  }
  bool is_feature_order_by_group = true;
  int last_group = -1;
  int last_sub_feature = -1;
  // if features are ordered, not need to use hist_buf
  for (int i = 0; i < dataset->num_features_; ++i) {
    int group = dataset->feature2group_[i];
    int sub_feature = dataset->feature2subfeature_[i];
    if (group < last_group) {
      is_feature_order_by_group = false;
    } else if (group == last_group) {
      if (sub_feature <= last_sub_feature) {
        is_feature_order_by_group = false;
        break;
      }
    }
    last_group = group;
    last_sub_feature = sub_feature;
  }
  if (!is_feature_order_by_group) {
    Log::Fatal("Features in dataset should be ordered by group");
  }

  if (is_load_from_binary) {
    if (dataset->max_bin_ != config_.max_bin) {
      Log::Fatal("Dataset max_bin %d != config %d", dataset->max_bin_, config_.max_bin);
    }
    if (dataset->min_data_in_bin_ != config_.min_data_in_bin) {
      Log::Fatal("Dataset min_data_in_bin %d != config %d", dataset->min_data_in_bin_, config_.min_data_in_bin);
    }
    if (dataset->use_missing_ != config_.use_missing) {
      Log::Fatal("Dataset use_missing %d != config %d", dataset->use_missing_, config_.use_missing);
    }
    if (dataset->zero_as_missing_ != config_.zero_as_missing) {
      Log::Fatal("Dataset zero_as_missing %d != config %d", dataset->zero_as_missing_, config_.zero_as_missing);
    }
    if (dataset->bin_construct_sample_cnt_ != config_.bin_construct_sample_cnt) {
      Log::Fatal("Dataset bin_construct_sample_cnt %d != config %d", dataset->bin_construct_sample_cnt_, config_.bin_construct_sample_cnt);
    }
    if ((dataset->max_bin_by_feature_.size() != config_.max_bin_by_feature.size()) ||
        !std::equal(dataset->max_bin_by_feature_.begin(), dataset->max_bin_by_feature_.end(),
            config_.max_bin_by_feature.begin())) {
      Log::Fatal("Dataset max_bin_by_feature does not match with config");
    }

    int label_idx = -1;
    if (Common::AtoiAndCheck(config_.label_column.c_str(), &label_idx)) {
      if (dataset->label_idx_ != label_idx) {
        Log::Fatal("Dataset label_idx %d != config %d", dataset->label_idx_, label_idx);
      }
    } else {
      Log::Info("Recommend use integer for label index when loading data from binary for sanity check.");
    }
  }
}

std::vector<std::string> DatasetLoader::LoadTextDataToMemory(const char* filename, const Metadata& metadata,
                                                             int rank, int num_machines, int* num_global_data,
                                                             std::vector<data_size_t>* used_data_indices,
                                                             CategoryEncodingProvider* category_encoding_provider) {
  TextReader<data_size_t> text_reader(filename, config_.header, config_.file_load_progress_interval_bytes);
  used_data_indices->clear();
  if (num_machines == 1 || config_.pre_partition) {
    // read all lines
    if (category_encoding_provider == nullptr) {
      *num_global_data = text_reader.ReadAllLines();
    } else {
      *num_global_data = text_reader.ReadAllLines(
        [category_encoding_provider] (const char* buffer, size_t size, data_size_t row_idx) {
          category_encoding_provider->AccumulateOneLineStat(buffer, size, row_idx);
        });
    }
  } else {  // need partition data
            // get query data
    const data_size_t* query_boundaries = metadata.query_boundaries();

    if (query_boundaries == nullptr) {
      // if not contain query data, minimal sample unit is one record
      std::function<bool(data_size_t, const char*, size_t)> filter_func = nullptr;
      if (category_encoding_provider == nullptr) {
        filter_func = [this, rank, num_machines](data_size_t, const char*, size_t) {
          if (random_.NextShort(0, num_machines) == rank) {
            return true;
          } else {
            return false;
          }
        };
      } else {
        filter_func = [this, rank, num_machines, category_encoding_provider](data_size_t row_idx, const char* buffer, size_t size) {
          category_encoding_provider->AccumulateOneLineStat(buffer, size, row_idx);
          if (random_.NextShort(0, num_machines) == rank) {
            return true;
          } else {
            return false;
          }
        };
      }
      *num_global_data = text_reader.ReadAndFilterLines(filter_func, used_data_indices);
    } else {
      // if contain query data, minimal sample unit is one query
      data_size_t num_queries = metadata.num_queries();
      data_size_t qid = -1;
      bool is_query_used = false;
      std::function<bool(data_size_t, const char*, size_t)> filter_func = nullptr;
      if (category_encoding_provider == nullptr) {
        filter_func = [this, rank, num_machines, &qid, &query_boundaries, &is_query_used, num_queries]
        (data_size_t line_idx, const char*, size_t) {
          if (qid >= num_queries) {
            Log::Fatal("Current query exceeds the range of the query file,\n"
                      "please ensure the query file is correct");
          }
          if (line_idx >= query_boundaries[qid + 1]) {
            // if is new query
            is_query_used = false;
            if (random_.NextShort(0, num_machines) == rank) {
              is_query_used = true;
            }
            ++qid;
          }
          return is_query_used;
        };
      } else {
        filter_func = [this, rank, num_machines, &qid, &query_boundaries,
          &is_query_used, num_queries, category_encoding_provider]
        (data_size_t line_idx, const char* buffer, size_t size) {
          if (qid >= num_queries) {
            Log::Fatal("Current query exceeds the range of the query file,\n"
                      "please ensure the query file is correct");
          }
          category_encoding_provider->AccumulateOneLineStat(buffer, size, line_idx);
          if (line_idx >= query_boundaries[qid + 1]) {
            // if is new query
            is_query_used = false;
            if (random_.NextShort(0, num_machines) == rank) {
              is_query_used = true;
            }
            ++qid;
          }
          return is_query_used;
        };
      }
      *num_global_data = text_reader.ReadAndFilterLines(filter_func, used_data_indices);
    }
  }
  return std::move(text_reader.Lines());
}

template <bool GET_SAMPLED_INDICES>
std::vector<std::string> DatasetLoader::SampleTextDataFromMemory(const std::vector<std::string>& data,
    std::vector<data_size_t>* sampled_indices) {
  int sample_cnt = config_.bin_construct_sample_cnt;
  if (static_cast<size_t>(sample_cnt) > data.size()) {
    sample_cnt = static_cast<int>(data.size());
  }
  std::vector<data_size_t>* sampled_indices_ptr;
  std::vector<data_size_t> local_sampled_indices;
  if (GET_SAMPLED_INDICES) {
    sampled_indices_ptr = sampled_indices;
  } else {
    sampled_indices_ptr = &local_sampled_indices;
  }
  *sampled_indices_ptr = random_.Sample(static_cast<int>(data.size()), sample_cnt);
  const std::vector<data_size_t>& sampled_indices_ref = *sampled_indices_ptr;
  std::vector<std::string> out(sampled_indices_ref.size());
  for (size_t i = 0; i < sampled_indices_ref.size(); ++i) {
    const size_t idx = sampled_indices_ref[i];
    out[i] = data[idx];
  }
  return out;
}

std::vector<std::string> DatasetLoader::SampleTextDataFromFile(const char* filename, const Metadata& metadata,
                                                               int rank, int num_machines, int* num_global_data,
                                                               std::vector<data_size_t>* used_data_indices,
                                                               std::vector<data_size_t>* sampled_indices,
                                                               CategoryEncodingProvider* category_encoding_provider) {
  const data_size_t sample_cnt = static_cast<data_size_t>(config_.bin_construct_sample_cnt);
  TextReader<data_size_t> text_reader(filename, config_.header, config_.file_load_progress_interval_bytes);
  std::vector<std::string> out_data;
  if (num_machines == 1 || config_.pre_partition) {
    if (category_encoding_provider == nullptr) {
      *num_global_data = static_cast<data_size_t>(text_reader.SampleFromFile<false>
        (&random_, sample_cnt, &out_data, sampled_indices, nullptr));
    } else {
      *num_global_data = static_cast<data_size_t>(text_reader.SampleFromFile<true>
        (&random_, sample_cnt, &out_data, sampled_indices,
        [category_encoding_provider] (const char* buffer, size_t size, data_size_t row_idx) {
          category_encoding_provider->AccumulateOneLineStat(buffer, size, row_idx);
        }));
    }
  } else {  // need partition data
            // get query data
    const data_size_t* query_boundaries = metadata.query_boundaries();
    if (query_boundaries == nullptr) {
      // if not contain query file, minimal sample unit is one record
      auto filter_return_func = [this, rank, num_machines] (data_size_t) {
        if (random_.NextShort(0, num_machines) == rank) {
          return true;
        } else {
          return false;
        }
      };
      std::function<bool(data_size_t, const char*, size_t)> filter_func = nullptr;
      if (category_encoding_provider == nullptr) {
        filter_func = [filter_return_func] (data_size_t row_idx, const char*, size_t) {
          return filter_return_func(row_idx);
        };

        *num_global_data = text_reader.SampleAndFilterFromFile<false>(
          filter_func, used_data_indices, &random_, sample_cnt, &out_data, sampled_indices);
      } else {
        filter_func = [filter_return_func, category_encoding_provider]
        (data_size_t row_idx, const char* buffer, size_t size) {
          category_encoding_provider->AccumulateOneLineStat(buffer, size, row_idx);
          return filter_return_func(row_idx);
        };

        *num_global_data = text_reader.SampleAndFilterFromFile<true>(
          filter_func, used_data_indices, &random_, sample_cnt, &out_data, sampled_indices);
      }
    } else {
      // if contain query file, minimal sample unit is one query
      data_size_t num_queries = metadata.num_queries();
      data_size_t qid = -1;
      bool is_query_used = false;

      auto filter_return_func =
        [this, rank, num_machines, &qid, &query_boundaries, &is_query_used, num_queries]
        (data_size_t line_idx) {
          if (qid >= num_queries) {
            Log::Fatal("Query id exceeds the range of the query file, "
                      "please ensure the query file is correct");
          }
          if (line_idx >= query_boundaries[qid + 1]) {
            // if is new query
            is_query_used = false;
            if (random_.NextShort(0, num_machines) == rank) {
              is_query_used = true;
            }
            ++qid;
          }
          return is_query_used;
        };

      std::function<bool(data_size_t, const char*, size_t)> filter_func = nullptr;
      if (category_encoding_provider == nullptr) {
        filter_func = [filter_return_func] (data_size_t row_idx, const char*, size_t) {
          return filter_return_func(row_idx);
        };
        *num_global_data = text_reader.SampleAndFilterFromFile<false>
          (filter_func, used_data_indices, &random_, sample_cnt, &out_data, sampled_indices);
      } else {
        filter_func = [filter_return_func, category_encoding_provider] (data_size_t row_idx, const char* buffer, size_t size) {
          category_encoding_provider->AccumulateOneLineStat(buffer, size, row_idx);
          return filter_return_func(row_idx);
        };
        *num_global_data = text_reader.SampleAndFilterFromFile<true>
          (filter_func, used_data_indices, &random_, sample_cnt, &out_data, sampled_indices);
      }
    }
  }
  return out_data;
}

void DatasetLoader::ConstructBinMappersFromTextData(int rank, int num_machines,
                                                    const std::vector<std::string>& sample_data,
                                                    const Parser* parser, Dataset* dataset,
                                                    const std::vector<data_size_t>& sampled_indices) {
  auto t1 = std::chrono::high_resolution_clock::now();
  std::vector<std::vector<double>> sample_values;
  std::vector<std::vector<int>> sample_indices;
  std::vector<std::pair<int, double>> oneline_features;
  double label;

  std::function<void(int)> parse_func = nullptr;
  if (sampled_indices.empty()) {
    parse_func = [&sample_data, &oneline_features, &label, parser] (int line_idx) {
      parser->ParseOneLine(sample_data[line_idx].c_str(), &oneline_features, &label);
    };
  } else {
    parse_func = [&sample_data, &oneline_features, &label, &sampled_indices, parser] (int line_idx) {
      parser->ParseOneLine(sample_data[line_idx].c_str(), &oneline_features, &label, sampled_indices[line_idx]);
    };
  }
  for (int i = 0; i < static_cast<int>(sample_data.size()); ++i) {
    oneline_features.clear();
    // parse features
    parse_func(i);
    for (std::pair<int, double>& inner_data : oneline_features) {
      if (static_cast<size_t>(inner_data.first) >= sample_values.size()) {
        sample_values.resize(inner_data.first + 1);
        sample_indices.resize(inner_data.first + 1);
      }
      if (std::fabs(inner_data.second) > kZeroThreshold || std::isnan(inner_data.second)) {
        sample_values[inner_data.first].emplace_back(inner_data.second);
        sample_indices[inner_data.first].emplace_back(i);
      }
    }
  }
  dataset->feature_groups_.clear();
  dataset->num_total_features_ = std::max(static_cast<int>(sample_values.size()), parser->NumFeatures());
  if (num_machines > 1) {
    dataset->num_total_features_ = Network::GlobalSyncUpByMax(dataset->num_total_features_);
  }
  if (!feature_names_.empty()) {
    CHECK_EQ(dataset->num_total_features_, static_cast<int>(feature_names_.size()));
  }
  // fill feature_names_ if not header, after dataset is constructed
  if (feature_names_.empty()) {
    for (int i = 0; i < dataset->num_total_features_; ++i) {
      std::stringstream str_buf;
      str_buf << "Column_" << i;
      feature_names_.push_back(str_buf.str());
    }
  }
  dataset->set_feature_names(feature_names_);
  if (!config_.max_bin_by_feature.empty()) {
    CHECK_EQ(static_cast<size_t>(dataset->num_total_features_), config_.max_bin_by_feature.size());
    CHECK_GT(*(std::min_element(config_.max_bin_by_feature.begin(), config_.max_bin_by_feature.end())), 1);
  }

  // get forced split
  std::string forced_bins_path = config_.forcedbins_filename;
  std::vector<std::vector<double>> forced_bin_bounds = DatasetLoader::GetForcedBins(forced_bins_path,
                                                                                    dataset->num_total_features_,
                                                                                    categorical_features_);

  // check the range of label_idx, weight_idx and group_idx
  CHECK(label_idx_ >= 0 && label_idx_ <= dataset->num_total_features_);
  CHECK(weight_idx_ < 0 || weight_idx_ < dataset->num_total_features_);
  CHECK(group_idx_ < 0 || group_idx_ < dataset->num_total_features_);

  std::vector<std::unique_ptr<BinMapper>> bin_mappers(dataset->num_total_features_);
  const data_size_t filter_cnt = static_cast<data_size_t>(
    static_cast<double>(config_.min_data_in_leaf* sample_data.size()) / dataset->num_data_);
  // start find bins
  if (num_machines == 1) {
    // if only one machine, find bin locally
    OMP_INIT_EX();
    #pragma omp parallel for schedule(guided)
    for (int i = 0; i < static_cast<int>(sample_values.size()); ++i) {
      OMP_LOOP_EX_BEGIN();
      if (ignore_features_.count(i) > 0) {
        bin_mappers[i] = nullptr;
        continue;
      }
      BinType bin_type = BinType::NumericalBin;
      if (categorical_features_.count(i)) {
        bin_type = BinType::CategoricalBin;
      }
      bin_mappers[i].reset(new BinMapper());
      if (config_.max_bin_by_feature.empty()) {
        bin_mappers[i]->FindBin(sample_values[i].data(), static_cast<int>(sample_values[i].size()),
                                sample_data.size(), config_.max_bin, config_.min_data_in_bin,
                                filter_cnt, config_.feature_pre_filter, bin_type, config_.use_missing, config_.zero_as_missing,
                                forced_bin_bounds[i]);
      } else {
        bin_mappers[i]->FindBin(sample_values[i].data(), static_cast<int>(sample_values[i].size()),
                                sample_data.size(), config_.max_bin_by_feature[i],
                                config_.min_data_in_bin, filter_cnt, config_.feature_pre_filter, bin_type, config_.use_missing,
                                config_.zero_as_missing, forced_bin_bounds[i]);
      }
      OMP_LOOP_EX_END();
    }
    OMP_THROW_EX();
  } else {
    // start and len will store the process feature indices for different machines
    // machine i will find bins for features in [ start[i], start[i] + len[i] )
    std::vector<int> start(num_machines);
    std::vector<int> len(num_machines);
    int step = (dataset->num_total_features_ + num_machines - 1) / num_machines;
    if (step < 1) { step = 1; }

    start[0] = 0;
    for (int i = 0; i < num_machines - 1; ++i) {
      len[i] = std::min(step, dataset->num_total_features_ - start[i]);
      start[i + 1] = start[i] + len[i];
    }
    len[num_machines - 1] = dataset->num_total_features_ - start[num_machines - 1];
    OMP_INIT_EX();
    #pragma omp parallel for schedule(guided)
    for (int i = 0; i < len[rank]; ++i) {
      OMP_LOOP_EX_BEGIN();
      if (ignore_features_.count(start[rank] + i) > 0) {
        continue;
      }
      BinType bin_type = BinType::NumericalBin;
      if (categorical_features_.count(start[rank] + i)) {
        bin_type = BinType::CategoricalBin;
      }
      bin_mappers[i].reset(new BinMapper());
      if (static_cast<int>(sample_values.size()) <= start[rank] + i) {
        continue;
      }
      if (config_.max_bin_by_feature.empty()) {
        bin_mappers[i]->FindBin(sample_values[start[rank] + i].data(),
                                static_cast<int>(sample_values[start[rank] + i].size()),
                                sample_data.size(), config_.max_bin, config_.min_data_in_bin,
                                filter_cnt, config_.feature_pre_filter, bin_type, config_.use_missing, config_.zero_as_missing,
                                forced_bin_bounds[i]);
      } else {
        bin_mappers[i]->FindBin(sample_values[start[rank] + i].data(),
                                static_cast<int>(sample_values[start[rank] + i].size()),
                                sample_data.size(), config_.max_bin_by_feature[i],
                                config_.min_data_in_bin, filter_cnt, config_.feature_pre_filter, bin_type,
                                config_.use_missing, config_.zero_as_missing, forced_bin_bounds[i]);
      }
      OMP_LOOP_EX_END();
    }
    OMP_THROW_EX();
    comm_size_t self_buf_size = 0;
    for (int i = 0; i < len[rank]; ++i) {
      if (ignore_features_.count(start[rank] + i) > 0) {
        continue;
      }
      self_buf_size += static_cast<comm_size_t>(bin_mappers[i]->SizesInByte());
    }
    std::vector<char> input_buffer(self_buf_size);
    auto cp_ptr = input_buffer.data();
    for (int i = 0; i < len[rank]; ++i) {
      if (ignore_features_.count(start[rank] + i) > 0) {
        continue;
      }
      bin_mappers[i]->CopyTo(cp_ptr);
      cp_ptr += bin_mappers[i]->SizesInByte();
      // free
      bin_mappers[i].reset(nullptr);
    }
    std::vector<comm_size_t> size_len = Network::GlobalArray(self_buf_size);
    std::vector<comm_size_t> size_start(num_machines, 0);
    for (int i = 1; i < num_machines; ++i) {
      size_start[i] = size_start[i - 1] + size_len[i - 1];
    }
    comm_size_t total_buffer_size = size_start[num_machines - 1] + size_len[num_machines - 1];
    std::vector<char> output_buffer(total_buffer_size);
    // gather global feature bin mappers
    Network::Allgather(input_buffer.data(), size_start.data(), size_len.data(), output_buffer.data(), total_buffer_size);
    cp_ptr = output_buffer.data();
    // restore features bins from buffer
    for (int i = 0; i < dataset->num_total_features_; ++i) {
      if (ignore_features_.count(i) > 0) {
        bin_mappers[i] = nullptr;
        continue;
      }
      bin_mappers[i].reset(new BinMapper());
      bin_mappers[i]->CopyFrom(cp_ptr);
      cp_ptr += bin_mappers[i]->SizesInByte();
    }
  }
  dataset->Construct(&bin_mappers, dataset->num_total_features_, forced_bin_bounds, Common::Vector2Ptr<int>(&sample_indices).data(),
                     Common::Vector2Ptr<double>(&sample_values).data(),
                     Common::VectorSize<int>(sample_indices).data(), static_cast<int>(sample_indices.size()), sample_data.size(), config_);
  if (dataset->has_raw()) {
    dataset->ResizeRaw(static_cast<int>(sample_data.size()));
  }

  auto t2 = std::chrono::high_resolution_clock::now();
  Log::Info("Construct bin mappers from text data time %.2f seconds",
            std::chrono::duration<double, std::milli>(t2 - t1) * 1e-3);
}

/*! \brief Extract local features from memory */
void DatasetLoader::ExtractFeaturesFromMemory(std::vector<std::string>* text_data, const Parser* parser, Dataset* dataset) {
  std::vector<std::pair<int, double>> oneline_features;
  double tmp_label = 0.0f;
  auto& ref_text_data = *text_data;
  std::vector<float> feature_row(dataset->num_features_);
  if (predict_fun_ == nullptr) {
    OMP_INIT_EX();
    // if doesn't need to prediction with initial model
    #pragma omp parallel for schedule(static) private(oneline_features) firstprivate(tmp_label, feature_row)
    for (data_size_t i = 0; i < dataset->num_data_; ++i) {
      OMP_LOOP_EX_BEGIN();
      const int tid = omp_get_thread_num();
      oneline_features.clear();
      // parser
      parser->ParseOneLine(ref_text_data[i].c_str(), &oneline_features, &tmp_label, i);
      // set label
      dataset->metadata_.SetLabelAt(i, static_cast<label_t>(tmp_label));
      // free processed line:
      ref_text_data[i].clear();
      // shrink_to_fit will be very slow in linux, and seems not free memory, disable for now
      // text_reader_->Lines()[i].shrink_to_fit();
      std::vector<bool> is_feature_added(dataset->num_features_, false);
      // push data
      for (auto& inner_data : oneline_features) {
        if (inner_data.first >= dataset->num_total_features_) { continue; }
        int feature_idx = dataset->used_feature_map_[inner_data.first];
        if (feature_idx >= 0) {
          is_feature_added[feature_idx] = true;
          // if is used feature
          int group = dataset->feature2group_[feature_idx];
          int sub_feature = dataset->feature2subfeature_[feature_idx];
          dataset->feature_groups_[group]->PushData(tid, sub_feature, i, inner_data.second);
          if (dataset->has_raw()) {
            feature_row[feature_idx] = static_cast<float>(inner_data.second);
          }
        } else {
          if (inner_data.first == weight_idx_) {
            dataset->metadata_.SetWeightAt(i, static_cast<label_t>(inner_data.second));
          } else if (inner_data.first == group_idx_) {
            dataset->metadata_.SetQueryAt(i, static_cast<data_size_t>(inner_data.second));
          }
        }
      }
      if (dataset->has_raw()) {
        for (size_t j = 0; j < feature_row.size(); ++j) {
          int feat_ind = dataset->numeric_feature_map_[j];
          if (feat_ind >= 0) {
            dataset->raw_data_[feat_ind][i] = feature_row[j];
          }
        }
      }
      dataset->FinishOneRow(tid, i, is_feature_added);
      OMP_LOOP_EX_END();
    }
    OMP_THROW_EX();
  } else {
    OMP_INIT_EX();
    // if need to prediction with initial model
    std::vector<double> init_score(dataset->num_data_ * num_class_);
    #pragma omp parallel for schedule(static) private(oneline_features) firstprivate(tmp_label, feature_row)
    for (data_size_t i = 0; i < dataset->num_data_; ++i) {
      OMP_LOOP_EX_BEGIN();
      const int tid = omp_get_thread_num();
      oneline_features.clear();
      // parser
      parser->ParseOneLine(ref_text_data[i].c_str(), &oneline_features, &tmp_label, i);
      // set initial score
      std::vector<double> oneline_init_score(num_class_);
      predict_fun_(oneline_features, oneline_init_score.data());
      for (int k = 0; k < num_class_; ++k) {
        init_score[k * dataset->num_data_ + i] = static_cast<double>(oneline_init_score[k]);
      }
      // set label
      dataset->metadata_.SetLabelAt(i, static_cast<label_t>(tmp_label));
      // free processed line:
      ref_text_data[i].clear();
      // shrink_to_fit will be very slow in Linux, and seems not free memory, disable for now
      // text_reader_->Lines()[i].shrink_to_fit();
      // push data
      std::vector<bool> is_feature_added(dataset->num_features_, false);
      for (auto& inner_data : oneline_features) {
        if (inner_data.first >= dataset->num_total_features_) { continue; }
        int feature_idx = dataset->used_feature_map_[inner_data.first];
        if (feature_idx >= 0) {
          is_feature_added[feature_idx] = true;
          // if is used feature
          int group = dataset->feature2group_[feature_idx];
          int sub_feature = dataset->feature2subfeature_[feature_idx];
          dataset->feature_groups_[group]->PushData(tid, sub_feature, i, inner_data.second);
          if (dataset->has_raw()) {
            feature_row[feature_idx] = static_cast<float>(inner_data.second);
          }
        } else {
          if (inner_data.first == weight_idx_) {
            dataset->metadata_.SetWeightAt(i, static_cast<label_t>(inner_data.second));
          } else if (inner_data.first == group_idx_) {
            dataset->metadata_.SetQueryAt(i, static_cast<data_size_t>(inner_data.second));
          }
        }
      }
      dataset->FinishOneRow(tid, i, is_feature_added);
      if (dataset->has_raw()) {
        for (size_t j = 0; j < feature_row.size(); ++j) {
          int feat_ind = dataset->numeric_feature_map_[j];
          if (feat_ind >= 0) {
            dataset->raw_data_[feat_ind][i] = feature_row[j];
          }
        }
      }
      OMP_LOOP_EX_END();
    }
    OMP_THROW_EX();
    // metadata_ will manage space of init_score
    dataset->metadata_.SetInitScore(init_score.data(), dataset->num_data_ * num_class_);
  }
  dataset->FinishLoad();
  // text data can be free after loaded feature values
  text_data->clear();
}

/*! \brief Extract local features from file */
void DatasetLoader::ExtractFeaturesFromFile(const char* filename, const Parser* parser,
                                            const std::vector<data_size_t>& used_data_indices, Dataset* dataset) {
  std::vector<double> init_score;
  if (predict_fun_ != nullptr) {
    init_score = std::vector<double>(dataset->num_data_ * num_class_);
  }
  std::function<void(data_size_t, const std::vector<std::string>&)> process_fun =
    [this, &init_score, &parser, &dataset]
  (data_size_t start_idx, const std::vector<std::string>& lines) {
    std::vector<std::pair<int, double>> oneline_features;
    double tmp_label = 0.0f;
    std::vector<float> feature_row(dataset->num_features_);
    OMP_INIT_EX();
    #pragma omp parallel for schedule(static) private(oneline_features) firstprivate(tmp_label, feature_row)
    for (data_size_t i = 0; i < static_cast<data_size_t>(lines.size()); ++i) {
      OMP_LOOP_EX_BEGIN();
      const int tid = omp_get_thread_num();
      oneline_features.clear();
      // parser
      parser->ParseOneLine(lines[i].c_str(), &oneline_features, &tmp_label, start_idx + i);
      // set initial score
      if (!init_score.empty()) {
        std::vector<double> oneline_init_score(num_class_);
        predict_fun_(oneline_features, oneline_init_score.data());
        for (int k = 0; k < num_class_; ++k) {
          init_score[k * dataset->num_data_ + start_idx + i] = static_cast<double>(oneline_init_score[k]);
        }
      }
      // set label
      dataset->metadata_.SetLabelAt(start_idx + i, static_cast<label_t>(tmp_label));
      std::vector<bool> is_feature_added(dataset->num_features_, false);
      // push data
      for (auto& inner_data : oneline_features) {
        if (inner_data.first >= dataset->num_total_features_) { continue; }
        int feature_idx = dataset->used_feature_map_[inner_data.first];
        if (feature_idx >= 0) {
          is_feature_added[feature_idx] = true;
          // if is used feature
          int group = dataset->feature2group_[feature_idx];
          int sub_feature = dataset->feature2subfeature_[feature_idx];
          dataset->feature_groups_[group]->PushData(tid, sub_feature, start_idx + i, inner_data.second);
          if (dataset->has_raw()) {
            feature_row[feature_idx] = static_cast<float>(inner_data.second);
          }
        } else {
          if (inner_data.first == weight_idx_) {
            dataset->metadata_.SetWeightAt(start_idx + i, static_cast<label_t>(inner_data.second));
          } else if (inner_data.first == group_idx_) {
            dataset->metadata_.SetQueryAt(start_idx + i, static_cast<data_size_t>(inner_data.second));
          }
        }
      }
      if (dataset->has_raw()) {
        for (size_t j = 0; j < feature_row.size(); ++j) {
          int feat_ind = dataset->numeric_feature_map_[j];
          if (feat_ind >= 0) {
            dataset->raw_data_[feat_ind][start_idx + i] = feature_row[j];
          }
        }
      }
      dataset->FinishOneRow(tid, start_idx + i, is_feature_added);
      OMP_LOOP_EX_END();
    }
    OMP_THROW_EX();
  };
  TextReader<data_size_t> text_reader(filename, config_.header, config_.file_load_progress_interval_bytes);
  if (!used_data_indices.empty()) {
    // only need part of data
    text_reader.ReadPartAndProcessParallel(used_data_indices, process_fun);
  } else {
    // need full data
    text_reader.ReadAllAndProcessParallel(process_fun);
  }

  // metadata_ will manage space of init_score
  if (!init_score.empty()) {
    dataset->metadata_.SetInitScore(init_score.data(), dataset->num_data_ * num_class_);
  }
  dataset->FinishLoad();
}


std::vector<std::vector<double>> DatasetLoader::GetForcedBins(std::string forced_bins_path, int num_total_features,
                                                              const std::unordered_set<int>& categorical_features) {
  std::vector<std::vector<double>> forced_bins(num_total_features, std::vector<double>());
  if (forced_bins_path != "") {
    std::ifstream forced_bins_stream(forced_bins_path.c_str());
    if (forced_bins_stream.fail()) {
      Log::Warning("Could not open %s. Will ignore.", forced_bins_path.c_str());
    } else {
      std::stringstream buffer;
      buffer << forced_bins_stream.rdbuf();
      std::string err;
      Json forced_bins_json = Json::parse(buffer.str(), &err);
      CHECK(forced_bins_json.is_array());
      std::vector<Json> forced_bins_arr = forced_bins_json.array_items();
      for (size_t i = 0; i < forced_bins_arr.size(); ++i) {
        int feature_num = forced_bins_arr[i]["feature"].int_value();
        CHECK_LT(feature_num, num_total_features);
        if (categorical_features.count(feature_num)) {
          Log::Warning("Feature %d is categorical. Will ignore forced bins for this feature.", feature_num);
        } else {
          std::vector<Json> bounds_arr = forced_bins_arr[i]["bin_upper_bound"].array_items();
          for (size_t j = 0; j < bounds_arr.size(); ++j) {
            forced_bins[feature_num].push_back(bounds_arr[j].number_value());
          }
        }
      }
      // remove duplicates
      for (int i = 0; i < num_total_features; ++i) {
        auto new_end = std::unique(forced_bins[i].begin(), forced_bins[i].end());
        forced_bins[i].erase(new_end, forced_bins[i].end());
      }
    }
  }
  return forced_bins;
}

}  // namespace LightGBM
