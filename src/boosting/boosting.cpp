/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <LightGBM/boosting.h>

#include "dart.hpp"
#include "gbdt.h"
#include "goss.hpp"
#include "rf.hpp"
#include "mvs.hpp"

namespace LightGBM {

std::string GetBoostingTypeFromModelFile(const char* filename) {
  TextReader<size_t> model_reader(filename, true);
  std::string type = model_reader.first_line();
  return type;
}

bool Boosting::LoadFileToBoosting(Boosting* boosting, const char* filename) {
  auto start_time = std::chrono::steady_clock::now();
  if (boosting != nullptr) {
    TextReader<size_t> model_reader(filename, true);
    size_t buffer_len = 0;
    auto buffer = model_reader.ReadContent(&buffer_len);
    if (!boosting->LoadModelFromString(buffer.data(), buffer_len)) {
      return false;
    }
  }
  std::chrono::duration<double, std::milli> delta = (std::chrono::steady_clock::now() - start_time);
  Log::Debug("Time for loading model: %f seconds", 1e-3*delta);
  return true;
}

Boosting* Boosting::CreateBoosting(const std::string& type, const char* filename) {
  if (filename == nullptr || filename[0] == '\0') {
    if (type == std::string("gbdt")) {
      return new GBDT();
    } else if (type == std::string("dart")) {
      return new DART();
    } else if (type == std::string("goss")) {
      return new GOSS();
    } else if (type == std::string("rf")) {
      return new RF();
    } else if (type == std::string("mvs")) {
      return new MVS();
    } else {
      return nullptr;
    }
  } else {
    std::unique_ptr<Boosting> ret;
    if (GetBoostingTypeFromModelFile(filename) == std::string("tree")) {
      if (type == std::string("gbdt")) {
        ret.reset(new GBDT());
      } else if (type == std::string("dart")) {
        ret.reset(new DART());
      } else if (type == std::string("goss")) {
        ret.reset(new GOSS());
      } else if (type == std::string("rf")) {
        return new RF();
      } else if (type == std::string("mvs")) {
        return new MVS();
      } else {
        Log::Fatal("Unknown boosting type %s", type.c_str());
      }
      LoadFileToBoosting(ret.get(), filename);
    } else {
      Log::Fatal("Unknown model format or submodel type in model file %s", filename);
    }
    return ret.release();
  }
}

}  // namespace LightGBM
