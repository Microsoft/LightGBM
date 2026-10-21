/*!
 * Copyright (c) 2016-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2016-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_SRC_METRIC_GATHER_EVAL_DATA_HPP_
#define LIGHTGBM_SRC_METRIC_GATHER_EVAL_DATA_HPP_

#include <LightGBM/meta.h>
#include <LightGBM/network.h>

#include <cstring>
#include <vector>

namespace LightGBM {

inline void GatherEvalData(
    const label_t* label, const double* score,
    const label_t* weights, data_size_t num_data,
    int num_class,
    std::vector<label_t>& all_labels,
    std::vector<double>& all_scores,
    std::vector<label_t>& all_weights,
    bool& has_weights) {
  has_weights = (weights != nullptr);

  // Serialize: [num_data(4B)][num_class(4B)][labels(num_data*4B)]
  //            [scores(num_class*num_data*8B)][has_weights(1B)]
  //            [weights if has_weights(num_data*4B)]
  comm_size_t scores_bytes = static_cast<comm_size_t>(num_data) * num_class * static_cast<comm_size_t>(sizeof(double));
  comm_size_t labels_bytes = static_cast<comm_size_t>(num_data) * static_cast<comm_size_t>(sizeof(label_t));
  comm_size_t buf_size = 2 * sizeof(comm_size_t) + labels_bytes + scores_bytes + 1;
  if (has_weights) {
    buf_size += labels_bytes;
  }
  std::vector<char> local_buf(buf_size);
  char* ptr = local_buf.data();

  comm_size_t ndata = static_cast<comm_size_t>(num_data);
  std::memcpy(ptr, &ndata, sizeof(comm_size_t));
  ptr += sizeof(comm_size_t);

  comm_size_t nc = static_cast<comm_size_t>(num_class);
  std::memcpy(ptr, &nc, sizeof(comm_size_t));
  ptr += sizeof(comm_size_t);

  std::memcpy(ptr, label, labels_bytes);
  ptr += labels_bytes;

  std::memcpy(ptr, score, scores_bytes);
  ptr += scores_bytes;

  *ptr = has_weights ? static_cast<char>(1) : static_cast<char>(0);
  ptr += 1;

  if (has_weights) {
    std::memcpy(ptr, weights, labels_bytes);
  }

  // Exchange buffer sizes and compute offsets
  std::vector<comm_size_t> size_len = Network::GlobalArray(buf_size);
  int num_machines = Network::num_machines();
  std::vector<comm_size_t> size_start(num_machines, 0);
  for (int i = 1; i < num_machines; ++i) {
    size_start[i] = size_start[i - 1] + size_len[i - 1];
  }
  comm_size_t total_size = size_start[num_machines - 1] + size_len[num_machines - 1];
  std::vector<char> all_buf(total_size);
  Network::Allgather(local_buf.data(), size_start.data(), size_len.data(), all_buf.data(), total_size);

  // First pass: count total data points and check weights
  data_size_t total_data = 0;
  bool any_has_weights = false;
  for (int i = 0; i < num_machines; ++i) {
    const char* p = all_buf.data() + size_start[i];
    comm_size_t ndata_i;
    std::memcpy(&ndata_i, p, sizeof(comm_size_t));
    total_data += static_cast<data_size_t>(ndata_i);
    p += sizeof(comm_size_t);
    comm_size_t nc_i;
    std::memcpy(&nc_i, p, sizeof(comm_size_t));
    p += sizeof(comm_size_t) + ndata_i * sizeof(label_t) + ndata_i * nc_i * sizeof(double);
    if (*p) any_has_weights = true;
  }

  // Allocate output. all_scores is laid out class-major over total rows:
  // [c0_row0..c0_row{total-1} | c1_row0..c1_row{total-1} | ...]
  // This matches the single-machine score buffer layout, so callers can use
  // the same indexing scr[total_data * m + a] in both paths.
  data_size_t total_scores = static_cast<data_size_t>(total_data) * num_class;
  all_labels.resize(total_data);
  all_scores.resize(total_scores);
  if (any_has_weights) {
    all_weights.resize(total_data);
  }
  has_weights = any_has_weights;

  // Second pass: copy data
  data_size_t offset = 0;
  for (int i = 0; i < num_machines; ++i) {
    const char* p = all_buf.data() + size_start[i];
    comm_size_t ndata_i;
    std::memcpy(&ndata_i, p, sizeof(comm_size_t));
    p += sizeof(comm_size_t);
    comm_size_t nc_i;
    std::memcpy(&nc_i, p, sizeof(comm_size_t));
    p += sizeof(comm_size_t);

    if (ndata_i > 0) {
      std::memcpy(all_labels.data() + offset, p, ndata_i * sizeof(label_t));
      p += ndata_i * sizeof(label_t);

      // Per-worker scores arrive class-major over the worker's local rows:
      // [c0_row0..c0_row{ndata_i-1} | c1_row0..c1_row{ndata_i-1} | ...].
      // Re-stripe into the global class-major layout so each class block lives
      // contiguously across all workers.
      const double* worker_scores = reinterpret_cast<const double*>(p);
      for (comm_size_t m = 0; m < nc_i; ++m) {
        std::memcpy(
          all_scores.data() + static_cast<data_size_t>(m) * total_data + offset,
          worker_scores + static_cast<data_size_t>(m) * ndata_i,
          ndata_i * sizeof(double));
      }
      p += static_cast<comm_size_t>(ndata_i) * nc_i * static_cast<comm_size_t>(sizeof(double));

      bool has_w = (*p != 0);
      p += 1;

      if (has_w) {
        std::memcpy(all_weights.data() + offset, p, ndata_i * sizeof(label_t));
      } else if (any_has_weights) {
        std::fill(all_weights.data() + offset, all_weights.data() + offset + ndata_i, static_cast<label_t>(1.0f));
      }

      offset += ndata_i;
    }
  }
}

}  // namespace LightGBM

#endif  // LIGHTGBM_SRC_METRIC_GATHER_EVAL_DATA_HPP_
