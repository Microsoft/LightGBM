/*!
 * Copyright (c) 2016-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2016-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <LightGBM/network.h>

#include <LightGBM/utils/common.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "linkers.h"

namespace LightGBM {

// static member definition
THREAD_LOCAL int Network::num_machines_ = 1;
THREAD_LOCAL int Network::rank_ = 0;
THREAD_LOCAL std::unique_ptr<Linkers> Network::linkers_;
THREAD_LOCAL BruckMap Network::bruck_map_;
THREAD_LOCAL RecursiveHalvingMap Network::recursive_halving_map_;
THREAD_LOCAL std::vector<comm_size_t> Network::block_start_;
THREAD_LOCAL std::vector<comm_size_t>  Network::block_len_;
THREAD_LOCAL comm_size_t Network::buffer_size_ = 0;
THREAD_LOCAL std::vector<char> Network::buffer_;
THREAD_LOCAL ReduceScatterFunction Network::reduce_scatter_ext_fun_ = nullptr;
THREAD_LOCAL AllgatherFunction Network::allgather_ext_fun_ = nullptr;


void Network::Init(Config config) {
  if (config.num_machines > 1) {
    linkers_.reset(new Linkers(config));
    rank_ = linkers_->rank();
    num_machines_ = linkers_->num_machines();
    bruck_map_ = linkers_->bruck_map();
    recursive_halving_map_ = linkers_->recursive_halving_map();
    block_start_ = std::vector<comm_size_t>(num_machines_);
    block_len_ = std::vector<comm_size_t>(num_machines_);
    buffer_size_ = 1024 * 1024;
    buffer_.resize(buffer_size_);
    Log::Info("Local rank: %d, total number of machines: %d", rank_, num_machines_);
  }
}

void Network::Init(int num_machines, int rank,
                   ReduceScatterFunction reduce_scatter_ext_fun, AllgatherFunction allgather_ext_fun) {
  if (num_machines > 1) {
    rank_ = rank;
    num_machines_ = num_machines;
    block_start_ = std::vector<comm_size_t>(num_machines_);
    block_len_ = std::vector<comm_size_t>(num_machines_);
    buffer_size_ = 1024 * 1024;
    buffer_.resize(buffer_size_);
    reduce_scatter_ext_fun_ = reduce_scatter_ext_fun;
    allgather_ext_fun_ = allgather_ext_fun;
    Log::Info("Local rank: %d, total number of machines: %d", rank_, num_machines_);
  }
}

void Network::Dispose() {
  num_machines_ = 1;
  rank_ = 0;
  linkers_.reset(new Linkers());
  reduce_scatter_ext_fun_ = nullptr;
  allgather_ext_fun_ = nullptr;
}

void Network::Allreduce(char* input, comm_size_t input_size, int type_size, char* output, const ReduceFunction& reducer) {
  if (num_machines_ <= 1) {
    Log::Fatal("Please initialize the network interface first");
  }
  comm_size_t count = input_size / type_size;
  // if small package or small count , do it by all gather.(reduce the communication times.)
  if (count < num_machines_ || input_size < 4096) {
    AllreduceByAllGather(input, input_size, type_size, output, reducer);
    return;
  }
  // assign the blocks to every rank.
  comm_size_t step = (count + num_machines_ - 1) / num_machines_;
  if (step < 1) {
    step = 1;
  }
  block_start_[0] = 0;
  for (int i = 0; i < num_machines_ - 1; ++i) {
    block_len_[i] = std::min<comm_size_t>(step * type_size, input_size - block_start_[i]);
    block_start_[i + 1] = block_start_[i] + block_len_[i];
  }
  block_len_[num_machines_ - 1] = input_size - block_start_[num_machines_ - 1];
  // do reduce scatter
  ReduceScatter(input, input_size, type_size, block_start_.data(), block_len_.data(), output, input_size, reducer);
  // do all gather
  Allgather(output, block_start_.data(), block_len_.data(), output, input_size);
}

void Network::AllreduceByAllGather(char* input, comm_size_t input_size, int type_size, char* output, const ReduceFunction& reducer) {
  if (num_machines_ <= 1) {
    Log::Fatal("Please initialize the network interface first");
  }
  // assign blocks
  comm_size_t all_size = input_size * num_machines_;
  block_start_[0] = 0;
  block_len_[0] = input_size;
  for (int i = 1; i < num_machines_; ++i) {
    block_start_[i] = block_start_[i - 1] + block_len_[i - 1];
    block_len_[i] = input_size;
  }
  // need use buffer here, since size of "output" is smaller than size after all gather
  if (input_size*num_machines_ > buffer_size_) {
    buffer_size_ = input_size*num_machines_;
    buffer_.resize(buffer_size_);
  }

  Allgather(input, block_start_.data(), block_len_.data(), buffer_.data(), all_size);
  for (int i = 1; i < num_machines_; ++i) {
    reducer(buffer_.data() + block_start_[i], buffer_.data() + block_start_[0], type_size, input_size);
  }
  // copy back
  std::memcpy(output, buffer_.data(), input_size);
}

void Network::Allgather(char* input, comm_size_t send_size, char* output) {
  if (num_machines_ <= 1) {
    Log::Fatal("Please initialize the network interface first");
    return;
  }
  // assign blocks
  block_start_[0] = 0;
  block_len_[0] = send_size;
  for (int i = 1; i < num_machines_; ++i) {
    block_start_[i] = block_start_[i - 1] + block_len_[i - 1];
    block_len_[i] = send_size;
  }
  // start all gather
  Allgather(input, block_start_.data(), block_len_.data(), output, send_size * num_machines_);
}

void Network::Allgather(char* input, const comm_size_t* block_start, const comm_size_t* block_len, char* output, comm_size_t all_size) {
  if (num_machines_ <= 1) {
    Log::Fatal("Please initialize the network interface first");
  }
  if (allgather_ext_fun_ != nullptr) {
    return allgather_ext_fun_(input, block_len[rank_], block_start, block_len, num_machines_, output, all_size);
  }
  const comm_size_t kRingThreshold = 10 * 1024 * 1024;  // 10MB
  const int kRingNodeThreshold = 64;
  if (all_size > kRingThreshold && num_machines_ < kRingNodeThreshold) {
    // when num_machines is small and data is large
    AllgatherRing(input, block_start, block_len, output, all_size);
  } else if (recursive_halving_map_.is_power_of_2) {
    AllgatherRecursiveDoubling(input, block_start, block_len, output, all_size);
  } else {
    AllgatherBruck(input, block_start, block_len, output, all_size);
  }
}

void Network::AllgatherBruck(char* input, const comm_size_t* block_start, const comm_size_t* block_len, char* output, comm_size_t all_size) {
  comm_size_t write_pos = 0;
  // use output as receive buffer
  std::memcpy(output, input, block_len[rank_]);
  write_pos += block_len[rank_];
  int accumulated_block = 1;
  for (int i = 0; i < bruck_map_.k; ++i) {
    // get current local block size
    int cur_block_size = std::min(1 << i, num_machines_ - accumulated_block);
    // get out rank
    int out_rank = bruck_map_.out_ranks[i];
    // get in rank
    int in_rank = bruck_map_.in_ranks[i];
    // get send information
    comm_size_t need_send_len = 0;
    // get recv information
    comm_size_t need_recv_len = 0;
    for (int j = 0; j < cur_block_size; ++j) {
      need_send_len += block_len[(rank_ + j) % num_machines_];
      need_recv_len += block_len[(rank_ + accumulated_block + j) % num_machines_];
    }
    // send and recv at same time
    linkers_->SendRecv(out_rank, output, need_send_len, in_rank, output + write_pos, need_recv_len);
    write_pos += need_recv_len;
    accumulated_block += cur_block_size;
  }
  // rotate in-place
  std::reverse<char*>(output, output + all_size);
  std::reverse<char*>(output, output + block_start[rank_]);
  std::reverse<char*>(output + block_start[rank_], output + all_size);
}

void Network::AllgatherRecursiveDoubling(char* input, const comm_size_t* block_start, const comm_size_t* block_len, char* output, comm_size_t) {
  // use output as receive buffer
  std::memcpy(output + block_start[rank_], input, block_len[rank_]);
  for (int i = 0; i < bruck_map_.k; ++i) {
    // get current local block size
    int cur_step = 1 << i;
    const int vgroup = rank_ / cur_step;
    const int vrank = vgroup * cur_step;
    int target = rank_ + cur_step;
    int target_vrank = (vgroup + 1) * cur_step;
    if (vgroup & 1) {
      target = rank_ - cur_step;
      target_vrank = (vgroup - 1) * cur_step;
    }
    // get send information
    comm_size_t need_send_len = 0;
    // get recv information
    comm_size_t need_recv_len = 0;
    for (int j = 0; j < cur_step; ++j) {
      need_send_len += block_len[(vrank + j)];
      need_recv_len += block_len[(target_vrank + j)];
    }
    // send and recv at same time
    linkers_->SendRecv(target, output + block_start[vrank], need_send_len,
                       target, output + block_start[target_vrank], need_recv_len);
  }
}

void Network::AllgatherRing(char* input, const comm_size_t* block_start, const comm_size_t* block_len, char* output, comm_size_t) {
  // use output as receive buffer
  std::memcpy(output + block_start[rank_], input, block_len[rank_]);
  int out_rank = (rank_ + 1) % num_machines_;
  int in_rank = (rank_ - 1 + num_machines_) % num_machines_;
  int out_block = rank_;
  int in_block = in_rank;
  for (int i = 1; i < num_machines_; ++i) {
    // send and recv at same time
    linkers_->SendRecv(out_rank, output + block_start[out_block], block_len[out_block],
                       in_rank, output + block_start[in_block], block_len[in_block]);
    out_block = (out_block - 1 + num_machines_) % num_machines_;
    in_block = (in_block - 1 + num_machines_) % num_machines_;
  }
}

void Network::ReduceScatter(char* input, comm_size_t input_size, int type_size,
                            const comm_size_t* block_start, const comm_size_t* block_len, char* output,
                            comm_size_t output_size, const ReduceFunction& reducer) {
  if (num_machines_ <= 1) {
    Log::Fatal("Please initialize the network interface first");
  }
  if (reduce_scatter_ext_fun_ != nullptr) {
    return reduce_scatter_ext_fun_(input, input_size, type_size, block_start, block_len, num_machines_, output, output_size, reducer);
  }
  const comm_size_t kRingThreshold = 10 * 1024 * 1024;  // 10MB
  if (recursive_halving_map_.is_power_of_2 || input_size < kRingThreshold) {
    ReduceScatterRecursiveHalving(input, input_size, type_size, block_start, block_len, output, output_size, reducer);
  } else {
    ReduceScatterRing(input, input_size, type_size, block_start, block_len, output, output_size, reducer);
  }
}

void Network::ReduceScatterRecursiveHalving(char* input, comm_size_t input_size, int type_size,
                                            const comm_size_t* block_start, const comm_size_t* block_len, char* output,
                                            comm_size_t, const ReduceFunction& reducer) {
  if (!recursive_halving_map_.is_power_of_2) {
    if (recursive_halving_map_.type == RecursiveHalvingNodeType::Other) {
      // send local data to neighbor first
      linkers_->Send(recursive_halving_map_.neighbor, input, input_size);
    } else if (recursive_halving_map_.type == RecursiveHalvingNodeType::GroupLeader) {
      // receive neighbor data first
      int need_recv_cnt = input_size;
      linkers_->Recv(recursive_halving_map_.neighbor, output, need_recv_cnt);
      // reduce
      reducer(output, input, type_size, input_size);
    }
  }
  if (recursive_halving_map_.type != RecursiveHalvingNodeType::Other) {
    for (int i = 0; i < recursive_halving_map_.k; ++i) {
      // get target
      int target = recursive_halving_map_.ranks[i];
      comm_size_t send_block_start = recursive_halving_map_.send_block_start[i];
      comm_size_t recv_block_start = recursive_halving_map_.recv_block_start[i];
      // get send information
      comm_size_t send_size = 0;
      for (int j = 0; j < recursive_halving_map_.send_block_len[i]; ++j) {
        send_size += block_len[send_block_start + j];
      }
      // get recv information
      comm_size_t need_recv_cnt = 0;
      for (int j = 0; j < recursive_halving_map_.recv_block_len[i]; ++j) {
        need_recv_cnt += block_len[recv_block_start + j];
      }
      // send and recv at same time
      linkers_->SendRecv(target, input + block_start[send_block_start], send_size, target, output, need_recv_cnt);
      // reduce
      reducer(output, input + block_start[recv_block_start], type_size, need_recv_cnt);
    }
  }
  if (!recursive_halving_map_.is_power_of_2) {
    if (recursive_halving_map_.type == RecursiveHalvingNodeType::GroupLeader) {
      // send result to neighbor
      linkers_->Send(recursive_halving_map_.neighbor,
                     input + block_start[recursive_halving_map_.neighbor],
                     block_len[recursive_halving_map_.neighbor]);
    } else if (recursive_halving_map_.type == RecursiveHalvingNodeType::Other) {
      // receive result from neighbor
      int need_recv_cnt = block_len[rank_];
      linkers_->Recv(recursive_halving_map_.neighbor, output, need_recv_cnt);
      return;
    }
  }
  // copy result
  std::memcpy(output, input + block_start[rank_], block_len[rank_]);
}

void Network::ReduceScatterRing(char* input, comm_size_t, int type_size,
                                const comm_size_t* block_start, const comm_size_t* block_len, char* output,
                                comm_size_t, const ReduceFunction& reducer) {
  const int out_rank = (rank_ + 1) % num_machines_;
  const int in_rank = (rank_ - 1 + num_machines_) % num_machines_;
  int out_block = in_rank;
  int in_block = (in_rank - 1 + num_machines_) % num_machines_;
  for (int i = 1; i < num_machines_; ++i) {
    linkers_->SendRecv(out_rank, input + block_start[out_block], block_len[out_block],
                       in_rank, output, block_len[in_block]);
    reducer(output, input + block_start[in_block], type_size, block_len[in_block]);
    out_block = (out_block - 1 + num_machines_) % num_machines_;
    in_block = (in_block - 1 + num_machines_) % num_machines_;
  }
  std::memcpy(output, input + block_start[rank_], block_len[rank_]);
}

int Network::rank() {
  return rank_;
}

int Network::num_machines() {
  return num_machines_;
}

}  // namespace LightGBM
