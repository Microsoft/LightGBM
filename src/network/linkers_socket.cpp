/*!
 * Copyright (c) 2016-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2016-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifdef USE_SOCKET

#include <LightGBM/config.h>
#include <LightGBM/utils/common.h>
#include <LightGBM/utils/text_reader.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "linkers.h"

namespace LightGBM {

Linkers::Linkers(Config config) {
  is_init_ = false;
  // start up socket
  TcpSocket::Startup();
  network_time_ = std::chrono::duration<double, std::milli>(0);
  num_machines_ = config.num_machines;
  local_listen_port_ = config.local_listen_port;
  socket_timeout_ = config.time_out;
  rank_ = -1;
  // parse clients from file
  ParseMachineList(config.machines, config.machine_list_filename);

  if (rank_ == -1) {
    // get ip list of local machine
    std::unordered_set<std::string> local_ip_list = TcpSocket::GetLocalIpList();
    // get local rank
    for (size_t i = 0; i < client_ips_.size(); ++i) {
      if (local_ip_list.count(client_ips_[i]) > 0 && client_ports_[i] == local_listen_port_) {
        rank_ = static_cast<int>(i);
        break;
      }
    }
  }
  if (rank_ == -1) {
    Log::Fatal("Machine list file doesn't contain the local machine");
  }
  // construct listener
  listener_ = std::unique_ptr<TcpSocket>(new TcpSocket());
  TryBind(local_listen_port_);

  for (int i = 0; i < num_machines_; ++i) {
    linkers_.push_back(nullptr);
  }

  // construct communication topo
  bruck_map_ = BruckMap::Construct(rank_, num_machines_);
  recursive_halving_map_ = RecursiveHalvingMap::Construct(rank_, num_machines_);

  // construct linkers
  Construct();
  // free listener
  listener_->Close();
  is_init_ = true;
}

Linkers::~Linkers() {
  if (is_init_) {
    for (size_t i = 0; i < linkers_.size(); ++i) {
      if (linkers_[i] != nullptr) {
        linkers_[i]->Close();
      }
    }
    TcpSocket::Finalize();
    Log::Info("Finished linking network in %f seconds", network_time_ * 1e-3);
  }
}

void Linkers::ParseMachineList(const std::string& machines, const std::string& filename) {
  std::vector<std::string> lines;
  if (machines.empty()) {
    TextReader<size_t> machine_list_reader(filename.c_str(), false);
    machine_list_reader.ReadAllLines();
    if (machine_list_reader.Lines().empty()) {
      Log::Fatal("Machine list file %s doesn't exist", filename.c_str());
    }
    lines = machine_list_reader.Lines();
  } else {
    lines = Common::Split(machines.c_str(), ',');
  }
  for (auto& line : lines) {
    line = Common::Trim(line);
    if (line.find("rank=") != std::string::npos) {
      std::vector<std::string> str_after_split = Common::Split(line.c_str(), '=');
      Common::Atoi(str_after_split[1].c_str(), &rank_);
      continue;
    }
    std::vector<std::string> str_after_split = Common::Split(line.c_str(), ' ');
    if (str_after_split.size() != 2) {
      str_after_split = Common::Split(line.c_str(), ':');
      if (str_after_split.size() != 2) {
        continue;
      }
    }
    if (client_ips_.size() >= static_cast<size_t>(num_machines_)) {
      Log::Warning("machine_list size is larger than the parameter num_machines, ignoring redundant entries");
      break;
    }
    str_after_split[0] = Common::Trim(str_after_split[0]);
    str_after_split[1] = Common::Trim(str_after_split[1]);
    client_ips_.push_back(str_after_split[0]);
    client_ports_.push_back(atoi(str_after_split[1].c_str()));
  }
  if (client_ips_.empty()) {
    Log::Fatal("Cannot find any ip and port.\n"
               "Please check machine_list_filename or machines parameter");
  }
  if (client_ips_.size() != static_cast<size_t>(num_machines_)) {
    Log::Warning("World size is larger than the machine_list size, change world size to %zu", client_ips_.size());
    num_machines_ = static_cast<int>(client_ips_.size());
  }
}

void Linkers::TryBind(int port) {
  Log::Info("Trying to bind port %d...", port);
  if (listener_->Bind(port)) {
    Log::Info("Binding port %d succeeded", port);
  } else {
    Log::Fatal("Binding port %d failed", port);
  }
}

void Linkers::SetLinker(int rank, const TcpSocket& socket) {
  linkers_[rank].reset(new TcpSocket(socket));
  // set timeout
  linkers_[rank]->SetTimeout(socket_timeout_ * 1000 * 60);
}

void Linkers::ListenThread(int incoming_cnt) {
  Log::Info("Listening...");
  char buffer[100];
  int connected_cnt = 0;
  while (connected_cnt < incoming_cnt) {
    // accept incoming socket
    TcpSocket handler = listener_->Accept();
    if (handler.IsClosed()) {
      continue;
    }
    // receive rank
    int read_cnt = 0;
    int size_of_int = static_cast<int>(sizeof(int));
    while (read_cnt < size_of_int) {
      int cur_read_cnt = handler.Recv(buffer + read_cnt, size_of_int - read_cnt);
      read_cnt += cur_read_cnt;
    }
    int* ptr_in_rank = reinterpret_cast<int*>(buffer);
    int in_rank = *ptr_in_rank;
    if (in_rank < 0 || in_rank >= num_machines_) {
      Log::Fatal("Invalid rank %d found during initialization of linkers. The world size is %d.", in_rank, num_machines_);
    }
    // add new socket
    SetLinker(in_rank, handler);
    ++connected_cnt;
  }
}

void Linkers::Construct() {
  // save ranks that need to connect with
  std::unordered_map<int, int> need_connect;
  for (int i = 0; i < num_machines_; ++i) {
    if (i != rank_) {
      need_connect[i] = 1;
    }
  }
  int incoming_cnt = 0;
  for (auto it = need_connect.begin(); it != need_connect.end(); ++it) {
    int machine_rank = it->first;
    if (machine_rank < rank_) {
      ++incoming_cnt;
    }
  }

  // start listener
  listener_->SetTimeout(socket_timeout_ * 1000 * 60);
  listener_->Listen(incoming_cnt);
  std::thread listen_thread(&Linkers::ListenThread, this, incoming_cnt);
  const int connect_fail_constant_factor = 20;
  const int connect_fail_retries_scale_factor = static_cast<int>(num_machines_ / connect_fail_constant_factor);
  const int connect_fail_retry_cnt = std::max(connect_fail_constant_factor, connect_fail_retries_scale_factor);
  const int connect_fail_retry_first_delay_interval = 200;  // 0.2 s
  const float connect_fail_retry_delay_factor = 1.3f;
  // start connect
  for (auto it = need_connect.begin(); it != need_connect.end(); ++it) {
    int out_rank = it->first;
    // let smaller rank connect to larger rank
    if (out_rank > rank_) {
      int connect_fail_delay_time = connect_fail_retry_first_delay_interval;
      for (int i = 0; i < connect_fail_retry_cnt; ++i) {
        TcpSocket cur_socket;
        if (cur_socket.Connect(client_ips_[out_rank].c_str(), client_ports_[out_rank])) {
          // send local rank
          cur_socket.Send(reinterpret_cast<const char*>(&rank_), sizeof(rank_));
          SetLinker(out_rank, cur_socket);
          break;
        } else {
          Log::Warning("Connecting to rank %d failed, waiting for %d milliseconds", out_rank, connect_fail_delay_time);
          cur_socket.Close();
          std::this_thread::sleep_for(std::chrono::milliseconds(connect_fail_delay_time));
          connect_fail_delay_time = static_cast<int>(connect_fail_delay_time * connect_fail_retry_delay_factor);
        }
      }
    }
  }
  // wait for listener
  listen_thread.join();
  // print connected linkers
  PrintLinkers();
}

bool Linkers::CheckLinker(int rank) {
  if (linkers_[rank] == nullptr || linkers_[rank]->IsClosed()) {
    return false;
  }
  return true;
}

void Linkers::PrintLinkers() {
  for (int i = 0; i < num_machines_; ++i) {
    if (CheckLinker(i)) {
      Log::Info("Connected to rank %d", i);
    }
  }
}

}  // namespace LightGBM

#endif  // USE_SOCKET
