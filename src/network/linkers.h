/*!
 * Copyright (c) 2016-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2016-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_SRC_NETWORK_LINKERS_H_
#define LIGHTGBM_SRC_NETWORK_LINKERS_H_

#include <LightGBM/config.h>
#include <LightGBM/meta.h>
#include <LightGBM/network.h>
#include <LightGBM/utils/common.h>

#include <string>
#include <algorithm>
#include <chrono>
#include <ctime>
#include <memory>
#include <thread>
#include <vector>

#ifdef USE_SOCKET
#include "socket_wrapper.hpp"
#endif

#ifdef USE_MPI
#include <mpi.h>
#define MPI_SAFE_CALL(mpi_return) CHECK((mpi_return) == MPI_SUCCESS)
#endif

namespace LightGBM {

/*!
* \brief A network basic communication wrapper.
* Will wrap low level communication methods, e.g. mpi, socket and so on.
* This class will wrap all linkers to other machines if needs
*/
class Linkers {
 public:
  Linkers() {
    is_init_ = false;
    rank_ = 0;
    num_machines_ = 1;
    #ifdef USE_SOCKET
    socket_timeout_ = 0;
    local_listen_port_ = 0;
    #endif
  }
  /*!
  * \brief Constructor
  * \param config Config of network settings
  */
  explicit Linkers(Config config);
  /*!
  * \brief Destructor
  */
  ~Linkers();
  /*!
  * \brief Recv data, blocking
  * \param rank Which rank will send data to local machine
  * \param data Pointer of receive data
  * \param len Recv size, will block until receive len size of data
  */
  inline void Recv(int rank, char* data, int len) const;

  inline void Recv(int rank, char* data, int64_t len) const;

  /*!
  * \brief Send data, blocking
  * \param rank Which rank local machine will send to
  * \param data Pointer of send data
  * \param len Send size
  */
  inline void Send(int rank, char* data, int len) const;

  inline void Send(int rank, char* data, int64_t len) const;
  /*!
  * \brief Send and Recv at same time, blocking
  * \param send_rank
  * \param send_data
  * \param send_len
  * \param recv_rank
  * \param recv_data
  * \param recv_len
  */
  inline void SendRecv(int send_rank, char* send_data, int send_len,
                       int recv_rank, char* recv_data, int recv_len);

  inline void SendRecv(int send_rank, char* send_data, int64_t send_len,
                       int recv_rank, char* recv_data, int64_t recv_len);
  /*!
  * \brief Get rank of local machine
  */
  inline int rank();
  /*!
  * \brief Get total number of machines
  */
  inline int num_machines();
  /*!
  * \brief Get Bruck map of this network
  */
  inline const BruckMap& bruck_map();
  /*!
  * \brief Get Recursive Halving map of this network
  */
  inline const RecursiveHalvingMap& recursive_halving_map();

  #ifdef USE_SOCKET
  /*!
  * \brief Bind local listen to port
  * \param port Local listen port
  */
  void TryBind(int port);
  /*!
  * \brief Set socket to rank
  * \param rank
  * \param socket
  */
  void SetLinker(int rank, const TcpSocket& socket);
  /*!
  * \brief Thread for listening
  * \param incoming_cnt Number of incoming machines
  */
  void ListenThread(int incoming_cnt);
  /*!
  * \brief Construct network topo
  */
  void Construct();
  /*!
  * \brief Parser machines information from file
  * \param machines
  * \param filename
  */
  void ParseMachineList(const std::string& machines, const std::string& filename);
  /*!
  * \brief Check one linker is connected or not
  * \param rank
  * \return True if linker is connected
  */
  bool CheckLinker(int rank);
  /*!
  * \brief Print connected linkers
  */
  void PrintLinkers();

  #endif  // USE_SOCKET

  #ifdef USE_MPI

  /*!
  * \brief Check if MPI has been initialized
  */
  static bool IsMpiInitialized();

  /*!
  * \brief Finalize the MPI session if it was initialized
  */
  static void MpiFinalizeIfIsParallel();

  /*!
  * \brief Abort the MPI session if it was initialized (called in case there was a error that needs abrupt ending)
  */
  static void MpiAbortIfIsParallel();

  #endif

 private:
  /*! \brief Rank of local machine */
  int rank_;
  /*! \brief Total number machines */
  int num_machines_;
  /*! \brief Bruck map */
  BruckMap bruck_map_;
  /*! \brief Recursive Halving map */
  RecursiveHalvingMap recursive_halving_map_;

  std::chrono::duration<double, std::milli> network_time_;

  bool is_init_;

  #ifdef USE_SOCKET
  /*! \brief use to store client ips */
  std::vector<std::string> client_ips_;
  /*! \brief use to store client ports */
  std::vector<int> client_ports_;
  /*! \brief time out for sockets, in minutes */
  int socket_timeout_;
  /*! \brief Local listen ports */
  int local_listen_port_;
  /*! \brief Linkers */
  std::vector<std::unique_ptr<TcpSocket>> linkers_;
  /*! \brief Local socket listener */
  std::unique_ptr<TcpSocket> listener_;
  #endif  // USE_SOCKET
};


inline int Linkers::rank() {
  return rank_;
}

inline int Linkers::num_machines() {
  return num_machines_;
}

inline const BruckMap& Linkers::bruck_map() {
  return bruck_map_;
}

inline const RecursiveHalvingMap& Linkers::recursive_halving_map() {
  return recursive_halving_map_;
}

inline void Linkers::Recv(int rank, char* data, int64_t len) const {
  int64_t used = 0;
  do {
    int cur_size = static_cast<int>(std::min<int64_t>(len - used, INT32_MAX));
    Recv(rank, data + used, cur_size);
    used += cur_size;
  } while (used < len);
}

inline void Linkers::Send(int rank, char* data, int64_t len) const {
  int64_t used = 0;
  do {
    int cur_size = static_cast<int>(std::min<int64_t>(len - used, INT32_MAX));
    Send(rank, data + used, cur_size);
    used += cur_size;
  } while (used < len);
}

inline void Linkers::SendRecv(int send_rank, char* send_data, int64_t send_len,
                              int recv_rank, char* recv_data, int64_t recv_len) {
  auto start_time = std::chrono::high_resolution_clock::now();
  std::thread send_worker(
    [this, send_rank, send_data, send_len]() {
    Send(send_rank, send_data, send_len);
  });
  Recv(recv_rank, recv_data, recv_len);
  send_worker.join();
  // wait for send complete
  auto end_time = std::chrono::high_resolution_clock::now();
  // output used time on each iteration
  network_time_ += std::chrono::duration<double, std::milli>(end_time - start_time);
}

#ifdef USE_SOCKET

inline void Linkers::Recv(int rank, char* data, int len) const {
  int recv_cnt = 0;
  while (recv_cnt < len) {
    int cur_recv = linkers_[rank]->Recv(data + recv_cnt,
      // len - recv_cnt
      std::min(len - recv_cnt, SocketConfig::kMaxReceiveSize));
    if (cur_recv <= 0) {
      // Connection closed by peer - expected during distributed training shutdown
      // Throw exception to exit cleanly instead of looping forever
      // See https://github.com/microsoft/LightGBM/issues/4074
      throw std::runtime_error("Network connection closed during distributed training");
    }
    recv_cnt += cur_recv;
  }
}

inline void Linkers::Send(int rank, char* data, int len) const {
  if (len <= 0) {
    return;
  }
  int send_cnt = 0;
  while (send_cnt < len) {
    int cur_send = linkers_[rank]->Send(data + send_cnt, len - send_cnt);
    if (cur_send <= 0) {
      // Connection closed by peer - expected during distributed training shutdown
      // Throw exception to exit cleanly instead of looping forever
      // See https://github.com/microsoft/LightGBM/issues/4074
      throw std::runtime_error("Network connection closed during distributed training");
    }
    send_cnt += cur_send;
  }
}

inline void Linkers::SendRecv(int send_rank, char* send_data, int send_len,
                              int recv_rank, char* recv_data, int recv_len) {
  auto start_time = std::chrono::high_resolution_clock::now();
  if (send_len < SocketConfig::kSocketBufferSize) {
    // if buffer is enough, send will non-blocking
    Send(send_rank, send_data, send_len);
    Recv(recv_rank, recv_data, recv_len);
  } else {
    // if buffer is not enough, use another thread to send, since send will be blocking
    std::thread send_worker(
      [this, send_rank, send_data, send_len]() {
      Send(send_rank, send_data, send_len);
    });
    Recv(recv_rank, recv_data, recv_len);
    send_worker.join();
  }
  // wait for send complete
  auto end_time = std::chrono::high_resolution_clock::now();
  // output used time on each iteration
  network_time_ += std::chrono::duration<double, std::milli>(end_time - start_time);
}

#endif  // USE_SOCKET

#ifdef USE_MPI

inline void Linkers::Recv(int rank, char* data, int len) const {
  MPI_Status status;
  int read_cnt = 0;
  while (read_cnt < len) {
    MPI_SAFE_CALL(MPI_Recv(data + read_cnt, len - read_cnt, MPI_BYTE, rank, MPI_ANY_TAG, MPI_COMM_WORLD, &status));
    int cur_cnt;
    MPI_SAFE_CALL(MPI_Get_count(&status, MPI_BYTE, &cur_cnt));
    read_cnt += cur_cnt;
  }
}

inline void Linkers::Send(int rank, char* data, int len) const {
  if (len <= 0) {
    return;
  }
  MPI_Status status;
  MPI_Request send_request;
  MPI_SAFE_CALL(MPI_Isend(data, len, MPI_BYTE, rank, 0, MPI_COMM_WORLD, &send_request));
  MPI_SAFE_CALL(MPI_Wait(&send_request, &status));
}

inline void Linkers::SendRecv(int send_rank, char* send_data, int send_len,
                              int recv_rank, char* recv_data, int recv_len) {
  MPI_Request send_request;
  // send first, non-blocking
  MPI_SAFE_CALL(MPI_Isend(send_data, send_len, MPI_BYTE, send_rank, 0, MPI_COMM_WORLD, &send_request));
  // then receive, blocking
  MPI_Status status;
  int read_cnt = 0;
  while (read_cnt < recv_len) {
    MPI_SAFE_CALL(MPI_Recv(recv_data + read_cnt, recv_len - read_cnt, MPI_BYTE, recv_rank, 0, MPI_COMM_WORLD, &status));
    int cur_cnt;
    MPI_SAFE_CALL(MPI_Get_count(&status, MPI_BYTE, &cur_cnt));
    read_cnt += cur_cnt;
  }
  // wait for send complete
  MPI_SAFE_CALL(MPI_Wait(&send_request, &status));
}

#endif  // USE_MPI
}  // namespace LightGBM
#endif   // LIGHTGBM_SRC_NETWORK_LINKERS_H_
