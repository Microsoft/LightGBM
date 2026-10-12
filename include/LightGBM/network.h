/*!
 * Copyright (c) 2016-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2016-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_INCLUDE_LIGHTGBM_NETWORK_H_
#define LIGHTGBM_INCLUDE_LIGHTGBM_NETWORK_H_

#include <LightGBM/config.h>
#include <LightGBM/meta.h>
#include <LightGBM/utils/log.h>

#include <functional>
#include <memory>
#include <vector>

namespace LightGBM {

/*! \brief forward declaration */
class Linkers;

/*! \brief The network structure for all_gather */
class BruckMap {
 public:
  /*! \brief The communication times for one all gather operation */
  int k;
  /*! \brief in_ranks[i] means the incoming rank on i-th communication */
  std::vector<int> in_ranks;
  /*! \brief out_ranks[i] means the out rank on i-th communication */
  std::vector<int> out_ranks;
  BruckMap();
  explicit BruckMap(int n);
  /*!
  * \brief Create the object of bruck map
  * \param rank Rank of this machine
  * \param num_machines The total number of machines
  * \return The object of bruck map
  */
  static BruckMap Construct(int rank, int num_machines);
};

/*!
* \brief node type on recursive halving algorithm
*        When number of machines is not power of 2, need group machines into power of 2 group.
*        And we can let each group has at most 2 machines.
*        if the group only has 1 machine. this machine is the normal node
*        if the group has 2 machines, this group will have two type of nodes, one is the leader.
*        leader will represent this group and communication with others.
*/
enum RecursiveHalvingNodeType {
  Normal,  // normal node, 1 group only have 1 machine
  GroupLeader,  // leader of group when number of machines in this group is 2.
  Other  // non-leader machines in group
};

/*! \brief Network structure for recursive halving algorithm */
class RecursiveHalvingMap {
 public:
  /*! \brief Communication times for one recursive halving algorithm  */
  int k;
  /*! \brief Node type */
  RecursiveHalvingNodeType type;
  bool is_power_of_2;
  int neighbor;
  /*! \brief ranks[i] means the machines that will communicate with on i-th communication*/
  std::vector<int> ranks;
  /*! \brief send_block_start[i] means send block start index at i-th communication*/
  std::vector<int> send_block_start;
  /*! \brief send_block_start[i] means send block size at i-th communication*/
  std::vector<int> send_block_len;
  /*! \brief send_block_start[i] means recv block start index at i-th communication*/
  std::vector<int> recv_block_start;
  /*! \brief send_block_start[i] means recv block size at i-th communication*/
  std::vector<int> recv_block_len;

  RecursiveHalvingMap();

  RecursiveHalvingMap(int k, RecursiveHalvingNodeType _type, bool _is_power_of_2);

  /*!
  * \brief Create the object of recursive halving map
  * \param rank Rank of this machine
  * \param num_machines The total number of machines
  * \return The object of recursive halving map
  */
  static RecursiveHalvingMap Construct(int rank, int num_machines);
};

/*! \brief A static class that contains some collective communication algorithm */
class Network {
 public:
  /*!
  * \brief Initialize
  * \param config Config of network setting
  */
  static void Init(Config config);
  /*!
  * \brief Initialize
  */
  static void Init(int num_machines, int rank, ReduceScatterFunction reduce_scatter_ext_fun, AllgatherFunction allgather_ext_fun);
  /*! \brief Free this static class */
  static void Dispose();
  /*! \brief Get rank of this machine */
  static int rank();
  /*! \brief Get total number of machines */
  static int num_machines();

  /*!
  * \brief Perform all_reduce. if data size is small,
           will perform AllreduceByAllGather, else with call ReduceScatter followed allgather
  * \param input Input data
  * \param input_size The size of input data
  * \param type_size The size of one object in the reduce function
  * \param output Output result
  * \param reducer Reduce function
  */
  static void Allreduce(char* input, comm_size_t input_size, int type_size,
                        char* output, const ReduceFunction& reducer);

  /*!
  * \brief Perform all_reduce by using all_gather. it can be use to reduce communication time when data is small
  * \param input Input data
  * \param input_size The size of input data
  * \param type_size The size of one object in the reduce function
  * \param output Output result
  * \param reducer Reduce function
  */
  static void AllreduceByAllGather(char* input, comm_size_t input_size, int type_size, char* output,
                                   const ReduceFunction& reducer);

  /*!
  * \brief Performing all_gather by using Bruck algorithm.
           Communication times is O(log(n)), and communication cost is O(send_size * number_machine)
  *        It can be used when all nodes have same input size.
  * \param input Input data
  * \param send_size The size of input data
  * \param output Output result
  */
  static void Allgather(char* input, comm_size_t send_size, char* output);

  /*!
  * \brief Performing all_gather by using Bruck algorithm.
           Communication times is O(log(n)), and communication cost is O(all_size)
  *        It can be used when nodes have different input size.
  * \param input Input data
  * \param block_start The block start for different machines
  * \param block_len The block size for different machines
  * \param output Output result
  * \param all_size The size of output data
  */
  static void Allgather(char* input, const comm_size_t* block_start, const comm_size_t* block_len, char* output, comm_size_t all_size);

  /*!
  * \brief Perform reduce scatter by using recursive halving algorithm.
           Communication times is O(log(n)), and communication cost is O(input_size)
  * \param input Input data
  * \param input_size The size of input data
  * \param type_size The size of one object in the reduce function
  * \param block_start The block start for different machines
  * \param block_len The block size for different machines
  * \param output Output result
  * \param output_size size of output data
  * \param reducer Reduce function
  */
  static void ReduceScatter(char* input, comm_size_t input_size, int type_size,
                            const comm_size_t* block_start, const comm_size_t* block_len, char* output, comm_size_t output_size,
                            const ReduceFunction& reducer);

  template<class T>
  static T GlobalSyncUpByMin(T local) {
    T global = local;
    Allreduce(reinterpret_cast<char*>(&local),
              sizeof(local), sizeof(local),
              reinterpret_cast<char*>(&global),
              [] (const char* src, char* dst, int type_size, comm_size_t len) {
      comm_size_t used_size = 0;
      const T *p1;
      T *p2;
      while (used_size < len) {
        p1 = reinterpret_cast<const T *>(src);
        p2 = reinterpret_cast<T *>(dst);
        if (*p1 < *p2) {
          std::memcpy(dst, src, type_size);
        }
        src += type_size;
        dst += type_size;
        used_size += type_size;
      }
    });
    return global;
  }
  template<class T>
  static T GlobalSyncUpByMax(T local) {
    T global = local;
    Allreduce(reinterpret_cast<char*>(&local),
              sizeof(local), sizeof(local),
              reinterpret_cast<char*>(&global),
              [] (const char* src, char* dst, int type_size, comm_size_t len) {
      comm_size_t used_size = 0;
      const T *p1;
      T *p2;
      while (used_size < len) {
        p1 = reinterpret_cast<const T *>(src);
        p2 = reinterpret_cast<T *>(dst);
        if (*p1 > *p2) {
          std::memcpy(dst, src, type_size);
        }
        src += type_size;
        dst += type_size;
        used_size += type_size;
      }
    });
    return global;
  }

  template<class T>
  static T GlobalSyncUpBySum(T local) {
    T global = (T)0;
    Allreduce(reinterpret_cast<char*>(&local),
      sizeof(local), sizeof(local),
      reinterpret_cast<char*>(&global),
      [](const char* src, char* dst, int type_size, comm_size_t len) {
        comm_size_t used_size = 0;
        const T* p1;
        T* p2;
        while (used_size < len) {
          p1 = reinterpret_cast<const T*>(src);
          p2 = reinterpret_cast<T*>(dst);
          *p2 += *p1;
          src += type_size;
          dst += type_size;
          used_size += type_size;
        }
      });
    return static_cast<T>(global);
  }

  template<class T>
  static T GlobalSyncUpByMean(T local) {
    return static_cast<T>(GlobalSyncUpBySum(local) / num_machines_);
  }

  template<class T>
  static std::vector<T> GlobalSum(std::vector<T>* local) {
    std::vector<T> global(local->size(), 0);
    Allreduce(reinterpret_cast<char*>(local->data()),
              static_cast<comm_size_t>(sizeof(T) * local->size()), sizeof(T),
              reinterpret_cast<char*>(global.data()),
              [](const char* src, char* dst, int type_size, comm_size_t len) {
      comm_size_t used_size = 0;
      const T *p1;
      T *p2;
      while (used_size < len) {
        p1 = reinterpret_cast<const T *>(src);
        p2 = reinterpret_cast<T *>(dst);
        *p2 += *p1;
        src += type_size;
        dst += type_size;
        used_size += type_size;
      }
    });
    return global;
  }

  template<class T>
  static std::vector<T> GlobalArray(T local) {
    std::vector<T> global(num_machines_, 0);
    int type_size = sizeof(T);
    std::vector<comm_size_t> block_start(num_machines_);
    std::vector<comm_size_t> block_len(num_machines_, type_size);
    for (int i = 1; i < num_machines_; ++i) {
      block_start[i] = block_start[i - 1] + block_len[i - 1];
    }
    Allgather(reinterpret_cast<char*>(&local), block_start.data(), block_len.data(), reinterpret_cast<char*>(global.data()), type_size*num_machines_);
    return global;
  }

 private:
  static void AllgatherBruck(char* input, const comm_size_t* block_start, const comm_size_t* block_len, char* output, comm_size_t all_size);

  static void AllgatherRecursiveDoubling(char* input, const comm_size_t* block_start, const comm_size_t* block_len, char* output, comm_size_t all_size);

  static void AllgatherRing(char* input, const comm_size_t* block_start, const comm_size_t* block_len, char* output, comm_size_t all_size);

  static void ReduceScatterRecursiveHalving(char* input, comm_size_t input_size, int type_size,
                                            const comm_size_t* block_start, const comm_size_t* block_len, char* output, comm_size_t output_size,
                                            const ReduceFunction& reducer);

  static void ReduceScatterRing(char* input, comm_size_t input_size, int type_size,
                                const comm_size_t* block_start, const comm_size_t* block_len, char* output, comm_size_t output_size,
                                const ReduceFunction& reducer);

  /*! \brief Number of all machines */
  static THREAD_LOCAL int num_machines_;
  /*! \brief Rank of local machine */
  static THREAD_LOCAL int rank_;
  /*! \brief The network interface, provide send/recv functions  */
  static THREAD_LOCAL std::unique_ptr<Linkers> linkers_;
  /*! \brief Bruck map for all gather algorithm*/
  static THREAD_LOCAL BruckMap bruck_map_;
  /*! \brief Recursive halving map for reduce scatter */
  static THREAD_LOCAL RecursiveHalvingMap recursive_halving_map_;
  /*! \brief Buffer to store block start index */
  static THREAD_LOCAL std::vector<comm_size_t> block_start_;
  /*! \brief Buffer to store block size */
  static THREAD_LOCAL std::vector<comm_size_t> block_len_;
  /*! \brief Buffer  */
  static THREAD_LOCAL std::vector<char> buffer_;
  /*! \brief Size of buffer_ */
  static THREAD_LOCAL comm_size_t buffer_size_;
  /*! \brief Funcs*/
  static THREAD_LOCAL ReduceScatterFunction reduce_scatter_ext_fun_;
  static THREAD_LOCAL AllgatherFunction allgather_ext_fun_;
};

}  // namespace LightGBM

#endif   // LIGHTGBM_INCLUDE_LIGHTGBM_NETWORK_H_
