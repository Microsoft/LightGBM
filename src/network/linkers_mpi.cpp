/*!
 * Copyright (c) 2016-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2016-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifdef USE_MPI

#include "linkers.h"

#include <iostream>

namespace LightGBM {

Linkers::Linkers(Config) {
  is_init_ = false;
  int argc = 0;
  char**argv = nullptr;
  int flag = 0;
  MPI_SAFE_CALL(MPI_Initialized(&flag));  // test if MPI has been initialized
  if (!flag) {  // if MPI not started, start it
    MPI_SAFE_CALL(MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &flag));
  }
  MPI_SAFE_CALL(MPI_Comm_size(MPI_COMM_WORLD, &num_machines_));
  MPI_SAFE_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank_));
  // wait for all client start up
  MPI_SAFE_CALL(MPI_Barrier(MPI_COMM_WORLD));
  bruck_map_ = BruckMap::Construct(rank_, num_machines_);
  recursive_halving_map_ = RecursiveHalvingMap::Construct(rank_, num_machines_);
  is_init_ = true;
}

Linkers::~Linkers() {
  // Don't call MPI_Finalize() here: If the destructor was called because only this node had an exception, calling MPI_Finalize() will cause all nodes to hang.
  // Instead we will handle finalize/abort for MPI in main().
}

bool Linkers::IsMpiInitialized() {
  int is_mpi_init;
  MPI_SAFE_CALL(MPI_Initialized(&is_mpi_init));
  return is_mpi_init;
}

void Linkers::MpiFinalizeIfIsParallel() {
  if (IsMpiInitialized()) {
    Log::Debug("Finalizing MPI session.");
    MPI_SAFE_CALL(MPI_Finalize());
  }
}

void Linkers::MpiAbortIfIsParallel() {
  try {
    if (IsMpiInitialized()) {
      std::cerr << "Aborting MPI communication." << std::endl << std::flush;
      MPI_SAFE_CALL(MPI_Abort(MPI_COMM_WORLD, -1));;
    }
  }
  catch (...) {
    std::cerr << "Exception was raised before aborting MPI. Aborting process..." << std::endl << std::flush;
    abort();
  }
}

}  // namespace LightGBM
#endif  // USE_MPI
