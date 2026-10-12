/*!
 * Copyright (c) 2016-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2016-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <LightGBM/application.h>

#include <iostream>
#include <string>

#ifdef USE_MPI
  #include "network/linkers.h"
#endif

int main(int argc, char** argv) {
  bool success = false;
  try {
    LightGBM::Application app(argc, argv);
    app.Run();

#ifdef USE_MPI
    LightGBM::Linkers::MpiFinalizeIfIsParallel();
#endif

    success = true;
  }
  catch (const std::exception& ex) {
    std::cerr << "Met Exceptions:" << std::endl;
    std::cerr << ex.what() << std::endl;
  }
  catch (const std::string& ex) {
    std::cerr << "Met Exceptions:" << std::endl;
    std::cerr << ex << std::endl;
  }
  catch (...) {
    std::cerr << "Unknown Exceptions" << std::endl;
  }

  if (!success) {
#ifdef USE_MPI
    LightGBM::Linkers::MpiAbortIfIsParallel();
#endif

    exit(-1);
  }
}
